import logging
from pytorch3d.io import IO
import torch
from pathlib import Path
from pytorch3d.renderer.mesh import TexturesUV as PT3DTexturesUV
from pytorch3d.renderer.mesh import TexturesVertex as PT3DTexturesVertex
from pytorch3d.structures.meshes import Meshes as PT3DMeshes
from pytorch3d.structures import packed_to_list as pt3d_packed_to_list
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
from od3d.cv.geometry.transform import proj3d2d, proj3d2d_broadcast
from od3d.cv.io import load_ply, save_ply
from enum import Enum
from typing import List
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from od3d.cv.geometry.transform import tform4x4, tform4x4_broadcast, inv_tform4x4, transf3d, add_homog_dim, \
    transf3d_broadcast, reproj2d3d_broadcast
from od3d.cv.visual.sample import sample_pxl2d_grid
from od3d.cv.geometry.grid import get_pxl2d_like, get_pxl2d
from typing import Union
import open3d as o3d
import numpy as np

class MESH_RENDER_MODALITIES(str, Enum):
    DEPTH = 'depth'
    MASK = 'mask'
    RGB = 'rgb'
    RGBA = 'rgba'
    FEATS = 'feats'
    MASK_VERTS_VSBL = 'mask_verts_vsbl'
    VERTS_NCDS = 'verts_ncds'

class MESH_RENDER_MODALITIES_GAUSSIAN_SPLAT(str, Enum):
    RGB = MESH_RENDER_MODALITIES.RGB
    FEATS = MESH_RENDER_MODALITIES.FEATS
    VERTS_NCDS = MESH_RENDER_MODALITIES.VERTS_NCDS

class Mesh:
    def __init__(self, verts, faces, rgb=None, feats=None):
        self.verts = verts
        self.faces = faces
        self.rgb = rgb
        self.feats = feats
        self.device = verts.device

    @staticmethod
    def convert_to_textureVertex(textures_uv: PT3DTexturesUV, meshes: PT3DMeshes) -> PT3DTexturesVertex:
        # note: this is a workaround, since the model textures_uv contains multiple values per vertex, but textures_vertex only one
        verts_colors_packed = torch.zeros_like(meshes.verts_packed())
        verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()  # (*)
        return PT3DTexturesVertex(pt3d_packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()))

    @staticmethod
    def load_from_file(fpath: Path, device='cpu', scale=1.):
        io = IO()
        mesh = io.load_mesh(fpath, device=device)
        verts = mesh[0].verts_list()[0] * scale
        faces = mesh[0].faces_list()[0]
        if mesh[0].textures is not None:
            verts_rgb = Mesh.convert_to_textureVertex(textures_uv=mesh[0].textures, meshes=mesh[0]).verts_features_list()[0]
        else:
            verts_rgb = None
        return Mesh(verts=verts, faces=faces, rgb=verts_rgb)

    @staticmethod
    def load_from_file_ply(fpath: Path):
        verts, faces = load_ply(fpath)
        return Mesh(verts=verts, faces=faces)


    def write_to_file(self, fpath: Path):
        logger.info(f'writing mesh to {fpath}')
        fpath.parent.mkdir(parents=True, exist_ok=True)
        save_ply(fpath, verts=self.verts, faces=self.faces)
    def verts_count(self):
        return self.verts.shape[0]

    @staticmethod
    def from_o3d(mesh_o3d: o3d.geometry.TriangleMesh, device='cpu'):

        vertices = torch.from_numpy(np.asarray(mesh_o3d.vertices)).to(dtype=torch.float, device=device)
        faces = torch.from_numpy(np.asarray(mesh_o3d.triangles)).to(dtype=torch.long, device=device)
        return Mesh(verts=vertices, faces=faces)
    # .TriangleMesh(vertices=vertices, triangles=triangles)

    def to_o3d(self):
        import open3d
        vertices = open3d.utility.Vector3dVector(self.verts.detach().cpu().numpy())
        faces = open3d.utility.Vector3iVector(self.faces.detach().cpu().numpy())
        o3d_obj_mesh = open3d.geometry.TriangleMesh(vertices=vertices, triangles=faces)
        return o3d_obj_mesh
    @staticmethod
    def create_sphere(center3d: torch.Tensor([0., 0., 0.]), radius: float = 1., device='cpu'):
        return Mesh.from_o3d(o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center3d.detach().cpu().numpy()), device=device)

    @property
    def verts_ncds(self):
        return (self.verts - self.verts.min(dim=0).values[None,]) / (self.verts.max(dim=0).values[None,] - self.verts.min(dim=0).values[None,])

    @staticmethod
    def create_plane_as_cone(center3d: torch.Tensor= torch.Tensor([0., 0., 0.]), radius:float=1., height:float=1., device='cpu'):
        R = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz((np.pi, 0., 0.))
        # note: height becomes larger with lower resolution
        plane3d_open3d = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height, resolution=100, split=1, create_uv_map=False).rotate(R=R, center=(0, 0, 0)).translate(center3d.detach().cpu().numpy())
        plane3d_open3d.paint_uniform_color([0.2, 0.2, 0.4])
        return Mesh.from_o3d(plane3d_open3d, device=device)

    # TODO: create ray
    # ray_range = scene_size
    # ray = open3d.geometry.TriangleMesh.create_arrow(cylinder_radius=1.0 * particle_size,
    #                                                 cone_radius=1.5 * particle_size, cylinder_height=ray_range,
    #                                                 cone_height=4.0 * particle_size)
    # ray.transform(inv_tform4x4(cam_tform4x4_obj).detach().cpu().numpy())

class Meshes(torch.nn.Module):
    def __init__(self, verts: List[torch.Tensor], faces: List[torch.Tensor], rgb: List[torch.Tensor]= None,
                 feats: List[torch.Tensor]=None, geodesic_prob_sigma=0.2,
                 gaussian_splat_enabled=False, gaussian_splat_opacity=0.7,
                 gaussian_splat_pts3d_size_rel_to_neighbor_dist =0.5,
                 pt3d_raster_perspective_correct=False):
        super().__init__()

        self.meshes_count = len(verts)
        self.verts = torch.nn.Parameter(torch.cat([_verts for _verts in verts], dim=0), requires_grad=False)
        self.faces = torch.nn.Parameter(torch.cat([_faces for _faces in faces], dim=0), requires_grad=False)
        self.device = self.verts.device

        self.gaussian_splat_enabled = gaussian_splat_enabled
        self.gaussian_splat_opacity = gaussian_splat_opacity
        self.gaussian_splat_pts3d_size_rel_to_neighbor_dist = gaussian_splat_pts3d_size_rel_to_neighbor_dist
        self.pt3d_raster_perspective_correct = pt3d_raster_perspective_correct

        self.geodesic_prob_sigma = geodesic_prob_sigma
        self._geodesic_dist = None

        self.verts_counts = [_verts.shape[0] for _verts in verts]
        self.faces_counts = [_faces.shape[0] for _faces in faces]
        self.verts_counts_acc_from_0 = [0] + [sum(self.verts_counts[:i+1]) for i in range(self.meshes_count)]
        self.faces_counts_acc_from_0 = [0] + [sum(self.faces_counts[:i+1]) for i in range(self.meshes_count)]
        self.verts_counts_max = max(self.verts_counts)
        self.faces_counts_max = max(self.faces_counts)

        self.mask_verts_not_padded = torch.ones(size=[len(self), self.verts_counts_max], dtype=torch.bool, device=self.verts.device)
        for i in range(len(self)):
            self.mask_verts_not_padded[i, self.verts_counts[i]:] = False

        if rgb is not None:
            self.rgb = torch.nn.Parameter(torch.cat([_rgb for _rgb in rgb], dim=0), requires_grad=False)
        else:
            self.rgb = None

        if feats is not None:
            self.feats = torch.nn.Parameter(torch.cat([_feats for _feats in feats], dim=0), requires_grad=True)
            self.feats_from_faces = torch.nn.Parameter(torch.cat([self.get_feats_with_mesh_id(mesh_id)[self.get_faces_with_mesh_id(mesh_id)] for mesh_id in range(len(self))], dim=0))
        else:
            self.feats = None
            self.feats_from_faces = None

        self.init_pt3d()
        self.pre_rendered_feats = None
        self.pre_rendered_modalities = {}

    def get_limits(self):
        meshes_limits = []
        for i in range(len(self)):
            mesh_verts = self.get_verts_with_mesh_id(mesh_id=i)
            mesh_limits = torch.stack([mesh_verts.min(dim=0)[0], mesh_verts.max(dim=0)[0]])
            meshes_limits.append(mesh_limits)
        meshes_limits = torch.stack(meshes_limits, dim=0)
        return meshes_limits

    def get_ranges(self):
        meshes_limits = self.get_limits()
        meshes_range = meshes_limits[:, 1, :] - meshes_limits[:, 0, :]
        return meshes_range

    def init_pt3d(self):
        self.pt3dmeshes = PT3DMeshes(
            verts=[self.get_verts_with_mesh_id(i) for i in range(self.meshes_count)],
            faces=[self.get_faces_with_mesh_id(i) for i in range(self.meshes_count)]
        )

    def write_to_file(self, fpath: Path):
        fpath.parent.mkdir(parents=True, exist_ok=True)
        save_ply(fpath, verts=self.verts, faces=self.faces)

    @dataclass
    class PreRendered():
        cams_tform4x4_obj: torch.Tensor
        cams_intr4x4: torch.Tensor
        imgs_sizes: torch.Tensor
        broadcast_batch_and_cams: bool
        meshes_ids: torch.Tensor
        down_sample_rate: float
        rendering: torch.Tensor

    @staticmethod
    def load_from_files(fpaths_meshes: List[Path], fpaths_meshes_tforms: List[Path] = None, device='cpu'):
        meshes = []
        for i, fpath_mesh in enumerate(fpaths_meshes):
            mesh = Mesh.load_from_file(fpath=fpath_mesh, device=device)
            if fpaths_meshes_tforms is not None and fpaths_meshes_tforms[i] is not None:
                mesh_tform = torch.load(fpaths_meshes_tforms[i]).to(device)
                mesh.verts = transf3d_broadcast(pts3d=mesh.verts, transf4x4=mesh_tform)
            meshes.append(mesh)
        return Meshes.load_from_meshes(meshes=meshes)

    @staticmethod
    def load_from_meshes(meshes: List[Mesh], device=None):
        if device is None:
            device = meshes[0].verts.device
        verts = [mesh.verts.to(device=device) for mesh in meshes]
        faces = [mesh.faces.to(device=device) for mesh in meshes]

        if meshes[0].rgb is not None:
            rgb = [mesh.rgb.to(device=device) for mesh in meshes]
        else:
            rgb = None
        return Meshes(verts=verts, faces=faces, rgb=rgb)

    @staticmethod
    def load_by_name(name: str, device='cpu', faces_count=None):
        if name == 'bunny':
            bunny_data = o3d.data.BunnyMesh()
            bunny_mesh_open3d = o3d.io.read_triangle_mesh(bunny_data.path)
            if faces_count is not None:
                bunny_mesh_open3d = bunny_mesh_open3d.simplify_quadric_decimation(faces_count)
            bunny_mesh = Meshes.load_from_meshes([Mesh.from_o3d(bunny_mesh_open3d, device=device)])
            bunny_rot = torch.Tensor(
                [[0., 0., 1., 0., ],
                 [1., 0., 0., 0., ],
                 [0., 1., 0., 0., ],
                 [0., 0., 0., 1., ]]).to(device=device)
            bunny_mesh.verts.data = transf3d_broadcast(pts3d=bunny_mesh.verts, transf4x4=bunny_rot)
            return bunny_mesh
        elif name == 'cuboid':
            from od3d.cv.geometry.primitives import Cuboids
            cuboids = Cuboids.create_dense_from_limits(limits=torch.Tensor([[[-1., -1., -1.], [1., 1., 1.]],]), device=device)
            return cuboids

        else:
            raise ValueError(f'Unknown mesh name: {name}')

    def __add__(self, meshes2):
        meshes = []
        for mesh_id in list(range(len(self))):
            meshes.append(self.get_meshes_with_ids(meshes_ids=[mesh_id]))
        for mesh_id in list(range(len(meshes2))):
            meshes.append(meshes2.get_meshes_with_ids(meshes_ids=[mesh_id]))
        return Meshes.load_from_meshes(meshes=meshes, device=self.device)

    @staticmethod
    def get_faces_from_verts(verts, ball_radius=0.3):
        import open3d
        import numpy as np
        from od3d.cv.geometry.transform import transf3d_broadcast, transf4x4_from_spherical

        verts_rot = transf3d_broadcast(pts3d=verts, transf4x4=transf4x4_from_spherical(azim=torch.Tensor([0.05]), elev=torch.Tensor([0.05]), theta=torch.Tensor([0.05]), dist=torch.Tensor([1.])))
        verts_centered = verts_rot - verts_rot.mean(dim=-2, keepdim=True)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(verts_centered)
        pcd.normals = open3d.utility.Vector3dVector(verts_centered)
        pcd.estimate_normals()
        # mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd)
        mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=pcd, radii=open3d.utility.DoubleVector([ball_radius]))
        faces = torch.from_numpy(np.asarray(mesh.triangles))
        return faces

    def __len__(self):
        return self.meshes_count
    def _apply(self, fn):
        super()._apply(fn)
        self.init_pt3d()

    def get_mesh_with_id(self, mesh_id):
        verts = self.get_verts_with_mesh_id(mesh_id)
        faces = self.get_faces_with_mesh_id(mesh_id)

        if self.rgb is None:
            rgb = None
        else:
            rgb = self.get_rgb_with_mesh_id(mesh_id)

        if self.feats is None:
            feats = None
        else:
            feats = self.get_feats_with_mesh_id(mesh_id=mesh_id)
        return Mesh(verts=verts, faces=faces, feats=feats, rgb=rgb)

    def get_meshes_with_ids(self, meshes_ids=None, clone=False):
        if meshes_ids == None:
            meshes_ids = list(range(len(self)))

        verts = [self.get_verts_with_mesh_id(mesh_id=mesh_id, clone=clone) for mesh_id in meshes_ids]
        faces = [self.get_faces_with_mesh_id(mesh_id=mesh_id, clone=clone) for mesh_id in meshes_ids]

        if self.rgb is None:
            rgb = None
        else:
            rgb = [self.get_rgb_with_mesh_id(mesh_id=mesh_id, clone=clone) for mesh_id in meshes_ids]
        if self.feats is None:
            feats = None
        else:
            feats = [self.get_feats_with_mesh_id(mesh_id=mesh_id, clone=clone) for mesh_id in meshes_ids]
        return Meshes(verts=verts, faces=faces, rgb=rgb, feats=feats)

    def get_geodesic_dist(self):
        if self._geodesic_dist is None:
            import gdist

            meshes_ids = list(range(len(self)))
            geodesic_dist = torch.ones(size=(self.verts.shape[0], self.verts.shape[0]), device=self.device) * torch.inf
            for mesh_id in meshes_ids:
                verts = self.get_verts_with_mesh_id(mesh_id=mesh_id)
                faces = self.get_faces_with_mesh_id(mesh_id=mesh_id)
                mesh_verts_geodesic_dist = gdist.local_gdist_matrix(
                    vertices=verts.detach().cpu().to(torch.float64).numpy(),
                    triangles=faces.cpu().detach().to(torch.int32).numpy(),
                    max_distance=99999,
                )
                # convert  scipy.sparse._csc.csc_matrix to torch.Tensor, fill sparse with torch.inf
                mesh_verts_geodesic_dist = torch.from_numpy(mesh_verts_geodesic_dist.toarray()).to(dtype=torch.float32,
                                                                                                   device=self.device)
                mesh_verts_geodesic_dist = mesh_verts_geodesic_dist / mesh_verts_geodesic_dist.max()
                mesh_verts_geodesic_dist[mesh_verts_geodesic_dist == 0] = torch.inf
                mesh_verts_geodesic_dist[torch.arange(mesh_verts_geodesic_dist.shape[0]).to(self.device), torch.arange(
                    mesh_verts_geodesic_dist.shape[0]).to(self.device)] = 0.

                # mesh_verts_geodesic_dist = torch.from_numpy(mesh_verts_geodesic_dist.toarray()).to(dtype=torch.float32, device=self.device)

                geodesic_dist[self.verts_counts_acc_from_0[mesh_id]: self.verts_counts_acc_from_0[mesh_id + 1],
                self.verts_counts_acc_from_0[mesh_id]: self.verts_counts_acc_from_0[
                    mesh_id + 1]] = mesh_verts_geodesic_dist

            # euclidean_dist = torch.cdist(self.verts[None,], self.verts[None,])[0]
            # geodesic_dist = torch.ones_like(euclidean_dist) * torch.inf
            # would need to get verts nearest neighbors and iteratively propagating geodesic_dist
            self._geodesic_dist = geodesic_dist

        return self._geodesic_dist


    def get_geodesic_prob(self):
        _geodesic_dist = self.get_geodesic_dist().clone()
        _geodesic_prob = torch.exp(input=- 0.5 * (_geodesic_dist / (self.geodesic_prob_sigma + 1e-10))**2)
        # replace inf with 0
        _geodesic_prob[torch.isinf(_geodesic_dist)] = 0.
        return _geodesic_prob



    def get_geodesic_prob_with_noise(self):
        geodesic_prob_with_noise = torch.eye(self.verts.shape[0]+1, device=self.device)
        geodesic_prob_with_noise[:-1, :-1] = self.get_geodesic_prob()
        return geodesic_prob_with_noise

    def get_verts_ncds_with_mesh_id(self, mesh_id):
        verts3d = self.get_verts_with_mesh_id(mesh_id)
        verts3d_ncds = (verts3d - verts3d.min(dim=0).values[None,]) / (
                verts3d.max(dim=0).values[None,] - verts3d.min(dim=0).values[None,])
        return verts3d_ncds

    def get_verts_ncds_cat_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        verts3d_ncds = []
        for mesh_id in mesh_ids:
            verts3d_ncds.append(self.get_verts_ncds_with_mesh_id(mesh_id=mesh_id))
        verts3d_ncds = torch.cat(verts3d_ncds, dim=0)
        return verts3d_ncds

    def get_verts_cat_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        verts3d = []
        for mesh_id in mesh_ids:
            verts3d.append(self.get_verts_with_mesh_id(mesh_id=mesh_id))
        verts3d = torch.cat(verts3d, dim=0)
        return verts3d

    def get_faces_cat_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        faces = []
        for mesh_id in mesh_ids:
            faces.append(self.get_faces_with_mesh_id(mesh_id=mesh_id))
        faces = torch.cat(faces, dim=0)
        return faces

    def get_verts_ncds_from_faces_with_mesh_id(self, mesh_id):
        verts3d_ncds = self.get_verts_ncds_with_mesh_id(mesh_id)
        feats_from_faces = verts3d_ncds[self.get_faces_with_mesh_id(mesh_id)]
        return feats_from_faces

    def get_feats_from_faces_with_mesh_id(self, mesh_id):
        # return self.feats_from_faces[self.faces_counts_acc_from_0[mesh_id]: self.faces_counts_acc_from_0[mesh_id+1]]
        return self.get_feats_with_mesh_id(mesh_id)[self.get_faces_with_mesh_id(mesh_id)]
    def get_rgb_with_mesh_id(self, mesh_id, clone=False):
        if not clone:
            return self.rgb[self.verts_counts_acc_from_0[mesh_id]: self.verts_counts_acc_from_0[mesh_id+1]]
        else:
            return self.rgb[self.verts_counts_acc_from_0[mesh_id]: self.verts_counts_acc_from_0[mesh_id + 1]].clone()

    def get_verts_with_mesh_id(self, mesh_id, clone=False):
        if not clone:
            return self.verts[self.verts_counts_acc_from_0[mesh_id]: self.verts_counts_acc_from_0[mesh_id+1]]
        else:
            return self.verts[self.verts_counts_acc_from_0[mesh_id]: self.verts_counts_acc_from_0[mesh_id + 1]].clone()

    def get_mesh_ids_for_verts(self):
        mesh_ids = torch.LongTensor(size=(0,)).to(device=self.device)
        for mesh_id in range(self.meshes_count):
            mesh_ids = torch.cat([mesh_ids, torch.LongTensor([mesh_id] * self.verts_counts[mesh_id]).to(device=self.device)], dim=0)
        return mesh_ids

    def get_feats_with_mesh_id(self, mesh_id, clone=False):
        if not clone:
            return self.feats[self.verts_counts_acc_from_0[mesh_id]: self.verts_counts_acc_from_0[mesh_id+1]]
        else:
            return self.feats[self.verts_counts_acc_from_0[mesh_id]: self.verts_counts_acc_from_0[mesh_id + 1]].clone()

    def get_faces_with_mesh_id(self, mesh_id, clone=False):
        if not clone:
            return self.faces[self.faces_counts_acc_from_0[mesh_id]: self.faces_counts_acc_from_0[mesh_id+1]]
        else:
            return self.faces[self.faces_counts_acc_from_0[mesh_id]: self.faces_counts_acc_from_0[mesh_id + 1]].clone()

    def get_faces_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_faces_with_pad(tensor=self.get_faces_with_mesh_id(mesh_id), mesh_id=mesh_id)
    def get_tensor_verts_with_pad(self, tensor, mesh_id):
        pad = torch.Size([self.verts_counts_max - self.verts_counts[mesh_id]])
        return torch.cat([tensor, torch.zeros(size=pad + tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)], dim=0)

    def get_tensor_faces_with_pad(self, tensor, mesh_id):
        pad = torch.Size([self.faces_counts_max - self.faces_counts[mesh_id]])
        return torch.cat([tensor, torch.zeros(size=pad + tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)], dim=0)

    def get_feats_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(tensor=self.get_feats_with_mesh_id(mesh_id), mesh_id=mesh_id)
    def get_verts_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(tensor=self.get_verts_with_mesh_id(mesh_id), mesh_id=mesh_id)

    def get_verts_ncds_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(tensor=self.get_verts_ncds_with_mesh_id(mesh_id), mesh_id=mesh_id)

    def get_rgb_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(tensor=self.get_rgb_with_mesh_id(mesh_id), mesh_id=mesh_id)

    # get_rgb_padded_with_mesh_id
    #def to(self, device):
    #    if self.device != device:
    #        self.verts = [v.to(device=device) for v in self.verts]
    #        self.faces = [f.to(device=device) for f in self.faces]
    #        if self.rgb is not None:
    #            self.rgb = [i.to(device=device) for i in self.rgb]
    #        if self.feats is not None:
    #            self.feats = [f.to(device=device) for f in self.feats]
    #            self.feats_from_faces = [self.feats[mesh_id][self.faces[mesh_id]] for mesh_id in range(len(self))]

    #           self.device = device
    def set_feats_cat_with_pad(self, feats):
        vts_ct_max = self.verts_counts_max
        device = self.verts.device
        self.feats = torch.nn.Parameter(torch.cat([feats[i*vts_ct_max: i*vts_ct_max + self.verts_counts[i]].to(device=device) for i in range(len(self))], dim=0), requires_grad=True)
        self.feats_from_faces = torch.nn.Parameter(torch.cat([self.get_feats_with_mesh_id(mesh_id)[self.get_faces_with_mesh_id(mesh_id)] for mesh_id in range(len(self))], dim=0))

    def set_feats_cat(self, feats):
        self.feats = torch.nn.Parameter(feats, requires_grad=True)
        self.feats_from_faces = torch.nn.Parameter(torch.cat([self.get_feats_with_mesh_id(mesh_id)[self.get_faces_with_mesh_id(mesh_id)] for mesh_id in range(len(self))], dim=0))


    def get_verts_ncds_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        return torch.stack([self.get_verts_ncds_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids], dim=0)

    def get_rgb_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        return torch.stack([self.get_rgb_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids], dim=0)

    def get_verts_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        return torch.stack([self.get_verts_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids], dim=0)

    def get_feats_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        return torch.stack([self.get_feats_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids], dim=0)

    def get_faces_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))
        return torch.stack([self.get_faces_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids], dim=0)

    #def add_feats_cat(self, feats):
    #    raise Not I
    #    self.feats = [feats[self.verts_counts_acc_from_0[i] : self.verts_counts_acc_from_0[i+1]].to(device=self.device) for i in range(len(self))]
    #    self.feats_from_faces = [self.feats[mesh_id][self.faces[mesh_id]] for mesh_id in range(len(self))]

    # def add_feats
    #def verts_stacked(self, mesh_ids: list=None):
    #    if mesh_ids == None:
    #        mesh_ids = list(range(len(self)))

        #verts_stacked = torch.zeros(size=(len(mesh_ids), self.verts_counts_max(), 3), device=self.device)
        #for mesh_id in mesh_ids:
        #    verts_stacked[mesh_id, : len(self.verts[mesh_id])] = self.verts[mesh_id]

    #    return torch.stack([self.verts[mesh_id] for mesh_id in mesh_ids], dim=0)

    #def feats_cat(self, mesh_ids: list=None):
    #    if mesh_ids == None:
    #        mesh_ids = list(range(len(self)))
    #
    #    return torch.cat([self.feats[mesh_id] for mesh_id in mesh_ids], dim=0)

    #def get_feats_ids_stacked(self, mesh_ids: list=None):
    #    if mesh_ids == None:
    #        mesh_ids = list(range(len(self)))

    #    device = self.verts.device
    #    return torch.stack([torch.arange(mesh_id*self.verts_counts_max, (mesh_id+1)*self.verts_counts_max, device=device) for mesh_id in mesh_ids], dim=0)

    #def get_verts_and_noise_ids_cat(self, mesh_ids: list=None, count_noise_ids=5):
    #   if mesh_ids == None:
    #        mesh_ids = list(range(len(self)))

    #    device = self.verts.device
    #    noise_ids = torch.ones(size=(count_noise_ids,), dtype=torch.long, device=device) * self.verts_counts_acc_from_0[-1]
    #    verts_ids = [torch.arange(self.verts_counts_acc_from_0[mesh_id], self.verts_counts_acc_from_0[mesh_id+1], device=device) for mesh_id in mesh_ids]
    #    return torch.cat([torch.cat([verts_ids[i], noise_ids], dim=0) for i in range(len(mesh_ids))], dim=0)

    def get_verts_and_noise_ids_stacked(self, mesh_ids: list=None, count_noise_ids=5):
        if mesh_ids == None:
            mesh_ids = list(range(len(self)))

        device = self.verts.device
        noise_ids = torch.ones(size=(count_noise_ids,), dtype=torch.long, device=device) * self.verts_counts_acc_from_0[-1]
        verts_ids = [torch.arange(self.verts_counts_acc_from_0[mesh_id], self.verts_counts_acc_from_0[mesh_id] + self.verts_counts_max, device=device) for mesh_id in mesh_ids]
        return torch.stack([torch.cat([verts_ids[i], noise_ids], dim=0) for i in range(len(mesh_ids))], dim=0)

    def normals3d(self, meshes_ids: Union[torch.LongTensor, List]=None):
        """
            Args:
                meshes_ids (Union[torch.LongTensor, List]): len(mesh_ids) == B
            Returns:
                normals3d (torch.Tensor): BxNx3
        """

        if isinstance(meshes_ids, List):
            meshes_ids = torch.LongTensor(meshes_ids)

        if isinstance(meshes_ids, torch.LongTensor):
            meshes_ids = meshes_ids.clone()

        pt3dmeshes = self.pt3dmeshes[meshes_ids]
        return pt3dmeshes.verts_normals_padded()


    def verts2d(self, cams_tform4x4_obj, cams_intr4x4, imgs_sizes, mesh_ids: Union[torch.LongTensor, List], down_sample_rate=1., broadcast_batch_and_cams=False):
        """
            Args:
                cams_tform4x4_obj (torch.Tensor): Bx4x4
                cams_intr4x4 (torch.Tensor): Bx4x4
                imgs_sizes (torch.Tensor): Bx2 / 2 (height, width)
                mesh_ids (list): len(mesh_ids) == B
            Returns:
                verts2d (torch.Tensor): BxNx2
        """
        #meshes_count = mesh_ids.shape[0]
        # cams_count = cams_tform4x4_obj.shape[0]

        if isinstance(mesh_ids, List):
            mesh_ids = torch.LongTensor(mesh_ids)

        if isinstance(mesh_ids, torch.LongTensor):
            mesh_ids = mesh_ids.clone()

        meshes_count = mesh_ids.shape[0]
        if cams_tform4x4_obj.dim() == 4:
            cams_count = cams_tform4x4_obj.shape[1]
        elif cams_tform4x4_obj.dim() == 3:
            cams_count = cams_tform4x4_obj.shape[0]
        else:
            raise ValueError(f'Set `cams_tform4x4_obj.dim()` must be 3 or 4')

        if broadcast_batch_and_cams:
            mesh_ids = mesh_ids
            if cams_tform4x4_obj.dim() == 3:
                cams_tform4x4_obj = cams_tform4x4_obj[None, :]
            if cams_intr4x4.dim() == 3:
                cams_intr4x4 = cams_intr4x4[None, :]
            cams_tform4x4_obj = cams_tform4x4_obj.expand(meshes_count, cams_count, 4, 4).reshape(-1, 4, 4)
            cams_intr4x4 = cams_intr4x4.expand(meshes_count, cams_count, 4, 4).reshape(-1, 4, 4)
            mesh_ids = mesh_ids[:, None].expand(meshes_count, cams_count).reshape(-1)
            #render_count = meshes_count * cams_count
        else:
            if meshes_count != cams_count:
                raise ValueError(f'Set `broadcast_batch_and_cams=True` to allow different number of cameras and meshes')

        B = cams_tform4x4_obj.shape[0]
        #if imgs_sizes.dim() == 2:
        #    imgs_sizes = imgs_sizes[None,].expand(B, imgs_sizes.shape[0], imgs_sizes.shape[1])
        cams_proj4x4_obj = torch.bmm(cams_intr4x4, cams_tform4x4_obj)
        verts3d = self.get_verts_stacked_with_mesh_ids(mesh_ids=mesh_ids)
        verts2d = proj3d2d_broadcast(verts3d, proj4x4=cams_proj4x4_obj[:, None])
        mask_verts_vsbl = self.render_feats(cams_tform4x4_obj=cams_tform4x4_obj, cams_intr4x4=cams_intr4x4, imgs_sizes=imgs_sizes, meshes_ids=mesh_ids, modality=MESH_RENDER_MODALITIES.MASK_VERTS_VSBL, down_sample_rate=down_sample_rate)
        mask_verts_vsbl *= (verts2d <= (imgs_sizes[None, None] - 1)).all(dim=-1)
        mask_verts_vsbl *= (verts2d >= 0).all(dim=-1)
        verts2d[~mask_verts_vsbl] = 0
        # verts2d.clamp()

        if broadcast_batch_and_cams:
            verts2d = verts2d.reshape(meshes_count, cams_count, *verts2d.shape[1:])
            mask_verts_vsbl = mask_verts_vsbl.reshape(meshes_count, cams_count, *mask_verts_vsbl.shape[1:])

        verts2d /= down_sample_rate

        return verts2d, mask_verts_vsbl

    def show(self, fpath: Path = None, return_visualization=False, viewpoints_count=1, meshes_add_translation=True):
        from od3d.cv.visual.show import show_scene
        return show_scene(meshes=self, fpath=fpath, return_visualization=return_visualization, viewpoints_count=viewpoints_count, meshes_add_translation=meshes_add_translation)

    """
    def show(self, pts3d=[], meshes_ids=None):
        from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
        from pytorch3d.structures import Pointclouds
        if meshes_ids is None:
            meshes_ids = list(range(len(self.pt3dmeshes)))
        pcls = Pointclouds(points=pts3d)
        verts = Pointclouds(points=self.get_verts_stacked_with_mesh_ids(mesh_ids=meshes_ids))
        fig = plot_scene({
            "Meshes":
                {
                    # **{f"mesh{i + 1}": self.pt3dmeshes[i] for i in meshes_ids},
                    **{f"verts{i + 1}": verts[i] for i in meshes_ids},
                    **{f"pcl{i + 1}": pcls[i] for i in range(len(pcls))}
                }
        }, axis_args=AxisArgs(backgroundcolor="rgb(200, 200, 230)", showgrid=True, zeroline=True, showline=True,
                                   showaxeslabels=True, showticklabels=True))
        fig.show()
        input('bla')
    """

    def del_pre_rendered(self):
        self.pre_rendered_modalities.clear()
        torch.cuda.empty_cache()

    def visualize(self, pcl=None):
        from od3d.cv.visual.show import show_imgs
        import numpy as np

        device = self.verts.device
        cxy = 250.
        fxy = 500.
        down_sample_rate = 2.
        imgs_sizes = torch.LongTensor([512, 512]).to(device=device)

        azim = torch.linspace(start=eval('-np.pi / 2'), end=eval('np.pi / 2'),
                              steps=5).to(
            device=device)  # 12
        elev = torch.linspace(start=eval('np.pi / 4'), end=eval('np.pi / 4'),
                              steps=1).to(
            device=device)  # start=-torch.pi / 6, end=torch.pi / 3, steps=4
        theta = torch.linspace(start=eval('0.'), end=eval('0.'),
                               steps=1).to(
            device=device)  # -torch.pi / 6, end=torch.pi / 6, steps=3



        # dist = torch.linspace(start=eval(config_sample.uniform.dist.min), end=eval(config_sample.uniform.dist.max), steps=config_sample.uniform.dist.steps).to(
        #    device=self.device)
        dist = torch.linspace(start=1., end=1., steps=1).to(device=device)

        azim_shape = azim.shape
        elev_shape = elev.shape
        theta_shape = theta.shape
        dist_shape = dist.shape
        in_shape = azim_shape + elev_shape + theta_shape + dist_shape
        azim = azim[:, None, None, None].expand(in_shape).reshape(-1)
        elev = elev[None, :, None, None].expand(in_shape).reshape(-1)
        theta = theta[None, None, :, None].expand(in_shape).reshape(-1)
        dist = dist[None, None, None, :].expand(in_shape).reshape(-1)
        from od3d.cv.geometry.transform import transf4x4_from_spherical
        cams_tform4x4_obj = transf4x4_from_spherical(azim=azim, elev=elev, theta=theta, dist=dist)

        T = cams_tform4x4_obj.shape[0]
        M = len(self)
        # M x T x 4 x 4
        pre_rendered_cams_tform4x4_obj = cams_tform4x4_obj[None, :].expand(M, T,
                                                                                 *cams_tform4x4_obj[0].shape).clone()
        pre_rendered_cams_tform4x4_obj[:, :, :3, 3] = 0.
        pre_rendered_meshes_size = self.get_verts_stacked_with_mesh_ids().flatten(1).max(dim=-1)[0]
        pre_rendered_meshes_dist = (pre_rendered_meshes_size * fxy) / (
                    500. * 0.8 - cxy)  # u = (x / z) * fx + cx  -> z = (fx * x) / (u - cx)
        pre_rendered_meshes_dist = pre_rendered_meshes_dist[:, None].expand(M, T)
        pre_rendered_cams_tform4x4_obj[:, :, 2, 3] = pre_rendered_meshes_dist

        # 1 x 1 x 4 x 4
        pre_rendered_cams_intr4x4 = torch.eye(4)[None, None].to(device=device).expand(M, 1, 4, 4)
        pre_rendered_cams_intr4x4[:, :, 0, 0] = fxy
        pre_rendered_cams_intr4x4[:, :, 1, 1] = fxy
        pre_rendered_cams_intr4x4[:, :, :2, 2] = cxy

        pre_rendered_meshes_ids = torch.arange(M).to(device=device)
        rendering = []
        for m in range(M):
            rendering.append(self.render_feats(
            cams_tform4x4_obj=pre_rendered_cams_tform4x4_obj[m:m+1],
            cams_intr4x4=pre_rendered_cams_intr4x4[m:m+1], imgs_sizes=imgs_sizes,
            meshes_ids=pre_rendered_meshes_ids[m:m+1], modality=MESH_RENDER_MODALITIES.VERTS_NCDS,
            broadcast_batch_and_cams=True,
            down_sample_rate=down_sample_rate))
        rendering = torch.cat(rendering, dim=0)

        if pcl is not None:
            pxl2d_pre_rendered = proj3d2d_broadcast(pts3d=pcl[:, None, None], proj4x4=tform4x4_broadcast(pre_rendered_cams_intr4x4, pre_rendered_cams_tform4x4_obj)) / down_sample_rate
            from od3d.cv.visual.mask import mask_from_pxl2d
            pxl2d_mask = mask_from_pxl2d(pxl2d=pxl2d_pre_rendered, dim_pxl=3, dim_pts=0, H=int(imgs_sizes[0] // down_sample_rate), W=int(imgs_sizes[1] // down_sample_rate))
            rendering[pxl2d_mask[:, :, None, :, :].repeat(1, 1, 3, 1, 1)] = 1.
            #rendering[:, :, :, ]
            #sample_pxl2d_grid(rendering.reshape(-1, C, H, W),
            #                  pxl2d=pxl2d_pre_rendered.reshape(-1, H, W, 2)).reshape(B, T, C, H, W)
        return show_imgs(rendering)


    def get_pre_rendered_feats(self, modality: MESH_RENDER_MODALITIES, cams_tform4x4_obj, cams_intr4x4, imgs_sizes, meshes_ids=None, broadcast_batch_and_cams=False, down_sample_rate=1. ):


        if modality not in self.pre_rendered_modalities.keys():
            cxy = 250.
            fxy = 500.
            T = cams_tform4x4_obj.shape[1]
            M = len(self)
            # M x T x 4 x 4
            pre_rendered_cams_tform4x4_obj = cams_tform4x4_obj[:1, :].expand(M, *cams_tform4x4_obj[0].shape).clone()
            pre_rendered_cams_tform4x4_obj[:, :, :3, 3] = 0.
            pre_rendered_meshes_size = self.get_verts_stacked_with_mesh_ids().flatten(1).max(dim=-1)[0]
            pre_rendered_meshes_dist = (pre_rendered_meshes_size * fxy) / (500. * 0.7 - cxy) # u = (x / z) * fx + cx  -> z = (fx * x) / (u - cx)
            pre_rendered_meshes_dist = pre_rendered_meshes_dist[:, None].expand(M, T)
            pre_rendered_cams_tform4x4_obj[:, :, 2, 3] = pre_rendered_meshes_dist

            # 1 x 1 x 4 x 4
            pre_rendered_cams_intr4x4 = cams_intr4x4[:1, :1].clone().expand(M, 1, *cams_intr4x4[0, 0].shape)
            pre_rendered_cams_intr4x4[:, :, 0, 0] = fxy
            pre_rendered_cams_intr4x4[:, :, 1, 1] = fxy
            pre_rendered_cams_intr4x4[:, :, :2, 2] = cxy

            pre_rendered_meshes_ids = torch.arange(M).to(device=meshes_ids.device)

            rendering = []
            for m in range(M):
                rendering.append(self.render_feats(
                    cams_tform4x4_obj=pre_rendered_cams_tform4x4_obj[m:m+1],
                    cams_intr4x4=pre_rendered_cams_intr4x4[m:m+1], imgs_sizes=imgs_sizes,
                    meshes_ids=pre_rendered_meshes_ids[m:m+1], modality=modality,
                    broadcast_batch_and_cams=broadcast_batch_and_cams,
                    down_sample_rate=down_sample_rate))
            rendering = torch.cat(rendering, dim=0)

            self.pre_rendered_modalities[modality] = Meshes.PreRendered(
                cams_tform4x4_obj=pre_rendered_cams_tform4x4_obj,
                cams_intr4x4=pre_rendered_cams_intr4x4,
                imgs_sizes=imgs_sizes,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                meshes_ids=pre_rendered_meshes_ids,
                down_sample_rate=down_sample_rate,
                rendering=rendering
            )

        else:
            assert (self.pre_rendered_modalities[modality].cams_tform4x4_obj[meshes_ids, :, :3, :3] == cams_tform4x4_obj[:, :, :3, :3]).all()
            assert (self.pre_rendered_modalities[modality].imgs_sizes == imgs_sizes).all()
            assert self.pre_rendered_modalities[modality].broadcast_batch_and_cams == broadcast_batch_and_cams
            assert self.pre_rendered_modalities[modality].down_sample_rate == down_sample_rate

        pre_rendered_feats = self.pre_rendered_modalities[modality].rendering[meshes_ids]
        pre_rendered_cam_intr4x4 = self.pre_rendered_modalities[modality].cams_intr4x4[meshes_ids]
        pre_rendered_cam_tform4x4_obj = self.pre_rendered_modalities[modality].cams_tform4x4_obj[meshes_ids]

        B, T, C, H, W = pre_rendered_feats.shape

        pxl2d_cams = get_pxl2d(H=H, W=W, dtype=pre_rendered_feats.dtype, device=pre_rendered_feats.device, B=None) * self.pre_rendered_modalities[modality].down_sample_rate
        pxl2d_cams = pxl2d_cams.expand(*pre_rendered_feats.shape[:2],  *pxl2d_cams.shape )
        pts3d_homog_cams = transf3d_broadcast(pts3d=add_homog_dim(pxl2d_cams, dim=4), transf4x4=cams_intr4x4.pinverse()[:, :, None, None,]) * cams_tform4x4_obj[:, :, 2, 3, None, None, None,]
        pts3d_pre_rendered = transf3d_broadcast(pts3d=pts3d_homog_cams, transf4x4=tform4x4(pre_rendered_cam_tform4x4_obj, inv_tform4x4(cams_tform4x4_obj))[:, :, None, None,])
        pxl2d_pre_rendered = proj3d2d_broadcast(pts3d=pts3d_pre_rendered, proj4x4=pre_rendered_cam_intr4x4[:, :, None, None]) / self.pre_rendered_modalities[modality].down_sample_rate
        cams_features = sample_pxl2d_grid(pre_rendered_feats.reshape(-1, C, H, W), pxl2d=pxl2d_pre_rendered.reshape(-1, H, W, 2)).reshape(B, T, C, H, W)

        return cams_features

    def get_pre_rendered_masks(self, cams_tform4x4_obj, cams_intr4x4, imgs_sizes, meshes_ids=None, broadcast_batch_and_cams=False, down_sample_rate=1.):
        assert self.pre_rendered_masks_verts_vsbl_cams_tform4x4_obj == cams_tform4x4_obj
        assert self.pre_rendered_masks_verts_vsbl_cams_intr4x4 == cams_intr4x4
        assert self.pre_rendered_masks_verts_vsbl_imgs_sizes == imgs_sizes
        assert self.pre_rendered_masks_verts_vsbl_meshes_ids == meshes_ids
        assert self.pre_rendered_masks_verts_vsbl_broadcast_batch_and_cams == broadcast_batch_and_cams
        assert self.pre_rendered_masks_verts_vsbl_down_sample_rate == down_sample_rate

        if self.pre_rendered_feats is None:
            self.pre_rendered_masks_verts_vsbl_cams_tform4x4_obj = cams_tform4x4_obj
            self.pre_rendered_masks_verts_vsbl_cams_intr4x4 = cams_intr4x4
            self.pre_rendered_masks_verts_vsbl_imgs_sizes = imgs_sizes
            self.pre_rendered_feats_meshes_ids = meshes_ids
            self.pre_rendered_feats_broadcast_batch_and_cams = broadcast_batch_and_cams
            self.pre_rendered_down_sample_rate = down_sample_rate
            self.pre_rendered_feats = self.render_feats(
                cams_tform4x4_obj=self.pre_rendered_feats_cams_tform4x4_obj,
                cams_intr4x4=self.pre_rendered_feats_cams_intr4x4, imgs_sizes=self.pre_rendered_feats_imgs_sizes,
                meshes_ids=self.pre_rendered_feats_meshes_ids, modality=MESH_RENDER_MODALITIES.MASK_VERTS_VSBL,
                broadcast_batch_and_cams=self.pre_rendered_feats_broadcast_batch_and_cams,
                down_sample_rate=self.pre_rendered_down_sample_rate)

        return self.pre_rendered_feats

    #def get_pre_rendered_masks(self):

    def render_feats(self, cams_tform4x4_obj, cams_intr4x4, imgs_sizes, meshes_ids=None, modality=MESH_RENDER_MODALITIES.FEATS, broadcast_batch_and_cams=False, down_sample_rate=1.):
        # imgs_size: (height, width)
        dtype = cams_tform4x4_obj.dtype
        device = cams_tform4x4_obj.device

        self.to(device)

        if down_sample_rate != 1.:
            cams_intr4x4 = cams_intr4x4.clone()
            if cams_intr4x4.dim() == 2:
                cams_intr4x4[:2] /= down_sample_rate
            elif cams_intr4x4.dim() == 3:
                cams_intr4x4[:, :2] /= down_sample_rate
            elif cams_intr4x4.dim() == 4:
                cams_intr4x4[:, :, :2] /= down_sample_rate
            else:
                raise NotImplementedError
            imgs_sizes = imgs_sizes.clone() // down_sample_rate
        else:
            cams_intr4x4 = cams_intr4x4.clone()
            imgs_sizes = imgs_sizes.clone()

        if meshes_ids is None:
            meshes_ids = torch.LongTensor(list(range(len(self)))).to(device=device)

        if isinstance(meshes_ids, torch.LongTensor):
            meshes_ids = meshes_ids.clone().to(device=device)

        meshes_count = meshes_ids.shape[0]
        if cams_tform4x4_obj.dim() == 4:
            cams_count = cams_tform4x4_obj.shape[1]
        elif cams_tform4x4_obj.dim() == 3:
            cams_count = cams_tform4x4_obj.shape[0]
        else:
            raise ValueError(f'Set `cams_tform4x4_obj.dim()` must be 3 or 4')

        if broadcast_batch_and_cams:
            meshes_ids = meshes_ids
            if cams_tform4x4_obj.dim() == 3:
                cams_tform4x4_obj = cams_tform4x4_obj[None, :]
            if cams_intr4x4.dim() == 3:
                cams_intr4x4 = cams_intr4x4[None, :]
            cams_tform4x4_obj = cams_tform4x4_obj.expand(meshes_count, cams_count, 4, 4).reshape(-1, 4, 4)
            cams_intr4x4 = cams_intr4x4.expand(meshes_count, cams_count, 4, 4).reshape(-1, 4, 4)
            meshes_ids = meshes_ids[:, None].expand(meshes_count, cams_count).reshape(-1)
            render_count = meshes_count * cams_count
        else:
            if meshes_count != cams_count:
                raise ValueError(f'Set `broadcast_batch_and_cams=True` to allow different number of cameras and meshes')
            render_count = meshes_count

        if self.gaussian_splat_enabled and modality in \
                [MESH_RENDER_MODALITIES.VERTS_NCDS, MESH_RENDER_MODALITIES.RGB, MESH_RENDER_MODALITIES.FEATS, MESH_RENDER_MODALITIES.MASK]: # MESH_RENDER_MODALITIES.FEATS:
            #from od3d.cv.render.gaussian_splats import render_gaussians
            from od3d.cv.render.gaussians_splats_v2 import render_gaussians

            pts3d = self.get_verts_stacked_with_mesh_ids(mesh_ids=meshes_ids).to(device).clone().detach()
            if modality == MESH_RENDER_MODALITIES.VERTS_NCDS:
                feats = self.get_verts_ncds_stacked_with_mesh_ids(mesh_ids=meshes_ids).to(device)
            elif modality == MESH_RENDER_MODALITIES.RGB:
                feats = self.get_rgb_stacked_with_mesh_ids(mesh_ids=meshes_ids).to(device)
            elif modality == MESH_RENDER_MODALITIES.MASK:
                feats = torch.ones_like(pts3d[..., 0:1])
            else:
                feats = self.get_feats_stacked_with_mesh_ids(mesh_ids=meshes_ids).to(device)

            pts3d_mask = self.mask_verts_not_padded.to(device)[meshes_ids]
            mesh_feats2d_rendered = render_gaussians(cams_tform4x4_obj=cams_tform4x4_obj, cams_intr4x4=cams_intr4x4,
                                                     imgs_size=imgs_sizes, pts3d=pts3d, pts3d_mask=pts3d_mask,
                                                     feats=feats, opacity=self.gaussian_splat_opacity,
                                                     pts3d_size_rel_to_neighbor_dist=
                                                     self.gaussian_splat_pts3d_size_rel_to_neighbor_dist)

            if broadcast_batch_and_cams:
                #logger.info(cams_intr4x4.reshape(meshes_count, cams_count, 4, 4)[:, 0])
                #logger.info(cams_tform4x4_obj.reshape(meshes_count, cams_count, 4, 4)[:, 0])
                mesh_feats2d_rendered = mesh_feats2d_rendered.reshape(meshes_count, cams_count, *mesh_feats2d_rendered.shape[-3:])
                # from od3d.cv.visual.show import show_imgs
                # show_imgs(mesh_feats2d_rendered)
            else:
                pass
            return mesh_feats2d_rendered
        # self.to(device)

        #num_cams = cams_tform4x4_obj.shape[0]

        #cams_tform4x4_obj = cams_tform4x4_obj.repeat_interleave(num_meshes, dim=0)
        #cams_intr4x4 = cams_intr4x4.repeat_interleave(num_meshes, dim=0)
        #imgs_sizes = imgs_sizes.repeat_interleave(num_meshes, dim=0)
        t3d_tform_pscl3d = torch.Tensor([[-1., 0., 0., 0.],
                                         [0., -1., 0., 0.],
                                         [0., 0., 1., 0.],
                                         [0., 0., 0., 1.]]).to(device=device, dtype=dtype)
        # t3d_cam_tform_obj = torch.matmul(t3d_tform_pscl3d, cams_tform4x4_obj)
        t3d_cam_tform_obj = t3d_tform_pscl3d[None].expand(cams_tform4x4_obj.shape).bmm(cams_tform4x4_obj)
        R = t3d_cam_tform_obj[..., :3, :3].permute(0, 2, 1)
        t = t3d_cam_tform_obj[..., :3, 3]
        focal_length = torch.stack([cams_intr4x4[..., 0, 0], cams_intr4x4[..., 1, 1]], dim=-1)
        principal_point = torch.stack([cams_intr4x4[..., 0, 2], cams_intr4x4[..., 1, 2]], dim=-1)

        cameras = PerspectiveCameras(device=device, R=R, T=t, focal_length=focal_length,
                                     principal_point=principal_point, in_ndc=False,
                                     image_size=imgs_sizes[None, ].expand(render_count, 2))

          # K=self.K_4x4[None,]) #, K=K) # , K=K , znear=0.001, zfar=100000,
        #  znear=0.001, zfar=100000, fov=10
        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.

        raster_settings = RasterizationSettings(
            image_size=[int(imgs_sizes[0]), int(imgs_sizes[1])],
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=self.pt3d_raster_perspective_correct,
            cull_backfaces=False # cull_backfaces=True
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        pt3dmeshes = self.pt3dmeshes[meshes_ids]
        fragments = rasterizer(pt3dmeshes)
        # pix_to_face: BxHxWx1, zbuf: BxHxWx1, bary_coords: BxHxWx1x3, dists: BxHxWx1
        if modality == MESH_RENDER_MODALITIES.MASK:
            mask = fragments.zbuf.permute(0, 3, 1, 2) > 0.
            if broadcast_batch_and_cams:
                mask = mask.reshape(meshes_count, cams_count, *mask.shape[-3:])

            return mask

        if modality == MESH_RENDER_MODALITIES.DEPTH:
            depth = fragments.zbuf.permute(0, 3, 1, 2)
            if broadcast_batch_and_cams:
                depth = depth.reshape(meshes_count, cams_count, *depth.shape[-3:])

            return depth

        if modality == MESH_RENDER_MODALITIES.MASK_VERTS_VSBL:
            B = fragments.pix_to_face.shape[0]
            faces_ids = torch.cat([self.get_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0)
            # verts_ids_vsbl = torch.cat([self.get_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0) [fragments.pix_to_face.reshape(B, -1)].reshape(B, -1)  # .unique(dim=1)
            verts_vsbl_mask = torch.zeros(size=(B, self.verts_counts_max), dtype=torch.bool, device=device)
            for b in range(B):
                # logger.info(f'meshes_ids {meshes_ids}')
                faces_ids_vsbl = fragments.pix_to_face[b]
                faces_ids_vsbl = faces_ids_vsbl.unique()
                faces_ids_vsbl = faces_ids_vsbl[faces_ids_vsbl >= 0]
                verts_ids_vsbl = faces_ids[faces_ids_vsbl].unique()
                verts_vsbl_mask[b, verts_ids_vsbl] = 1

            return verts_vsbl_mask

        if modality == MESH_RENDER_MODALITIES.FEATS:
            feats_from_faces = torch.cat([self.get_feats_from_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0)
        else:
            feats_from_faces = torch.cat([self.get_verts_ncds_from_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0)

        mesh_feats2d_rendered = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, feats_from_faces)[:, ..., 0,:].permute(0, 3, 1, 2)
        #mask = fragments.pix_to_face >= 0
        #mesh_feats2d_prob = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

        if broadcast_batch_and_cams:
            mesh_feats2d_rendered = mesh_feats2d_rendered.reshape(meshes_count, cams_count, *mesh_feats2d_rendered.shape[-3:])

        return mesh_feats2d_rendered


