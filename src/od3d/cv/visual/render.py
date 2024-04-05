import torch
import logging
from pytorch3d.renderer import (
    # look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    # SoftPhongShader,
    HardPhongShader,
    TexturesVertex
)
from pytorch3d.structures.meshes import Meshes
from pytorch3d.io import IO
from pytorch3d.renderer.mesh.utils import interpolate_face_attributes



def render_mask(fpath_mesh, cam_tform_obj, cam_intr, img_size):
    rgba_synthetic = render_mesh(fpath_mesh, cam_tform_obj, cam_intr, img_size, modality="rgba")
    return (rgba_synthetic[3:] > 0)
def render_depth(fpath_mesh, cam_tform_obj, cam_intr, img_size):
    depth_synthetic = render_mesh(fpath_mesh, cam_tform_obj, cam_intr, img_size, modality="depth")
    return depth_synthetic

def render_mesh(fpath_mesh, cam_tform_obj, cam_intr, img_size, modality="rgba", feats=None):

    dtype = cam_tform_obj.dtype
    device = cam_tform_obj.device

    focal_length = torch.Tensor([cam_intr[0, 0], cam_intr[1, 1]]).to(device=device, dtype=dtype)
    principal_point = torch.Tensor([cam_intr[0, 2], cam_intr[1, 2]]).to(device=device, dtype=dtype)
    io = IO()
    mesh = io.load_mesh(fpath_mesh, device=device)
    verts = mesh[0].verts_list()[0].to(device)
    faces = mesh[0].faces_list()[0].to(device)
    verts_shape = verts.shape
    verts_rgb = torch.ones(size=verts_shape, device=device)[None,] * 0.5  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )

    t3d_tform_pscl3d = torch.Tensor([[-1., 0., 0., 0.],
                                     [0., -1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]]).to(device=device, dtype=dtype)
    t3d_cam_tform_obj = torch.matmul(t3d_tform_pscl3d, cam_tform_obj)

    R = t3d_cam_tform_obj[:3, :3].T[None,]
    t = t3d_cam_tform_obj[:3, 3][None,]

    cameras = PerspectiveCameras(device=device, R=R, T=t, focal_length=focal_length[None,],
                                 principal_point=principal_point[None,], in_ndc=False,
                                 image_size=img_size[None,])  # K=self.K_4x4[None,]) #, K=K) # , K=K , znear=0.001, zfar=100000,
    #  znear=0.001, zfar=100000, fov=10
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=[int(img_size[0]), int(img_size[1])],
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    if modality == "depth":
        fragments = rasterizer(mesh)
        return (fragments.zbuf[0]).permute(2, 0, 1)
    elif modality == "mask_verts_vsbl":
        fragments = rasterizer(mesh)
        B = fragments.pix_to_face.shape[0]
        verts_ids_vsbl = faces[fragments.pix_to_face.reshape(B, -1)].reshape(B, -1).unique(dim=1)
        verts_vsbl_mask = torch.zeros(size=(B, verts_shape[0]), dtype=torch.bool, device=device)
        verts_vsbl_mask[verts_ids_vsbl] = 1
    elif modality == "interpolate":
        # pix_to_face, zbuf, bary_coord, dists
        fragments = rasterizer(mesh)
        if feats is None:
            verts_rgb_ncds = verts.clone()
            verts_rgb_ncds = (verts_rgb_ncds - verts_rgb_ncds.min(dim=0).values[None,]) / (verts_rgb_ncds.max(dim=0).values[None,] - verts_rgb_ncds.min(dim=0).values[None,])
            feats = verts_rgb_ncds
        return interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, feats[faces])[0, ..., 0, :].permute(2, 0, 1)
    elif modality == "nearest":
        pix_to_face, zbuf, bary_coord, dists = rasterizer(mesh)
        # TODO:
        raise NotImplementedError
        #ori_shape = bary_coord.shape
        #exr = bary_coord * (bary_coord < 0)
        #bary_coords_ = bary_coord.view(-1, bary_coord.shape[-1])
        #arg_max_idx = bary_coords_.argmax(1)
        #bary_coord = (
        #        torch.zeros_like(bary_coords_)
        #        .scatter(1, arg_max_idx.unsqueeze(1), 1.0)
        #        .view(*ori_shape)
        #        + exr
        #)
        return interpolate_face_attributes(pix_to_face, bary_coord, verts_rgb).squeeze()
    else:
        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
        # -z direction.
        lights = PointLights(device=device, location=[[0.0, 0.0, 10.0]])

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )

        rgba_synthetic_batch = renderer(mesh)
        rgba_synthetic = (rgba_synthetic_batch[0, ..., :] * 255).to(torch.uint8).permute(2, 0, 1)

        if modality == "rgba" or modality == "all":
            return rgba_synthetic
        else:
            return rgba_synthetic[:3]