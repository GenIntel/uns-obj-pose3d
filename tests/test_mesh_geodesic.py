
from od3d.cv.geometry.mesh import Meshes

def test_geodesic_distance():
    mesh = Meshes.load_by_name('cuboid', device='cuda:0', faces_count=1000)

    #geodesic_distances = mesh.get_verts_geodestic_distances()

    vertex_id = 0
    #mesh.rgb = (mesh.geodesic_dist[vertex_id, :, None].repeat(1, 3) * 10).clamp(0, 1)
    mesh.rgb = (mesh.geodesic_prob[vertex_id, :, None].repeat(1, 3)).clamp(0, 1)
    mesh.show()
