import logging
logger = logging.getLogger(__name__)
import typer
app = typer.Typer()

import torch
from od3d.datasets.co3d import CO3D
from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES, CUBOID_SOURCES
from od3d.cv.visual.show import show_imgs, show_scene
from od3d.cv.geometry.transform import transf3d_broadcast
from pathlib import Path

@app.command()
def sequence():
    sequence_category = 'bicycle'
    sequence_name = '397_49943_98337' # '397_49943_98337' '397_49968_98375' '430_60627_118124' '62_4318_10726'
    cams_count = -1 # -1=all
    flag_show_scene = False
    co3d = CO3D.create_by_name('co3dv1_10s_zsp_labeled_cuboid_ref',
                               config={'categories': ['bicycle'],
                                       'mesh_feats_type': 'M_dinov2_vitb14_frozen_base_no_norm_T_centerzoom512_R_acc',
                                       'dist_verts_mesh_feats_reduce_type': 'min_avg'})
    #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'

    categories = co3d.categories
    sequence = co3d.get_sequence_by_category_and_name(category=sequence_category, name=sequence_name)
    cams_tform4x4_world, cams_intr4x4, cams_imgs = sequence.get_cams(
        cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.ZSP_LABELED_CUBOID_REF,
        show_imgs=True, cams_count=cams_count)

    # cams_imgs = torch.stack(cams_imgs, dim=0)
    cams_tform4x4_world = torch.stack(cams_tform4x4_world, dim=0)
    cams_intr4x4 = torch.stack(cams_intr4x4, dim=0)

    # show_imgs(rgbs=cams_imgs, fpath='sequence_imgs.png')

    mesh = sequence.get_mesh(mesh_source=CUBOID_SOURCES.DEFAULT)
    mesh.verts = transf3d_broadcast(mesh.verts, sequence.zsp_labeled_cuboid_ref_tform_obj)

    feats = sequence.feats
    # torch.nn.utils.rnn.pad_sequence(seq1_feats + seq2_feats, batch_first=True, padding_value=torch.nan)
    # torch.nn.utils.rnn.unpad_sequence(padded_sequences, lengths, batch_first=False)



    viewpoints_per_vert = [len(verts_feats) for verts_feats in feats]
    viewpoints = sequence.feats_viewpoints
    viewpoints = torch.nn.utils.rnn.pad_sequence(viewpoints)
    # remove scale because we did this for cam_tform_obj as well
    zsp_labeled_cuboid_ref_tform_obj = sequence.zsp_labeled_cuboid_ref_tform_obj.clone()
    _scale = zsp_labeled_cuboid_ref_tform_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    zsp_labeled_cuboid_ref_tform_obj[:3] = zsp_labeled_cuboid_ref_tform_obj[:3] / _scale

    viewpoints = transf3d_broadcast(viewpoints, sequence.zsp_labeled_cuboid_ref_tform_obj)
    viewpoints = torch.nn.functional.normalize(viewpoints, dim=-1)
    viewpoints = torch.nn.utils.rnn.unpad_sequence(viewpoints, lengths=viewpoints_per_vert)

    sequence_dict = {
        'feats': feats,
        'viewpoints': viewpoints,
        'cams_intr4x4': cams_intr4x4,
        'cams_tform4x4_world': cams_tform4x4_world,
        'pts3d': mesh.verts,
        'imgs': cams_imgs}

    torch.save(sequence_dict, f'{sequence_category}_{sequence_name}.pt')

    if flag_show_scene:
        imgs = show_scene(meshes=[mesh], cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4,
                          cams_imgs=cams_imgs, return_visualization=True, viewpoints_count=2, pts3d=[torch.cat(viewpoints, dim=0)])
        show_imgs(rgbs=imgs, fpath=f'{sequence_category}_{sequence_name}.png')
