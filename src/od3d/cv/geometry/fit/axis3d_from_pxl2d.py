import torch
from od3d.cv.geometry.transform import reproj2d3d_broadcast, so3_exp_map, rot3x3
from logging import getLogger
logger = getLogger(__file__)


def axis3d_from_pxl2d(kpts2d_orient, cam_intr4x4):
    orients = {}

    A_orthog = []
    A_aligned = []
    A_aligned_norm = []
    b_aligned = []
    A_min = []
    A_max = []
    for key in kpts2d_orient.keys():
        kpts2d_orient_lr = kpts2d_orient[key].clone()
        if len(kpts2d_orient_lr) > 0:
            kpts3d_homog_orient_lr = reproj2d3d_broadcast(kpts2d_orient_lr, cam_intr4x4.inverse())

            vector_homog_orient_lr = (kpts3d_homog_orient_lr[:, 0] - kpts3d_homog_orient_lr[:, 1])
            vector_homog_orient_lr = vector_homog_orient_lr / vector_homog_orient_lr.norm(dim=-1, keepdim=True)

            #kpts3d_homog_orient_lr = kpts3d_homog_orient_lr / kpts3d_homog_orient_lr.norm(dim=-1, keepdim=True)

            orient3d_planes = []
            for i in range(len(kpts3d_homog_orient_lr)):
                orient3d_plane_i = torch.linalg.cross(kpts3d_homog_orient_lr[i, 0], kpts3d_homog_orient_lr[i, 1])
                orient3d_plane_i = orient3d_plane_i / orient3d_plane_i.norm(dim=-1, keepdim=True)
                vector_homog_orient_i = (kpts3d_homog_orient_lr[i, 0] - kpts3d_homog_orient_lr[i, 1])
                vector_homog_orient_i = vector_homog_orient_i / vector_homog_orient_i.norm(dim=-1, keepdim=True)

                orient3d_planes.append(orient3d_plane_i)
                A_orthog_kp = torch.zeros(size=(1, 9))
                A_aligned_kp = torch.zeros(size=(1, 9))
                A_aligned_norm_kp = torch.zeros(size=(1, 9))
                b_kp = torch.zeros(size=(1,))
                if key == 'left-right':
                    A_orthog_kp[0, :3] = orient3d_plane_i
                    A_aligned_kp[0, :3] = vector_homog_orient_i
                    A_aligned_norm_kp[0, :2] = 1.
                elif key == 'back-front':
                    A_orthog_kp[0, 3:6] = orient3d_plane_i
                    A_aligned_kp[0, 3:6] = vector_homog_orient_i
                    A_aligned_norm_kp[0, 3:5] = 1.
                elif key == 'top-bottom':
                    A_orthog_kp[0, 6:9] = orient3d_plane_i
                    A_aligned_kp[0, 6:9] = vector_homog_orient_i
                    A_aligned_norm_kp[0, 6:8] = 1.
                b_kp[0] = 1.
                b_aligned.append(b_kp)
                A_orthog.append(A_orthog_kp)
                A_aligned.append(A_aligned_kp)
                A_aligned_norm.append(A_aligned_norm_kp)

            if len(orient3d_planes) > 1:
                # orient3d = vector_homog_orient_lr[0] # .mean(dim=0)
                orient3d = torch.linalg.cross(orient3d_planes[0], orient3d_planes[1])
                orient3d = orient3d / orient3d.norm(dim=-1, keepdim=True)
                eins = torch.einsum('d,cd->c', orient3d, vector_homog_orient_lr[:, :])  # [:1])

                if (eins < 0.).all():
                    logger.info(f'switch axis for {key}')
                    orients[key] = -orient3d
                elif (eins > 0.).all():
                    orients[key] = orient3d
                else:
                    logger.warning(f'Inconsistency of axis3d with axis2d alignment. Skipping orient for {key}...')
                    orients[key] = []
            else:
                orients[key] = []
        else:
            orients[key] = []
    if len(orients['left-right']) == 0:
        orient3d = torch.linalg.cross(orients['back-front'], orients['top-bottom'])
        orient3d = orient3d / orient3d.norm(dim=-1, keepdim=True)
        orients['left-right'] = orient3d
        #orient3d = torch.linalg.cross(orients['top-bottom'], orients['left-right'])
        #orient3d = orient3d / orient3d.norm(dim=-1, keepdim=True)
        #orients['back-front'] = orient3d
    if len(orients['back-front']) == 0:
        orient3d = torch.linalg.cross(orients['top-bottom'], orients['left-right'])
        orient3d = orient3d / orient3d.norm(dim=-1, keepdim=True)
        orients['back-front'] = orient3d
        #orient3d = torch.linalg.cross(orients['left-right'], orients['back-front'])
        #orient3d = orient3d / orient3d.norm(dim=-1, keepdim=True)
        #orients['top-bottom'] = orient3d
    if len(orients['top-bottom']) == 0:
        orient3d = torch.linalg.cross(orients['left-right'], orients['back-front'])
        orient3d = orient3d / orient3d.norm(dim=-1, keepdim=True)
        orients['top-bottom'] = orient3d
        #orient3d = torch.linalg.cross(orients['back-front'], orients['top-bottom'])
        #orient3d = orient3d / orient3d.norm(dim=-1, keepdim=True)
        #orients['left-right'] = orient3d

    orients = torch.stack([orients['left-right'], orients['back-front'], orients['top-bottom']], dim=0)

    from od3d.cv.geometry.transform import so3_log_map

    cam_rot3x3_obj = orients.T
    U, S, V = torch.linalg.svd(cam_rot3x3_obj)
    cam_rot3x3_obj = rot3x3(U, V)
    cam_rot3x3_obj = so3_exp_map(so3_log_map(cam_rot3x3_obj))

    # cam_rot3x3_obj = rot3x3(cam_rot3x3_obj, torch.Tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]))

    # A.append(torch.Tensor([[]]))
    A_orthog = torch.cat(A_orthog, dim=0)
    A_aligned = torch.cat(A_aligned, dim=0)
    A_aligned_norm = torch.cat(A_aligned_norm, dim=0)
    b_aligned = torch.cat(b_aligned, dim=0)

    tmp_rot3_obj = torch.zeros(3).to(device=cam_rot3x3_obj.device)
    l_aligned = torch.tensor(torch.zeros_like(b_aligned), requires_grad=True, dtype=torch.float64)
    # cam_rot3x3_obj_param = cam_rot3x3_obj
    for i in range(500):
        cam_rot3x3_obj = torch.nn.Parameter(cam_rot3x3_obj.detach(), requires_grad=True)

        optimizer = torch.optim.SGD(params=[cam_rot3x3_obj], lr=0.1) # 0.01 too low -> no update, too large oscillating

        loss_orthog_kpts = (torch.bmm(A_orthog[None,], cam_rot3x3_obj.T.reshape(1, 9, 1))[0, :, 0]).norm()
        loss_orthog_coords = (rot3x3(cam_rot3x3_obj.T, cam_rot3x3_obj) - torch.eye(3, device=cam_rot3x3_obj.device)).flatten().norm() # ().mean()
        # loss_aligned_kpts = (torch.bmm(A_aligned[None,], cam_rot3x3_obj.T.reshape(1, 9, 1))[0, :, 0]).norm()  #  - cam_rot3x3_obj.detach().T.reshape(1, 9).repeat(A_aligned_norm.shape[0], 1)[A_aligned_norm == 1.].reshape(A_aligned_norm.shape[0], -1).norm(dim=-1)).norm()
        #loss_aligned_kpts = (-torch.log(torch.bmm(A_aligned[None,], cam_rot3x3_obj.T.reshape(1, 9, 1))[0, :, 0])).sum()
        #loss_aligned_kpts =
        loss = loss_orthog_kpts + 0.1 * loss_orthog_coords # + 0.1 * loss_aligned_kpts
        loss.backward()
        #logger.info(f"Loss {loss}, orthog {loss_orthog_kpts}, aligned {loss_aligned_kpts}")# , cam_rot3x3_obj {cam_rot3x3_obj}")
        optimizer.step()

        if (i+1) % 10 == 0:
            U, S, V = torch.linalg.svd(cam_rot3x3_obj)
            cam_rot3x3_obj = rot3x3(U, V)
        """
        cam_rot3x3_obj = rot3x3(so3_exp_map(tmp_rot3_obj.detach()), cam_rot3x3_obj.detach())
        cam_rot3x3_obj = so3_exp_map(so3_log_map(cam_rot3x3_obj.detach()))
        tmp_rot3_obj = torch.nn.Parameter(torch.zeros(3).to(device=cam_rot3x3_obj.device), requires_grad=True)

        optimizer = torch.optim.SGD(params=[tmp_rot3_obj], lr=0.01) # 0.01 too low -> no update, too large oscillating

        cam_rot3x3_obj = rot3x3(so3_exp_map(tmp_rot3_obj), cam_rot3x3_obj)
        loss_orthog_kpts = (torch.bmm(A_orthog[None,], cam_rot3x3_obj.T.reshape(1, 9, 1))[0, :, 0]).norm()

        loss_aligned_kpts = (torch.bmm(A_aligned[None,], cam_rot3x3_obj.T.reshape(1, 9, 1))[0, :, 0]).norm()  #  - cam_rot3x3_obj.detach().T.reshape(1, 9).repeat(A_aligned_norm.shape[0], 1)[A_aligned_norm == 1.].reshape(A_aligned_norm.shape[0], -1).norm(dim=-1)).norm()
        #loss_aligned_kpts = (-torch.log(torch.bmm(A_aligned[None,], cam_rot3x3_obj.T.reshape(1, 9, 1))[0, :, 0])).sum()
        #loss_aligned_kpts =
        loss = loss_orthog_kpts # + 0.1 * loss_aligned_kpts
        loss.backward()
        #logger.info(f"Loss {loss}, orthog {loss_orthog_kpts}, aligned {loss_aligned_kpts}")# , cam_rot3x3_obj {cam_rot3x3_obj}")
        optimizer.step()
        """

    logger.info(f'Alignment {torch.bmm(A_aligned[None,], cam_rot3x3_obj.T.reshape(1, 9, 1))[0, :, 0]}')
    logger.info(f"trace {cam_rot3x3_obj.trace()}")
    logger.info(f"R.T, R {rot3x3(cam_rot3x3_obj.T, cam_rot3x3_obj)}")
    logger.info(f"{cam_rot3x3_obj}")
    cam_rot3x3_obj = so3_exp_map(so3_log_map(cam_rot3x3_obj.detach()))

    return cam_rot3x3_obj.detach()

