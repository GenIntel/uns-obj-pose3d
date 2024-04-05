import logging
logger = logging.getLogger(__name__)
import torch
from od3d.cv.select import batched_index_select


def sample_models(pts, fit_func, fit_pts_count: int, fits_count: int, pts_dist=None, pts_affinity=None):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        fit_func: returns multiple fitted models for points and points ids, in: ...xNxF ...xPxS -> ...xPxM
        fit_pts_count (int): number of points required to fit a model (P)
        fits_count (int): number of proposed model fits (S)
        pts_affinity: ...xNxN
        pts_dist: ...xNxN
    Returns:
        models: ...xSx...
    """
    device = pts.device
    N = pts.shape[-2]
    batch_dims = pts.shape[:-2]

    if pts_dist is not None:
        if pts_affinity is None:
            pts_affinity = 1./pts_dist
        else:
            logger.warning('ignoring `pts_dist` as `pts_affinity` is defined as well')

    if pts_affinity is None:
        pts_affinity = torch.ones(size=batch_dims + (N, N)).to(device=device)
    else:
        logger.warning('ignoring `pts_affinity` as it is not implemented yet')


    pts_sample_probs = torch.ones(size=batch_dims + (fits_count, N)).to(device=device)

    # ...xPxS
    pts_ids = torch.multinomial(pts_sample_probs.view(-1, N), num_samples=fit_pts_count).view(batch_dims + (fits_count, fit_pts_count))

    # ...xPxM
    models = fit_func(pts, pts_ids)

    return models

def ransac(pts, fit_func, score_func, fit_pts_count: int, fits_count: int, pts_dist=None, pts_affinity=None, return_score=False):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        fit_func: returns multiple fitted models for points and points ids, in: ...xNxF ...xPxS -> ...xPxM
        score_func: return scores for multiple fitted models, in: ...xNxF, ...xPxM -> ...xP
        fit_pts_count (int): number of points required to fit a model (P)
        fits_count (int): number of proposed model fits (S)
        pts_affinity: ...xNxN
        pts_dist: ...xNxN
    Returns:
        model: returns best fit model
    """
    batch_dims = pts.shape[:-2]
    batch_dims_count = len(batch_dims)

    models = sample_models(pts=pts, fit_func=fit_func, fit_pts_count=fit_pts_count, fits_count=fits_count, pts_dist=pts_dist, pts_affinity=pts_affinity)
    # ...xP
    scores = score_func(pts, models)

    # B...
    best_id = scores.argmax(dim=-1)

    best_model = batched_index_select(input=models, index=best_id[..., None]).squeeze(dim=batch_dims_count)
    best_score = batched_index_select(input=scores, index=best_id[..., None]).squeeze(dim=batch_dims_count)

    if return_score:
        return best_model, best_score

    else:
        return best_model