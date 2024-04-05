import logging
logger = logging.getLogger(__name__)
import torch

def einsum_cdist(comb, featsA, featsB):
    """
    Expand and permute a tensor based on the einsum equation.

    Parameters:
        comb (str): Einsum equation specifying the dimensions. e.g. 'ab,bc->ac'
        featsA (torch.Tensor): Input tensor.
        featsB (torch.Tensor): Input tensor.

    Returns:
        dist (torch.Tensor): Distance tensor.
    """

    # Parse the einsum equation to get input and output subscripts
    input_subscripts, output_subscripts = comb.split('->')
    input_subscripts = input_subscripts.split(',')
    featsA_input_subscripts = [*input_subscripts[0]]
    featsB_input_subscripts = [*input_subscripts[1]]
    removed_subscripts = list(set(featsA_input_subscripts + featsB_input_subscripts) - set([*output_subscripts]))
    featsA_input_subscripts = ''.join(featsA_input_subscripts)
    featsB_input_subscripts = ''.join(featsB_input_subscripts)
    if len(removed_subscripts) > 1:
        logger.warning(f'calc_sim: removed_subscripts={removed_subscripts}')
    removed_subscript = removed_subscripts[0]
    batch_subscripts = list(
        set([*input_subscripts[0]]).intersection(set([*input_subscripts[1]])) - set([removed_subscript]))
    featsA_specific_subscripts = list(
        set([*input_subscripts[0]]) - set([*batch_subscripts]) - set([removed_subscript]))
    featsB_specific_subscripts = list(
        set([*input_subscripts[1]]) - set([*batch_subscripts]) - set([removed_subscript]))
    #logger.info(f'batch subscripts: {batch_subscripts}')
    #logger.info(f'feats A specific subscripts: {featsA_specific_subscripts}')
    #logger.info(f'feats B specific subscripts: {featsB_specific_subscripts}')
    #logger.info(f'removed subscripts: {removed_subscript}')
    featsA_intermed_subscripts = ''.join(batch_subscripts + featsA_specific_subscripts + [removed_subscript])
    featsB_intermed_subscripts = ''.join(batch_subscripts + featsB_specific_subscripts + [removed_subscript])
    #logger.info(f'feats A subscripts change: {featsA_input_subscripts} -> {featsA_intermed_subscripts}')
    #logger.info(f'feats B subscripts change: {featsB_input_subscripts} -> {featsB_intermed_subscripts}')
    featsA_bAd = torch.einsum(featsA_input_subscripts + '->' + featsA_intermed_subscripts, featsA)
    featsB_bBd = torch.einsum(featsB_input_subscripts + '->' + featsB_intermed_subscripts, featsB)
    batch_shapes = featsA_bAd.shape[:len(batch_subscripts)]
    featsA_specific_shapes = featsA_bAd.shape[
                             len(batch_subscripts):len(batch_subscripts) + len(featsA_specific_subscripts)]
    featsB_specific_shapes = featsB_bBd.shape[
                             len(batch_subscripts):len(batch_subscripts) + len(featsB_specific_subscripts)]
    feats_dim = featsA_bAd.shape[-1]
    #logger.info(f'batch shapes: {batch_shapes}')
    #logger.info(f'feats A specific shapes: {featsA_specific_shapes}')
    #logger.info(f'feats B specific shapes: {featsB_specific_shapes}')
    #logger.info(f'feats dim: {feats_dim}')
    dist = torch.cdist(featsA_bAd.reshape(batch_shapes.numel(), featsA_specific_shapes.numel(), feats_dim),
                       featsB_bBd.reshape(batch_shapes.numel(), featsB_specific_shapes.numel(), feats_dim))
    dist = dist.reshape(batch_shapes + featsA_specific_shapes + featsB_specific_shapes)

    dist_intermed_subscripts = ''.join(batch_subscripts + featsA_specific_subscripts + featsB_specific_subscripts)
    dist_out_subscripts = ''.join([*output_subscripts])
    #logger.info(f'dist subscripts change: {dist_intermed_subscripts} -> {dist_out_subscripts}')
    dist = torch.einsum(dist_intermed_subscripts + '->' + dist_out_subscripts, dist)
    return dist