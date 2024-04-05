import torch

# gather: input, dim, index -> return tensor with source indices (on right side)
#   -> index = source-index
#   -> number of dimensions for input, and index must be the same
#   -> out will have same shape as index
#   -> useful for index2value

# scatter: input, dim, index, src -> return tensor with target indices (on left side)
#   -> index = target-index
#   -> out will have same shape as input, with override
#   -> backward pass only works for src.shape == index.shape
#   -> useful for index2mask

# masked_scatter: input, mask, source -> return tensor with masked copy of source into target
#   -> mask must be broadcastable for source and target

def append_const_front(pts, dim, value=1.):
    device = pts.device
    dtype = pts.dtype
    if dim == -1:
        dim = pts.dim() - 1
    ones1d = torch.ones(size=list(pts.shape[:dim]) + [1] + list(pts.shape[dim + 1:])).to(device=device, dtype=dtype) * value
    return torch.cat([ones1d, pts], dim=dim)

def append_const_back(pts, dim, value=1.):
    device = pts.device
    dtype = pts.dtype
    if dim == -1:
        dim = pts.dim() - 1
    ones1d = torch.ones(size=list(pts.shape[:dim]) + [1] + list(pts.shape[dim + 1:])).to(device=device, dtype=dtype) * value
    return torch.cat([pts, ones1d], dim=dim)

def batched_index_fill(input, value, index, dim=None):
    """
    Args:
        input: B...x...xIx...
        value: B...xN
        dim (int): dimension of input which should be indexed (I)
        index: B...xN
    """

    if dim is None:
        dim = index.dim()-1

    batch_dims = index.shape[:-1]
    batch_dims_count = len(batch_dims)

    views = batch_dims + torch.Size([1 if i != dim else -1 for i in range(batch_dims_count, input.dim())])
    index = index.view(views)
    if isinstance(value, torch.Tensor):
        value = value.view(views)

    expanse = batch_dims + torch.Size([input.shape[i] if i != dim else -1 for i in range(batch_dims_count, input.dim())]) #
    index = index.expand(expanse)
    if isinstance(value, torch.Tensor):
        value = value.expand(expanse)

    return torch.scatter(input=input, index=index, src=value, dim=dim)

def batched_indexMD_fill(inputMD, indexMD, value, dims=None):
    """
    Args:
        input: B...x...xI1x...xI2...
        dims (List[int]): dimensions of input which should be filled [I1, ..., IM]
        indexMD: B...xNxM
        valueMD: B...xN
    Returns:
        out: B...xNx...
    """

    batch_dims = indexMD.shape[:-2]
    batch_dims_count = len(batch_dims)

    if dims is None:
        dims = batch_dims_count + torch.arange(indexMD.shape[-1])

    assert(len(dims) == indexMD.shape[-1])

    M = len(dims)
    non_batch_non_index_dims_count = inputMD.dim() - batch_dims_count - M

    index1D = index_MD_to_1D(indexMD, inputMD, dims)

    input_permutation_MD_to_1D = torch.LongTensor([i for i in range(batch_dims_count)])
    input_permutation_MD_to_1D = torch.cat([input_permutation_MD_to_1D, torch.LongTensor([d for d in dims])])
    input_permutation_MD_to_1D = torch.cat([input_permutation_MD_to_1D, torch.LongTensor([i for i in set(list(range(inputMD.dim())))-set(input_permutation_MD_to_1D.tolist())])])
    input1D_shape = batch_dims + (-1,)
    if non_batch_non_index_dims_count > 0:
        input1D_shape += torch.Size([inputMD.shape[i] for i in input_permutation_MD_to_1D[-non_batch_non_index_dims_count:]])
    input1D = inputMD.permute(*input_permutation_MD_to_1D).view(input1D_shape)

    out1D = batched_index_fill(input=input1D, index=index1D, value=value)

    return out1D


def batched_index_select(input, index, dim=None):
    """
    Args:
        input: B...x...xIx...
        dim (int): dimension of input which should be indexed (I)
        index: B...xN
    Returns:
        out: B...xNx...
    """
    if dim is None:
        dim = index.dim()-1

    batch_dims = index.shape[:-1]
    batch_dims_count = len(batch_dims)

    views = batch_dims + torch.Size([1 if i != dim else -1 for i in range(batch_dims_count, input.dim())])
    index = index.view(views)

    expanse = batch_dims + torch.Size([input.shape[i] if i != dim else -1 for i in range(batch_dims_count, input.dim())]) #
    index = index.expand(expanse)

    return torch.gather(input, dim, index)

def index_MD_to_1D(indexMD, inputMD, dims):
    """
    Args:
        input: B...x...xI1x...xI2...
        dims (List[int]): dimensions of input which should be indexed [I1, ..., IM]
        indexMD: B...xNxM
    Returns:
        index1D: B...xN
    """

    M = len(dims)
    index1D = torch.zeros_like(indexMD[..., 0])
    for m in range(M):
        if m > 0:
            index1D *= inputMD.shape[dims[m]]
        index1D += indexMD[..., m]
    return index1D




def batched_indexMD_select(inputMD, indexMD, dims=None):
    """
    Args:
        input: B...x...xI1x...xI2...
        dims (List[int]): dimensions of input which should be indexed [I1, ..., IM]
        indexMD: B...xNxM
    Returns:
        out: B...xNx...
    """

    batch_dims = indexMD.shape[:-2]
    batch_dims_count = len(batch_dims)

    if dims is None:
        dims = batch_dims_count + torch.arange(indexMD.shape[-1])

    assert(len(dims) == indexMD.shape[-1])

    M = len(dims)
    non_batch_non_index_dims_count = inputMD.dim() - batch_dims_count - M

    index1D = index_MD_to_1D(indexMD, inputMD, dims)

    input_permutation_MD_to_1D = torch.LongTensor([i for i in range(batch_dims_count)])
    input_permutation_MD_to_1D = torch.cat([input_permutation_MD_to_1D, torch.LongTensor([d for d in dims])])
    input_permutation_MD_to_1D = torch.cat([input_permutation_MD_to_1D, torch.LongTensor([i for i in set(list(range(inputMD.dim())))-set(input_permutation_MD_to_1D.tolist())])])
    input1D_shape = batch_dims + (-1,)
    if non_batch_non_index_dims_count > 0:
        input1D_shape += torch.Size([inputMD.shape[i] for i in input_permutation_MD_to_1D[-non_batch_non_index_dims_count:]])
    input1D = inputMD.permute(*input_permutation_MD_to_1D).view(input1D_shape)

    out1D = batched_index_select(input=input1D, index=index1D)

    return out1D
