import torch

from od3d.cv.select import batched_indexMD_select, index_MD_to_1D

def test_index_MD_to_1D():

    indexMD = torch.LongTensor([1, 10]).view(1, 1, -1).expand(100, 8, 2)

    inputMD = torch.zeros(size=(100, 50, 30))
    dims = [1, 2]
    index_1D = index_MD_to_1D(indexMD=indexMD, inputMD=inputMD, dims=dims)
    assert (index_1D == 60).all()

    indexMD = torch.LongTensor([10, 1]).view(1, 1, -1).expand(100, 8, 2)
    index_1D = index_MD_to_1D(indexMD=indexMD, inputMD=inputMD, dims=dims)
    assert (index_1D == 501).all()

    indexMD = torch.LongTensor([10, 1, 1]).view(1, 1, 1, -1).expand(100, 8, 5, 3)
    inputMD = torch.zeros(size=(100, 50, 30, 20)) # 50*30 + 30 + 1
    dims = [1, 2, 3]
    index_1D = index_MD_to_1D(indexMD=indexMD, inputMD=inputMD, dims=dims)

    assert (index_1D == 15031).all()