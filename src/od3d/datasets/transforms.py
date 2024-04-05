import numpy as np
import torch
import torchvision

class Resize:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transform = torchvision.transforms.Resize(size=(height, width))

    def __call__(self, sample):
        assert len(sample['img'].shape) == 4
        b, c, h, w = sample['img'].shape
        if h != self.height or w != self.width:
            sample['img'] = self.transform(sample['img'])
            if 'kp' in sample:
                assert len(sample['kp'].shape) == 3
                sample['kp'][:, :, 0] *= self.width / w
                sample['kp'][:, :, 1] *= self.height / h
        return sample


class ToTensor:
    def __init__(self):
        self.trans = torchvision.transforms.ToTensor()

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        if "kpvis" in sample and not isinstance(sample["kpvis"], torch.Tensor):
            sample["kpvis"] = torch.Tensor(sample["kpvis"])
        if "kp" in sample and not isinstance(sample["kp"], torch.Tensor):
            sample["kp"] = torch.Tensor(sample["kp"])
        return sample


class Normalize:
    def __init__(self):
        self.trans = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        return sample


def hflip(sample):
    sample["img"] = torchvision.transforms.functional.hflip(sample["img"])
    if 'kp' in sample:
        sample["kp"][:, 1] = sample["img"].size[0] - sample["kp"][:, 1] - 1
    sample["azimuth"] = np.pi * 2 - sample["azimuth"]
    sample["theta"] = np.pi * 2 - sample["theta"]
    raise NotImplementedError("Horizontal flip is not tested.")

    return sample


class RandomHorizontalFlip:
    def __init__(self):
        self.trans = torchvision.transforms.RandomApply([lambda x: hflip(x)], p=0.5)

    def __call__(self, sample):
        sample = self.trans(sample)
        return sample


class ColorJitter:
    def __init__(self):
        self.trans = torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.4, hue=0
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        return sample
