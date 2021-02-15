#ormalization = norms, xy_crop = xy_crop, t_crop = t_crop, augment = augment, xy_interval = xy_interval, t_skip = t_interval

import random, pdb, os, torch
from PIL import Image
import numpy as np

class TransformJPEGs(object):
    def __init__(self, xy_crop, t_crop, t_interval, augment = False, mean = None, std = None):
        self.xy_crop = xy_crop
        self.t_crop = t_crop
        self.t_interval = t_interval
        self.augment = augment
        self.mean = mean
        self.std = std

        self.transforms = []
        self.transforms.append(RandomXYCenterCropJPGs(xy_crop, augment))
        self.transforms.append(RandomTCenterCropJPGs(t_crop, t_interval, augment))
        self.transforms.append(ToTensor())
        self.transforms.append(Normalize(mean, std))

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img

class RandomXYCenterCropJPGs(object):

    def __init__(self, xy_crop, augment):
        # This crops first and then performs the interval

        self.xy_crop = xy_crop
        self.augment = augment

    def __call__(self, img_stack):

        img = img_stack[0]

        self.x1_0 = int((img.size[0] - self.xy_crop)/2) # starting point for crop if there is no augmentation
        self.y1_0 = int((img.size[1] - self.xy_crop)/2) # starting point for crop if there is no augmentation

        if self.augment:
            self.max_augment_x = min(int(self.xy_crop/2), int((img.size[0] - self.xy_crop)/2))
            self.max_augment_y = min(int(self.xy_crop/2), int((img.size[1] - self.xy_crop)/2))
            self.p = random.random()
        else:
            self.max_augment_x = 0
            self.max_augment_y = 0

        self.tl_x = random.randint(-1*self.max_augment_x, self.max_augment_x)
        self.tl_y = random.randint(-1*self.max_augment_y, self.max_augment_y)

        x1 = self.tl_x + self.x1_0
        y1 = self.tl_y + self.y1_0
        x2 = x1 + self.xy_crop
        y2 = y1 + self.xy_crop

        for i, img in enumerate(img_stack):
            if self.p < 0.5:
                img_stack[i] = img_stack[i].crop((x1, y1, x2, y2))
            else:
                img_stack[i] = img_stack[i].crop((x1, y1, x2, y2)).transpose(method=Image.FLIP_LEFT_RIGHT)
        return img_stack

class RandomTCenterCropJPGs(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, t_crop, t_interval = 1, augment = False):
        
        self.t_crop = t_crop
        self.t_interval = t_interval
        self.augment = augment

    def __call__(self, img_stack):

        self.t_0 = int((len(img_stack) - self.t_crop)/2) # starting point for crop if there is no augmentation

        if self.augment:
            self.max_augment_t = min(int(self.t_crop/2), int((len(img_stack) - self.t_crop)/2))
        else:
            self.max_augment_t = 0

        self.tl_t = random.randint(-1*self.max_augment_t, self.max_augment_t)

        t1 = self.tl_t + self.t_0
        t2 = t1 + self.t_crop

        out = img_stack[t1:t2:self.t_interval]
        return out

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=1):
        self.norm_value = norm_value

    def __call__(self, img_stack):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        output = []

        for pic in img_stack:

            if isinstance(pic, np.ndarray):
                # handle numpy array
                img = torch.from_numpy(pic.copy().transpose((2, 0, 1)))
                # backward compatibility
                return img.float().div(self.norm_value)

            # handle PIL Image
            if pic.mode == 'I':
                img = torch.from_numpy(np.array(pic, np.int32, copy=False))
            elif pic.mode == 'I;16':
                img = torch.from_numpy(np.array(pic, np.int16, copy=False))
            else:
                img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            elif pic.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(img, torch.ByteTensor):
                output.append(img.float().div(self.norm_value))
            else:
                output.append(img)

        output = torch.stack(output, 0).permute(1, 0, 2, 3)
        return output

    def randomize_parameters(self):
        pass

class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean = None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        if self.mean is None:
            mean = tensor.mean(axis = (1,2,3))
            std = tensor.std(axis = (1,2,3))
        else:
            mean = self.mean
            std = self.std

        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

