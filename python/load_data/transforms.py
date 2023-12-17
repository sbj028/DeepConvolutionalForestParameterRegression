"""
Possible transforms in a function.

"""
import random
import numpy as np
import torchvision.transforms as transforms


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, n_channels, params=None, grayscale=False):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())

    if opt.convert:

        """
        If input_Nc use ImageNet normalization:
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        
        Else use dataset specific mean and std computed per channel before patchify the scene
        """
        transform_list += [transforms.ToTensor()]

        if n_channels == 3:
            transform_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        else:
            transform_list += [transforms.Normalize(mean=opt.norm_mean, std=opt.norm_std)]

    return transforms.Compose(transform_list)


def unnormalize(n_channels):

    if n_channels != 3:
        raise ValueError('Number of channels has to be 3 to use unnormalize module')
    else:
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                       transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                            std=[1., 1., 1.]), ])
        return invTrans
