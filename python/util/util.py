import os
import torch
import numpy as np

"""
Module containing simple help functions.
"""


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def gen_image_rescale(image_arr_in, omin, omax, nmin, nmax, rounded=False):
    """
    General function for rescaling an image from old range = [omin, omax]
    to new range = [nmax, nmin]. The function can for example be used to map
    pixel values from any range to [0, 255] (uint8)

    :param image_arr_in: (np.array)   --image_arr to be rescaled
    :param omin: (int)                --old min, i.e. min(image_arr)
    :param omax: (int)                --old max, i.e. max(image_arr)
    :param nmin: (int)                --new min, e.g. 0
    :param nmax: (int)                --new max, e.g. 255
    :param rounded: (bool)            --If True, round scaled pixels to nearest integer.
    :return: scaled_im
    """
    image_arr = np.copy(image_arr_in)  # Needed to not change values directly on image_arr_in
    scaled_im = nmin + ((nmax - nmin) / (omax - omin)) * (image_arr - omin)

    if not rounded:  # Return scaled pixels in numpy array as it is
        return scaled_im
    elif rounded:  # Round pixels in numpy array to nearest integer and return
        return np.rint(scaled_im)
    else:
        raise NotImplementedError(f'rounded = {rounded} need to be a bool.')


def tensor2im(input_image, relu, imtype=np.uint8, normalize=False, rounded=False, n_range=[0, 255]):
    """"Converts a Tensor array into a numpy image array.

    :param input_image (tensor)   -- The input image tensor array
    :param relu (bool)            -- If use relu as output activation. See opt.relu.
    :param imtype (type)          -- The desired type of the converted numpy array
    :param normalize (bool)       -- To plot images withwandb, need to normalize in range [0,255] to avoid noisy images
    :param rounded (bool)         -- If want to round the normalized pixels to closest int, set to True for wandb plots
    :param n_range (list of 2 ints) -- New range for pixels after normalization, i.e. [new_min, new_max]
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # Convert first image in batch to numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array

        """
        Torch tensor image is of shape (C,H,W) (i.e. (0=C, 1=H, 2=W) while numpy images/arrays have the following form: 
        (H,W,C). The following convert the torch tensor image to numpy format with ordinary numpy shape.
        """
        # To be able to convert greyscale images to PIL, need to be converted to fake RGB first
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        """
        Since using reLU activation function, the generated output images are in range [0, inf].
        Only considering imtype = np.float32, and no scaling of the output, as pixel values hopefully reflects the 
        AGB-values.  
        """
        if imtype == np.uint8 and not relu:
            image_numpy = (np.transpose(image_numpy,
                                        (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling to [-1,1]

        elif imtype == np.float32 and relu:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)))  # No normalization here

        elif imtype == np.uint8 and relu and normalize:  # Normalize in n_range
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)))  # No normalization here
            image_numpy = gen_image_rescale(image_numpy, image_numpy.min(), image_numpy.max(),
                                            n_range[0], n_range[1], rounded=rounded)

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

    return image_numpy.astype(imtype)
