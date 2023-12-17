from torch.utils.data import Dataset
from .load_data import make_dataset
from .transforms import get_transform, get_params
import torch
import tifffile
import numpy as np


class RemoteSensingDataset(Dataset):
    """"
    Remote sensing dataset, i.e. input data
    """

    def __init__(self, opt, input_path):
        """
        Init function, include all operations that we want the dataset to use/run once.

        Assume that both input and output data use the same extension, i,e. tiff etc,
        :param opt:
        :param input_path
        """

        self.opt = opt
        self.Sentinel_pths = sorted(make_dataset(input_path, opt.max_dataset_size))  # Get input data pths
        self.input_nc = opt.input_nc
        self.extension = opt.extension  # image extension on input/outut data, i.e. .tiff
        self.convert = opt.convert  # True if want to use torchvision.transforms

    def __len__(self):
        """
        To return length of dataset, used by DataLoader to create batches.
        Assumption: length of input and corresponding output dataset are the same.
        """
        return len(self.Sentinel_pths)

    def __getitem__(self, idx):
        """
        Function that processes and return 1 datapont at a time.

        :param idx:
        :return:
        """
        # Read images given random integer index:
        input_pth = self.Sentinel_pths[idx]

        """
        If image extensions are .tiff or similar read data as in pix2pix. 
        Assumption: length of input and corresponding output dataset are the same.
        """
        if input_pth.endswith('.tiff') or input_pth.endswith('.tif'):

            # Read images as np arrays:
            input_img_tmp = tifffile.imread(input_pth)

            # Transform to tensors:
            input_arr = np.array(input_img_tmp).astype(np.float32)

            # Expand dim of input tensor from H x W to Hx Wx C if needed:
            if len(input_arr.shape) == 2:
                input_arr = np.expand_dims(input_arr, 2)

            h, w, c = input_arr.shape

            if c == self.input_nc:
                """ Case with only input data """
                if self.convert == True or self.convert == 'true' or self.convert == 'True':
                    # Get transformations on input data:
                    transform_params = get_params(self.opt, (h, w))
                    input_trans = get_transform(self.opt, self.input_nc, transform_params,
                                                grayscale=(self.input_nc == 1))
                    input_tensor = input_trans(input_arr)
                    # Return input_tensor
                elif self.convert == False or self.convert == 'false' or self.convert == 'False':
                    # Only return input_tensor
                    input_tensor = torch.from_numpy(
                        np.transpose(np.array(input_arr).astype(np.float32), (2, 0, 1)))

                return {'input': input_tensor, 'input_paths': input_pth}
            else:
                raise ValueError(f"Number of channels in input is {c} and doesn't agree with "
                                 f"arg.input_nc={self.input_nc}")

        else:
            return NotImplementedError(f"Image extension {self.extension} is not implemented, try another one")


class AGBDataset(Dataset):
    """
    AGB dataset, i.e. target dataset.
    """

    def __init__(self, opt, target_path):
        """
        Init function, include all operations that we want the dataset to use/run once.

        Assume that both input and output data use the same extension, i,e. tiff etc,
        :param opt:
        :param target_path
        """

        self.opt = opt
        self.AGB_pths = sorted(make_dataset(target_path, opt.max_dataset_size))  # Get input data pths
        self.target_nc = opt.target_nc
        self.extension = opt.extension  # image extension on input/outut data, i.e. .tiff

    def __len__(self):
        """
        To return length of dataset, used by DataLoader to create batches.
        Assumption: length of input and corresponding output dataset are the same.
        """
        return len(self.AGB_pths)

    def __getitem__(self, idx):
        """
        Function that processes and return 1 datapont at a time.
        Tips from kristoffer:

        :param idx:
        :return:
        """
        # Read images given random integer index:
        target_pth = self.AGB_pths[idx]

        """
        If image extensions are .tiff or similar read data as in pix2pix.
        Assumption: length of input and corresponding output dataset are the same.
        """
        if target_pth.endswith('.tiff') or target_pth.endswith('.tif'):

            # Read images as np arrays:
            target_img_tmp = tifffile.imread(target_pth)

            # Expand dim of target tensor from H x W to Hx Wx C if needed:
            if len(target_img_tmp.shape) == 2:
                target_img_tmp = np.expand_dims(target_img_tmp, 2)

            h, w, c = target_img_tmp.shape

            """
            Check if target tensor comes with both data and mask for timber volume. 
            If this is the case, then c = input_nc + 1
            """

            if c == self.target_nc + 1:
                """ Case with both input data and input nan mask """
                # Transform to tensors:
                target_tensor = torch.from_numpy(
                    np.transpose(np.array(target_img_tmp[:, :, 0]).astype(np.float32), (2, 0, 1)))
                target_mask_tensor = torch.from_numpy(
                    np.transpose(np.array(target_img_tmp[:, :, 1]).astype(np.float32), (2, 0, 1)))
                return {'target': target_tensor, 'target_mask': target_mask_tensor, 'target_paths': target_pth}

            elif c == self.target_nc:
                """ Case with only input data """
                # Transform to tensors:
                target_tensor = torch.from_numpy(np.transpose(np.array(target_img_tmp).astype(np.float32), (2, 0, 1)))
                return {'target': target_tensor, 'target_paths': target_pth}
            else:
                raise ValueError(f"Number of channels in input is {c} and doesn't agree with "
                                 f"arg.input_nc= {self.input_nc} or arg.input_nc +1= {self.input_nc + 1} ")

        else:
            return NotImplementedError(f"Image extension {self.extension} is not implemented, try another one")


class RS_AGBDataset(Dataset):
    def __init__(self, opt, input_path, target_path):
        """
        Load input and target data at the same time, used for training when both are available.
        Init function, include all operations that we want the dataset to use/run once.

        Assumption 1: Both input and output data use the same extension, i,e. tiff etc,
        Assumption 2: The input data is of shape 64x64x3 while the target data is of shape 64x64x1.
.
        :param opt:
        :param input_path
        :param target_path
        """

        self.opt = opt
        self.Sentinel_pths = sorted(make_dataset(input_path, opt.max_dataset_size))  # Get input data pths
        self.AGB_pths = sorted(make_dataset(target_path, opt.max_dataset_size))  # Get input data pths
        self.input_nc = opt.input_nc
        self.target_nc = opt.target_nc
        self.extension = opt.extension  # image extension on input/outut data, i.e. .tiff
        self.convert = opt.convert  # True if want to use torchvision.transforms

    def __len__(self):
        """
        To return length of dataset, used by DataLoader to create batches.
        Assumption: length of input and corresponding output dataset are the same.
        """
        return len(self.AGB_pths)

    def __getitem__(self, idx):
        """
        Function that processes and return 1 datapoint at a time.

        :param idx:
        :return:
        """

        # Read images given random integer index:
        input_pth = self.Sentinel_pths[idx]
        target_pth = self.AGB_pths[idx]

        """
        If image extensions are .tiff/.tif read data as in pix2pix.
        """
        if input_pth.endswith('.tiff') or input_pth.endswith('.tif'):

            # Read images as np arrays:
            input_img_tmp = tifffile.imread(input_pth)
            target_img_tmp = tifffile.imread(target_pth)

            # Transform to np.array of float32:
            input_arr = np.array(input_img_tmp).astype(np.float32)

            # Expand dim of input tensor from H x W to Hx Wx C if needed:
            if len(input_arr.shape) == 2:
                input_arr = np.expand_dims(input_arr, 2)

            h, w, c = input_arr.shape

            # Expand dim of target tensor from H x W to Hx Wx C if needed:
            if len(target_img_tmp.shape) == 2:
                target_img_tmp = np.expand_dims(target_img_tmp, 2)

            h_target, w_target, c_target = np.array(target_img_tmp).astype(np.float32).shape

            """
            1) Check if target tensor comes with both data and ground reference (GR) mask. 
            If this is the case, then c_target = target_nc + 1
            
            2) Check if target tensor comes with both data, timber volume mask and GR mask. 
            If this is the case, then c_target = target_nc + 2
            
            3) Check if target comes without any mask(s) 
            """

            if c == self.input_nc and c_target == self.target_nc + 2:
                """ Case with both input data, target data and target forest mask and target GR mask : """

                """ Input data:"""
                if self.convert == True or self.convert == 'true' or self.convert == 'True':
                    # Get transformations on input data:
                    transform_params = get_params(self.opt, (h, w))
                    input_trans = get_transform(self.opt, self.input_nc, transform_params,
                                                grayscale=(self.input_nc == 1))
                    # Return input_tensor
                    input_tensor = input_trans(input_arr[:, :, :])
                elif self.convert == False or self.convert == 'false' or self.convert == 'False':
                    # Return input_tensor
                    input_tensor = torch.from_numpy(
                        np.transpose(np.array(input_arr).astype(np.float32), (2, 0, 1)))

                """ Target data:"""
                # Target data and target mask converted to tensor:
                target_tmp = np.array(target_img_tmp[:, :, 0]).astype(np.float32)  # Forest target map
                target_forest_mask_tmp = np.array(target_img_tmp[:, :, 1]).astype(np.float32)  # Forest mask
                target_gr_mask_tmp = np.array(target_img_tmp[:, :, 2]).astype(np.float32)  # Forest GR mask

                # Add channel dimension if it missing:
                if len(target_tmp.shape) == 2:
                    target_tmp = np.expand_dims(target_tmp, 2)
                if len(target_forest_mask_tmp.shape) == 2:
                    target_forest_mask_tmp = np.expand_dims(target_forest_mask_tmp, 2)
                if len(target_gr_mask_tmp.shape) == 2:
                    target_gr_mask_tmp = np.expand_dims(target_gr_mask_tmp, 2)

                target_tensor = torch.from_numpy(np.transpose(target_tmp, (2, 0, 1)))
                target_forest_mask_tensor = torch.from_numpy(np.transpose(target_forest_mask_tmp, (2, 0, 1)))
                target_gr_mask_tensor = torch.from_numpy(np.transpose(target_gr_mask_tmp, (2, 0, 1)))

                # Return input_tensor, input_mask_tensor, target_tensor, target_vol_mask_tensor
                return {'input': input_tensor, 'target': target_tensor,
                        'target_vol_mask_tensor': target_forest_mask_tensor,
                        'target_gr_mask_tensor': target_gr_mask_tensor,
                        'input_paths': input_pth, 'target_paths': target_pth}

            elif c == self.input_nc and c_target == self.target_nc + 1:
                """ Case with both input data, target data and target GR mask : """

                """ Input data:"""
                if self.convert == True or self.convert == 'true' or self.convert == 'True':
                    # Get transformations on input data:
                    transform_params = get_params(self.opt, (h, w))
                    input_trans = get_transform(self.opt, self.input_nc, transform_params,
                                                grayscale=(self.input_nc == 1))
                    # Return input_tensor
                    input_tensor = input_trans(input_arr[:, :, :])
                elif self.convert == False or self.convert == 'false' or self.convert == 'False':
                    # Return input_tensor
                    input_tensor = torch.from_numpy(
                        np.transpose(np.array(input_arr).astype(np.float32), (2, 0, 1)))

                """ Target data:"""
                # Target data and target mask converted to tensor:
                target_tmp = np.array(target_img_tmp[:, :, 0]).astype(np.float32)
                target_gr_mask_tmp = np.array(target_img_tmp[:, :, 1]).astype(np.float32)

                # Add channel dimension if it missing:
                if len(target_tmp.shape) == 2:
                    target_tmp = np.expand_dims(target_tmp, 2)
                if len(target_gr_mask_tmp.shape) == 2:
                    target_gr_mask_tmp = np.expand_dims(target_gr_mask_tmp, 2)

                target_tensor = torch.from_numpy(np.transpose(target_tmp, (2, 0, 1)))
                target_gr_mask_tensor = torch.from_numpy(np.transpose(target_gr_mask_tmp, (2, 0, 1)))

                # Return input_tensor, input_mask_tensor, target_tensor, target_gr_mask_tensor
                return {'input': input_tensor, 'target': target_tensor, 'target_gr_mask_tensor': target_gr_mask_tensor,
                        'input_paths': input_pth, 'target_paths': target_pth}

            elif c == self.input_nc and c_target == self.target_nc:
                """ Case with only input data and target data (no mask) """

                if self.convert == True or self.convert == 'true' or self.convert == 'True':
                    # Get transformations on input data:
                    transform_params = get_params(self.opt, (h, w))
                    input_trans = get_transform(self.opt, self.input_nc, transform_params,
                                                grayscale=(self.input_nc == 1))
                    # Return input_tensor
                    input_tensor = input_trans(input_arr)
                elif self.convert == False or self.convert == 'false' or self.convert == 'False':
                    # Return input_tensor
                    input_tensor = torch.from_numpy(
                        np.transpose(np.array(input_arr).astype(np.float32), (2, 0, 1)))

                target_tensor = torch.from_numpy(np.transpose(np.array(target_img_tmp).astype(np.float32), (2, 0, 1)))

                # return input_tensor, target_tensor
                return {'input': input_tensor, 'target': target_tensor, 'input_paths': input_pth,
                        'target_paths': target_pth}

            else:
                raise ValueError(f"The number actual of channels in input/target should either be the same as given by "
                                 f"arg.input_nc/arg.target_nc or as arg.input_nc and arg.target_nc+1. Got number of "
                                 f"channels in input: {c} and number of target channels: {c_target}, which doesn't "
                                 f"agree with given arg.input_nc= {self.input_nc} and arg.target_nc= {self.target_nc}."
                                 f"Given option is not implemented ")

        else:
            return NotImplementedError(f"Image extension {self.extension} is not implemented, try another one")
