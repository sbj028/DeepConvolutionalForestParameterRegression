from torch.fft import fft, fftn
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy, mse_loss

"""
Collection of losses used in the project
"""


class AGBPixelLossImpute:
    """
    Pixel-based losses, i.e. MAE=L1 loss and MSE/L2 loss
    """

    def __init__(self, l1_l2='l1'):
        super(AGBPixelLossImpute, self).__init__()
        self.l1_l2 = l1_l2

        if self.l1_l2 == 'l1':
            self.pixel_loss = nn.L1Loss()  # MAE
        elif self.l1_l2 == 'l2':  # Squared l2 loss
            self.pixel_loss = nn.MSELoss()
        else:
            raise NotImplementedError(
                'Loss type {} is not recognized.'.format(l1_l2))

    def __call__(self, fake_B, real_B, gr_mask, impute_w, device):
        """
        Calculate pixel loss given prediction (fake_B) and true target (real_B)
        :param fake_B:  (tensor) Output from tex. generator net.
        :param real_B:  (tensor) Real target corresponding to input
        :param gr_mask: (tensor) Binary mask where "1" represent pixels that have been imputed with GR measurements.
                        "0" represent no imputing.
        :param impute_w: (int) Decides how much the imputing loss adds to total loss.
        :param device   if cpu/gpu etc.
        :return: The calculated pixel-based loss
        """
        # To get the target to have equally many channels as the fake_B (and the input)
        real_B_tensor = real_B.expand_as(fake_B)

        # Mask real and fake for imputing loss computed on GR:
        fake_B_masked_gr = fake_B * gr_mask
        real_B_tensor_masked_gr = real_B_tensor * gr_mask

        # Regular AGB pixel loss computed for all pixels in patch:
        reg_loss = self.pixel_loss(fake_B, real_B_tensor).to(device)

        # Compute GR imputing loss  and weight it with impute_w:
        gr_impute_loss = (self.pixel_loss(fake_B_masked_gr,  real_B_tensor_masked_gr) * impute_w).to(device)

        loss_tot = reg_loss + gr_impute_loss
        return loss_tot


class VolumePixelLossImpute:
    """
    Class that applies weighted pixel loss when there is a forest mask, and impute loss for points in GR mask

    Pixel-based losses, i.e. MAE=L1 loss and MSE/L2 loss to be used together with weight masks:
    """

    def __init__(self, l1_l2='l1'):
        """
        :param l1_l2: (str)     -- If want to use l1 or l2 norm.
        """
        super(VolumePixelLossImpute, self).__init__()
        self.l1_l2 = l1_l2

        if self.l1_l2 == 'l1':
            self.pixel_loss = nn.L1Loss()  # MAE
        elif self.l1_l2 == 'l2':  # Squared l2 loss
            self.pixel_loss = nn.MSELoss()
        else:
            raise NotImplementedError(
                'Loss type {} is not recognized.'.format(l1_l2))

    def __call__(self, fake_B, real_B, vol_mask, gr_mask, impute_w, device):
        """
        Calculate weighted pixel loss given prediction (fake_B), true target (real_B), and weight mask of bools
        :param fake_B:  (tensor) Output from tex. generator net.
        :param real_B:  (tensor) Real target corresponding to input
        :param vol_mask: (tensor)  Mask of 1s and 0s. Pixels with "1" e.g. represents location with S1a coverage and
                                        e.g. timber volume data. "0" represents location without S1A and/or without
                                        timber volume data.
        :param gr_mask: (tensor) Binary mask where "1" represent pixels that have been imputed with GR measurements.
                        "0" represent no imputing.
        :param impute_w: (int) Decides how much the imputing loss adds to total loss.
        :param device:               --if cpu/gpu etc.

        :return: The calculated pixel-based loss
        """

        # To get the target to have equally many channels as the fake_B (and the input)
        real_B_tensor = real_B.expand_as(fake_B)

        # Mask real and fake for imputing loss computed on Vol:
        fake_B_masked_vol = fake_B * vol_mask
        real_B_tensor_masked_vol = real_B_tensor * vol_mask

        # Mask real and fake for imputing loss computed on GR:
        fake_B_masked_gr = fake_B * gr_mask
        real_B_tensor_masked_gr = real_B_tensor * gr_mask

        # Compute timber volume loss computed for all pixels in forest volume mask:
        vol_loss = self.pixel_loss(fake_B_masked_vol,  real_B_tensor_masked_vol).to(device)

        # Compute GR imputing loss  and weight it with impute_w:
        gr_impute_loss = (self.pixel_loss(fake_B_masked_gr,  real_B_tensor_masked_gr) * impute_w).to(device)

        loss_tot = vol_loss + gr_impute_loss

        return loss_tot


class AGBSpectralLossImpute():
    """
    Enable definition of spectral loss, and different versions of it.
    """

    def __init__(self, spec_loss_name):
        """
        Initialize the Spectral loss class.

        :param spec_loss_name (str) Type of spectral loss function. Currently only 'fft' loss is supported.

        """
        super(AGBSpectralLossImpute, self).__init__()
        self.spec_loss_name = spec_loss_name

        if self.spec_loss_name == 'fft':
            self.spec_loss = FFT_Loss()
        else:
            raise NotImplementedError(f"Spectral loss {self.spec_loss_name} not implemented")

    def __call__(self, fake_B, real_B, gr_mask, impute_w, device):
        """
        Calculate pixel loss given prediction (fake_B) and true target (real_B)
        :param fake_B: (tensor) Output from tex. generator net.
        :param real_B:     (tensor) Real target corresponding to input
        :param gr_mask: (tensor) Binary mask where "1" represent pixels that have been imputed with GR measurements.
                        "0" represent no imputing.
        :param impute_w: (int) Decides how much the imputing loss adds to total loss.
        :param device   if cpu/gpu etc.
        :return: The calculated pixel-based loss
        """
        # To get the target to have equally many channels as the fake_B (and the input)
        real_B_tensor = real_B.expand_as(fake_B)

        # Mask real and fake for imputing loss computed on GR:
        fake_B_masked_gr = fake_B * gr_mask
        real_B_tensor_masked_gr = real_B_tensor * gr_mask

        # Regular AGB pixel loss computed for all pixels in patch:
        reg_loss = self.spec_loss(fake_B, real_B_tensor, device)

        # Compute GR imputing loss  and weight it with impute_w:
        gr_impute_loss = (self.spec_loss(fake_B_masked_gr, real_B_tensor_masked_gr, device) * impute_w)

        spec_loss = reg_loss + gr_impute_loss

        return spec_loss


class VolumeSpectralLossImpute:
    """
    Enable definition of weighted  spectral loss. Used when we have a forest mask (not for GR mask).
    So far is only the weighted fft loss implemented.
    """

    def __init__(self, spec_loss_name):
        """
        Initialize the Spectral loss class.
        Parameters:

        spec_loss_name (str) -- the type of spectral loss function. Currently only 'fft' loss is supported.
        """
        super(VolumeSpectralLossImpute, self).__init__()
        self.spec_loss_name = spec_loss_name

        if self.spec_loss_name == 'fft':
            self.spec_loss = FFT_Loss()
        else:
            raise NotImplementedError(f"Spectral loss {self.spec_loss_name} not implemented, use weighted FFT loss.")

    def __call__(self, fake_B, real_B, vol_mask, gr_mask, impute_w, device):
        """
        Calculate pixel loss given prediction (fake_B) and true target (real_B)
        :param fake_B: (tensor) Output from tex. generator net.
        :param real_B:     (tensor) Real target corresponding to input
        :param vol_mask: (tensor)  Mask of 1s and 0s. Pixels with "1" e.g. represents location with S1a coverage and
                                e.g. timber volume data. "0" represents location without S1A and/or without
                                timber volume data.
        :param gr_mask: (tensor) Binary mask where "1" represent pixels that have been imputed with GR measurements.
                        "0" represent no imputing.
        :param impute_w: (int) Decides how much the imputing loss adds to total loss.
        :param device   if cpu/gpu etc.
        :return: The calculated pixel-based loss
        """
        # To get the target to have equally many channels as the fake_B (and the input)
        real_B_tensor = real_B.expand_as(fake_B)

        # Mask real and fake for imputing loss computed on Vol:
        fake_B_masked_vol = fake_B * vol_mask
        real_B_tensor_masked_vol = real_B_tensor * vol_mask

        # Mask real and fake for imputing loss computed on GR:
        fake_B_masked_gr = fake_B * gr_mask
        real_B_tensor_masked_gr = real_B_tensor * gr_mask

        # Compute timber volume loss computed for all pixels in forest volume mask:
        vol_loss = self.spec_loss(fake_B_masked_vol, real_B_tensor_masked_vol, device)

        # Compute GR imputing loss  and weight it with impute_w:
        gr_impute_loss = self.spec_loss(fake_B_masked_gr, real_B_tensor_masked_gr, device) * impute_w

        spec_loss = vol_loss + gr_impute_loss

        return spec_loss


class FFT_Loss:
    """
    The FFT loss from https://github.com/uitml/fourier-odyssey/blob/main/src/losses.py

    The FFT loss can be applied to both the regular case (with only real and target patches) and to the masked case
    (when real and target patches comes together with a mask of 1s and 0s to distinguish between which pixels that the
    network should train on).
    """

    def __init__(self):
        """
        Regular FFT Loss without weighting mask
        """
        super(FFT_Loss, self).__init__()

    def __call__(self, fake_B, real_B, device):
        """
        Calculate pixel loss given prediction (fake_B) and true target (real_B)
        :param fake_B: (tensor) Output from tex. generator net.
        :param real_B:     (tensor) Real target corresponding to input
        :return: The calculated pixel-based loss
        """

        # The Fourier transform:
        fake_B_fft = fft(fake_B)
        real_B_fft = fft(real_B)

        fft_loss_real = mse_loss(fake_B_fft.real, real_B_fft.real, reduction="sum")
        fft_loss_imag = mse_loss(fake_B_fft.imag, real_B_fft.imag, reduction="sum")

        self.loss_freq = fft_loss_real + fft_loss_imag
        return self.loss_freq.to(device)
