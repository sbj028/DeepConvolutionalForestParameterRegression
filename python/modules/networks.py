import functools
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .basicmodelmodule import init_weights

###############################################################################
# Helper Functions
#
# Identity() and get_norm_layer() are borrowed from pix2pix implementation.
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError(f"Normalization layer {norm_type} is not found")
    return norm_layer


def replace_batchnorm(model, norm):
    """
    To replace default BatchNorm in CustomUNet with either InstanceNorm or no normalization (Identity())
    :param model:   CustomUNet
    :param norm: Retrieved from norm in input arg,  [instance | none]
    :return: 
    """
    if norm not in ['batch', 'instance', 'none']:
        raise NotImplementedError(f"Normalization layer {norm} is not implemented, chose 'batch, 'instance'"
                                  f" or 'none'.")
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_batchnorm(module, norm)
        if norm == 'none':
            if isinstance(module, torch.nn.BatchNorm2d):
                model._modules[name] = nn.Identity()
        elif norm == 'instance':
            if isinstance(module, torch.nn.BatchNorm2d):
                model._modules[name] = nn.InstanceNorm2d(module.num_features)

    return model


def init_net_G(net):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    return net


def init_net_D(net, init_type='normal', init_gain=0.02, ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    init_weights(net, init_type, init_gain=init_gain)

    return net


def define_G(target_nc, input_nc, input_shp, netG, pretrainweights, encoder_name="resnet34", enc_depth=4,
             norm='batch'):
    """
    Create the generator used to translate patches from one domain to another:

    :param target_nc: (int)     - Number of channels in target.
    :param enc_depth: (int)     - Number of down (and up) samplings .
    :param encoder_name: (str)  - What encoder to use, eg. resnet 18, resnet34 ++
    :param input_shp: (int)     - Shape of input, eg. for 64x64 image input_shp = 64
    :param input_nc:            - int, # Of Channels in images from input domain
    :param netG:                - str, For the moment, only one generator is possible, i.e. customUNetG
    :param pretrainweights:    - bool, For the moment, only True (i.e. load weights from pretrained ResNetXX model
    :param norm:                - str, from pix2pix, don't know if it is needed.

    :return:
    """
    net = None
    if netG == 'customUnet':
        net = Custom_UNetG(enc_depth, target_nc, input_nc, input_shp, encoder_name, norm,
                           pretrain_weights=pretrainweights)
    else:
        raise NotImplementedError(f"The generator model {netG} is not implemented, please set it to 'customUNetG'.")
    return init_net_G(net)


def define_D(input_nc, ndf, netD, gan_mode, input_shp, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02):
    """ Create a discriminator, based on Pix2Pix implementation in:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        gan_mode(str)      -- gan_mode is only added as an argument to know if we should use batch norm/instance norm
                            for vanilla or lsgan
        input_shp (in)     -- input shape of image, i.e. width/height.
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    if gan_mode in ['lsgan', 'vanilla']:
        norm_layer = get_norm_layer(norm_type=norm)
        if netD == 'basic':  # default PatchGAN classifier
            net = NLayerDiscriminator(input_nc, gan_mode, input_shp, ndf, n_layers=3, norm_layer=norm_layer)
        elif netD == 'n_layers':  # more options
            net = NLayerDiscriminator(input_nc, gan_mode, input_shp, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
        elif netD == 'pixel':  # classify if each pixel is real or fake
            net = PixelDiscriminator(input_nc, gan_mode, input_shp, ndf, norm_layer=norm_layer)
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    else:
        raise NotImplementedError('Gan mode name [%s] is not implemented' % gan_mode)
    # return init_net(net, init_type, init_gain, gpu_ids)
    return init_net_D(net, init_type, init_gain)


##############################################################################
# Classes
##############################################################################
class Custom_UNetG(nn.Module):

    def __init__(self, enc_depth, target_nc, input_nc, input_shp, encoder_name, norm, pretrain_weights=True):
        """

        :param enc_depth (int):             E.g number of downsamplings layers. Probably between 3,4,5
        :param input_nc (int)               Number of channels in input
        :param input_shape (list):          Number of pixels in width/height, assuming W=H
        :param encoder_name (str):          Check https://segmentation-modelspytorch.readthedocs.io/en/latest/index.html
                                            for alternatives to "resnet34".
        :param norm (str):                  Which normalization to use, e.g. batch, instance or none.
        :param pretrain_weights (bool):     True if want to use imagenet-weights, False if they should be initialized
                                            randomly.
        """
        super(Custom_UNetG, self).__init__()

        self.pretrain_weights = pretrain_weights
        self.input_nc = input_nc
        self.target_nc = target_nc

        # Function that calculates average of weights along the channel axis
        def avg_wts(inp_c_weights):
            average_weights = torch.mean(inp_c_weights, axis=1)

            # Add single channel dimension to the weights:
            average_weights = torch.unsqueeze(average_weights, 1)
            return average_weights

        # To Reinitialize the weights of the last conv layer:
        def init_weights(out_tensor_weight):
            weights = torch.nn.init.kaiming_normal_(out_tensor_weight, a=0, mode='fan_in', nonlinearity='relu')
            return nn.parameter.Parameter(weights)

        # Define if pretrained weights should be used or not:
        if pretrain_weights == True or pretrain_weights == 'True' or pretrain_weights == 'true':
            encoder_weights = "imagenet"
        elif pretrain_weights == None or pretrain_weights == 'None' or pretrain_weights == 'none':
            encoder_weights = None
        else:
            raise NotImplementedError(f"Got pretrain_weights={pretrain_weights}, but must be either 'True' or 'None'.")

        # Define decoder channels, based on encoder depth and input H and W
        d = []
        d_tmp = input_shp
        for i in range(enc_depth):
            d.append(int(d_tmp))
            d_tmp /= 2

        # Load Unet segmentaton net with pretrained imagenet weights, for documentation see:
        # https://segmentation-modelspytorch.readthedocs.io/en/latest/index.html
        if pretrain_weights:
            # activation arg is the ReLU since it worked well in paper 1
            self.custom_net = smp.Unet(encoder_name=encoder_name, encoder_depth=enc_depth,
                                       encoder_weights=encoder_weights, decoder_channels=d, in_channels=3,
                                       activation=nn.ReLU, aux_params=None)
            if self.input_nc != 3:

                # Check if number of input channels are different from 3:
                if self.input_nc == 9:
                    """
                    Change the number of input-channels in the first layer from 3 pretrained channels (image net) to 9
                    by re-using the weights of the 3 first channels 3 times:
                    """

                    weights_conv1 = self.custom_net.encoder.conv1.weight.clone()  # Original imagenet weights
                    self.custom_net.encoder.conv1.in_channels = 9  # Enable input with 9 channels

                    # Concatenate the 3 initial weight channels with it self 3 times:
                    cat_weights = torch.cat((weights_conv1, weights_conv1, weights_conv1), dim=1)

                    # Update weights in the net:
                    self.custom_net.encoder.conv1.weight = nn.parameter.Parameter(cat_weights)

                elif self.input_nc == 4:

                    """
                    Change the number of input-channels in the first layer from 3 pretrained channels (image net) to 4:
                    """
                    weights_conv1 = self.custom_net.encoder.conv1.weight.clone()  # Original imagenet weights
                    self.custom_net.encoder.conv1.in_channels = 4  # Enable input with 4 channels

                    # Concatenate the 3 previous layers with channels with the new weight channels,
                    # that is an average of the three first:
                    cat_weights = torch.cat((weights_conv1, avg_wts(weights_conv1)), dim=1)

                    # Update weights in the net:
                    self.custom_net.encoder.conv1.weight = nn.parameter.Parameter(cat_weights)

                elif self.input_nc == 2:
                    # Perform one initial conv2d to increase input channels from 2 to 3 so that the original
                    # imagenet weights can be applied. Following suggestion 2 in
                    # https://segmentation-models.readthedocs.io/en/latest/tutorial.html#quick-start:

                    self.pre_conv2d = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1)

                elif self.input_nc == 1:
                    # Take the average of all weights from each of the three channels and use this as input

                    weights_conv1 = self.custom_net.encoder.conv1.weight.clone()  # Original imagenet weights
                    self.custom_net.encoder.conv1.in_channels = 1  # Same as the row above

                    avg_inp_weights = avg_wts(weights_conv1)

                    # Update weights in the net:
                    self.custom_net.encoder.conv1.weight = nn.parameter.Parameter(avg_inp_weights)

                else:
                    raise NotImplementedError(f"Got number of input_channels = {self.input_nc}, but model only accepts"
                                              f" input_channels = 1, 2, 3, 4 and 9.")
        elif not pretrain_weights:
            # Can use arbitrary number of channels as the weights are initialized randomly.

            # activation arg is the ReLU
            self.custom_net = smp.Unet(encoder_name=encoder_name, encoder_depth=enc_depth,
                                       encoder_weights=encoder_weights, decoder_channels=d, in_channels=self.input_nc,
                                       activation=nn.ReLU, aux_params=None)

        """
        Re-initialize the weights of the last layer:
        """
        last_weights = torch.empty(self.custom_net.segmentation_head._modules['0'].weight.shape)
        last_weights_init = init_weights(last_weights)

        """
        We want the output to have equally many channels as the target has. 
        """
        # Update the weights:
        if self.target_nc == 1:
            # Since original segmentation head maps to 1 channel output, keep as it is but update weights to be random.
            self.custom_net.segmentation_head._modules['0'].weight = last_weights_init
        elif self.target_nc > 1:
            # Update # channels in output layer:
            out_ks = self.custom_net.segmentation_head._modules['0'].kernel_size
            out_s = self.custom_net.segmentation_head._modules['0'].stride
            out_p = self.custom_net.segmentation_head._modules['0'].padding
            in_ch = self.custom_net.segmentation_head._modules['0'].in_channels

            self.custom_net.segmentation_head._modules['0'] = \
                nn.Conv2d(in_ch, self.target_nc, kernel_size=out_ks, stride=out_s, padding=out_p)
        else:
            raise ValueError(f"Got target_nc={self.target_nc}, but need to be 1 or larger.")

        """
        Check which normalization method that should be used, and update if it's not BatchNormalization: 
        """
        if norm != 'batch':
            self.custom_net = replace_batchnorm(self.custom_net, norm)

    def forward(self, input):
        # Forward pass, the encoder and decoder of the UNet already have all skip connections implemented
        if self.input_nc == 2 and self.pretrain_weights:

            # In the case with 2 channels and use of pre-trained weights, do as suggested in step 2 in
            # https://segmentation-models.readthedocs.io/en/latest/tutorial.html#quick-start
            pre_layer = self.pre_conv2d(input)
            return self.custom_net(pre_layer)
        else:
            # Do regular forward.
            return self.custom_net(input)


class NLayerDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator
    Based on PatchGAN discriminator from
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc, gan_mode, input_shp=64, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            input_shp (int) -- Width and height of input image. Assume that width=height.
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer

        """
        if gan_mode in ['lsgan', 'vanilla']:

            super(NLayerDiscriminator, self).__init__()
            if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

            kw = 4
            padw = 1
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
            nf_mult = 1
            for n in range(1, n_layers):  # gradually increase the number of filters
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

            sequence += [
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
            self.model = nn.Sequential(*sequence)

        else:
            raise NotImplementedError('Gan mode [%s] is not implemented' % self.gan_mode)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class GANLossImputeAGB(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports lsgan and vanilla.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLossImputeAGB, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'gan mode {self.gan_mode} is not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, gr_mask, impute_w):
        """Calculate loss given Discriminator's output and grount truth labels.

        :param prediction:  (tensor) Tpyically the prediction output from a discriminator
        :param target_is_real (bool) If the ground truth label is for real images or fake images
        :param gr_mask: (tensor) Binary mask where "1" represent pixels that have been imputed with GR measurements.
                    "0" represent no imputing.
        :param impute_w: (int) Decides how much the imputing loss adds to total loss.
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)

            # Mask real and fake for imputing loss:
            prediction_masked_gr = prediction * gr_mask
            target_tensor_masked_gr = target_tensor * gr_mask

            # Regular loss computed for all pixels:
            reg_loss = self.loss(prediction, target_tensor)

            # Weighted imputing loss
            gr_impute_loss = self.loss(prediction_masked_gr, target_tensor_masked_gr) * impute_w

            loss_tot = reg_loss + gr_impute_loss
        return loss_tot


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports lsgan and vanilla.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'gan mode {self.gan_mode} is not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)

        return loss


class GANLossImputeVolume(nn.Module):
    """
    GAN loss fro dataset with weighted mask tensor.

    Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLossImputeVolume, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f'gan mode {self.gan_mode} is not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, vol_mask, gr_mask, impute_w):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
            vol_mask (tensor)       -- A weight/mask tensor if use the weighted loss
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)

            # Mask real and fake for imputing loss computed on Vol:
            prediction_masked_vol = prediction * vol_mask
            target_tensor_masked_vol = target_tensor * vol_mask

            # Mask real and fake for imputing loss computed on GR:
            prediction_masked_gr = prediction * gr_mask
            target_tensor_masked_gr = target_tensor * gr_mask

            # Compute timber volume loss computed for all pixels in forest volume mask:
            vol_loss = self.loss(prediction_masked_vol, target_tensor_masked_vol)

            # Compute GR imputing loss  and weight it with impute_w:
            gr_impute_loss = self.loss(prediction_masked_gr, target_tensor_masked_gr) * impute_w

            loss_tot = vol_loss + gr_impute_loss

        return loss_tot


class PixelDiscriminator(nn.Module):
    """
    Defines a 1x1 PatchGAN discriminator (pixelGAN)
    Based on the pixelGAN discriminator from
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc, gan_mode, input_shp=64, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            input_shp (int) -- Width and height of input image. Assume that width=height.
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """

        if gan_mode in ['lsgan', 'vanilla']:
            super(PixelDiscriminator, self).__init__()
            if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

            self.net = [
                nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                norm_layer(ndf * 2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

            self.net = nn.Sequential(*self.net)
        else:
            raise NotImplementedError('Gan mode [%s] is not implemented' % self.gan_mode)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
