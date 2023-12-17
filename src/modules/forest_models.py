import torch
import os
from modules.networks import define_G, define_D, get_norm_layer, GANLossImputeAGB, GANLossImputeVolume, GANLoss
from .basicmodelmodule import BasicModelModules
from modules import losses


class AGBForestModel(BasicModelModules):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        By default, we use LSGAN loss, UNet without batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', dataset_mode='aligned')
        if is_train:
            # parser.set_defaults(pool_size=0, gan_mode='vanilla') # This is default
            parser.set_defaults(pool_size=0)

        return parser

    def __init__(self, opt):
        """

        :param opt:

        opt.stage: (int) # Stage 1= pretraining with pixel loss, stage=2 2nd part, training with additional losses
        """

        # # Set device:
        # # TODO: add as args in future.
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        BasicModelModules.__init__(self, opt)
        self.gan_mode = opt.gan_mode
        self.volume = opt.volume
        self.l_type = opt.l_type
        self.spec_loss_name = opt.spec_loss_name
        self.alpha = opt.alpha
        self.gamma = opt.gamma
        self.impute_w = opt.impute_w
        self.input_shp = opt.input_shp
        self.netD_type = opt.netD

        """ Use regular losses : """

        if self.l_type not in ['pixel', 'gan', 'prepixel_gan', 'prepixel_spec', 'prepixel_gan_spec', 'pregan_spec']:
            raise ValueError(f"Got l_type={self.l_type}, but need to be any of ['pixel', 'gan', 'prepixel_gan', "
                             f"'prepixel_spec', 'prepixel_gan_spec', 'pregan_spec'].")

        if not self.volume:
            if self.gan_mode in ['lsgan', 'vanilla']:
                if self.isTrain and self.stage == 1:  # If train +pretrain mode:
                    # Check which loss names that are used:
                    if self.l_type == 'pixel':  # Only pixel loss
                        self.l_names = ['Pre_Pixel', 'G_tot']
                    elif self.l_type == 'gan':  # Baseline Pix2Pix G and D losses.
                        self.l_names = ['Pre_G_GAN', 'Pre_G_Pixel', 'G_tot', 'D_tot']
                    else:
                        raise NotImplementedError(f"l_type must be set to 'pixel' or 'gan' when self.stage={self.stage}.")
                elif self.isTrain and self.stage == 2:  # If train + 2nd stage of training:
                    if self.l_type == 'prepixel_gan':
                        # Pretrain G on Pixel loss, then fine-tune G on all gan losses
                        self.l_names = ['G_GAN', 'G_Pixel', 'G_tot', 'D_tot']
                    elif self.l_type == 'prepixel_spec':
                        # Pretrain G on Pixel loss, then fine-tune G on additional spectral loss
                        self.l_names = ['G_Pixel', 'G_Spec', 'G_tot']
                    elif self.l_type == 'prepixel_gan_spec':
                        # Pretrain G on Pixel loss, then fine-tune G on GAN + spectral loss
                        self.l_names = ['G_GAN', 'G_Pixel', 'G_Spec', 'G_tot', 'D_tot']
                    elif self.l_type == 'pregan_spec':
                        # Pretrain G with GAN fine-tune G on additional spectral loss
                        self.l_names = ['G_GAN', 'G_Pixel', 'G_Spec', 'G_tot', 'D_tot']

            # specify the images you want to save/display. The training/test scripts will
            # call <BaseModel.get_current_visuals>
            self.visual_names = ['real_A', 'fake_B', 'real_B']

            # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
            if self.isTrain and self.stage == 1:
                if self.l_type == 'gan':  # If train +pretrain mode + GAN:
                    self.model_names = ['G', 'D']
                else:  # If no gan loss or test time, only have G
                    self.model_names = ['G']
            elif self.isTrain and self.stage == 2:
                # If use GAN-based losses also want to store the discriminator to the disc:
                if self.l_type == 'prepixel_gan' or self.l_type == 'prepixel_gan_spec' or self.l_type == 'pregan_spec':
                    self.model_names = ['G', 'D']
                else:  # If no gan loss or test time, only have G
                    self.model_names = ['G']

            # Define the Unet-based generator for I2I generation of AGB images:
            self.netG = define_G(opt.target_nc, opt.input_nc, opt.input_shp, opt.netG,
                                 opt.pretrainweights, opt.encoder_name, opt.enc_depth, opt.norm)

            if self.isTrain:  # If train or validation and use GAN loss, also need discriminator.
                if self.l_type == 'gan' or self.l_type == 'prepixel_gan' or self.l_type == 'prepixel_gan_spec' or \
                        self.l_type == 'pregan_spec':
                    # Defining Discriminator without force target_nc=input_nc:
                    self.netD = define_D(opt.input_nc + opt.target_nc, opt.ndf, self.netD_type, self.gan_mode, opt.input_shp,
                                         opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

            """ Pretraining: """
            if self.isTrain and self.stage == 1:  # If train +pretrain mode:

                if self.l_type == 'pixel':  # Only pixel loss
                    """ Pretraining with pixel loss"""

                    self.criterionPretrainPixel = losses.AGBPixelLossImpute(opt.l1_l2)
                    # Initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_G)

                if self.l_type == 'gan':  # Only gan loss

                    """ Pretraining with GAN losses"""
                    if self.netD_type in self.netD_type in ['n_layers', 'basic']:
                        self.criterionGAN = GANLoss(self.gan_mode).to(self.device)
                    else: # PixelDiscriminator
                        self.criterionGAN = GANLossImputeAGB(self.gan_mode).to(self.device)
                    self.criterionPixel = losses.AGBPixelLossImpute(opt.l1_l2)

                    # Initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    if self.gan_mode in ['lsgan', 'vanilla']:
                        """Use the same hyperparameters as in pix2pix"""
                        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizers.append(self.optimizer_G)
                        self.optimizers.append(self.optimizer_D)
                    else:
                        raise NotImplementedError(f"GAN mode {self.gan_mode} not implemented")

            elif self.isTrain and self.stage == 2:  # If pretrain +main training mode:
                """ Main training mode with additional losses: """
                # define loss functions

                """ Pixel + GAN losses: """
                if self.l_type == 'prepixel_gan': # Pretrain G on Pixel loss, then fine-tune G on all gan losses
                    """GAN losses:"""
                    if self.netD_type in self.netD_type in ['n_layers', 'basic']:
                        self.criterionGAN = GANLoss(self.gan_mode).to(self.device)
                    else:  # PixelDiscriminator
                        self.criterionGAN = GANLossImputeAGB(self.gan_mode).to(self.device)
                    self.criterionPixel = losses.AGBPixelLossImpute(opt.l1_l2)

                    # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    if self.gan_mode in ['lsgan', 'vanilla']:
                        """Use the same hyperparameters as in pix2pix"""
                        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizers.append(self.optimizer_G)
                        self.optimizers.append(self.optimizer_D)
                    else:
                        raise NotImplementedError(f"GAN mode {self.gan_mode} not implemented")
                    """ Pixel + Spectral losses: """
                    # Pretrain G on Pixel loss, then fine-tune G on additional spectral loss
                elif self.l_type == 'prepixel_spec':

                    """ Pixel loss"""
                    self.criterionPixel = losses.AGBPixelLossImpute(opt.l1_l2)

                    """ Spectral loss """
                    self.criterionSpectral = losses.AGBSpectralLossImpute(self.spec_loss_name)

                    # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_G)

                    """ GAN + Spectral losses: """
                elif self.l_type == 'pregan_spec': # Pretrain G with GAN fine-tune G on additional spectral loss

                    """ GAN losses"""
                    if self.netD_type in self.netD_type in ['n_layers', 'basic']:
                        self.criterionGAN = GANLoss(self.gan_mode).to(self.device)
                    else:  # PixelDiscriminator
                        self.criterionGAN = GANLossImputeAGB(self.gan_mode).to(self.device)
                    self.criterionPixel = losses.AGBPixelLossImpute(opt.l1_l2)

                    """ Spectral loss """
                    self.criterionSpectral = losses.AGBSpectralLossImpute(self.spec_loss_name)

                    # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    if self.gan_mode in ['lsgan', 'vanilla']:
                        """Use the same hyperparameters as in pix2pix"""
                        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizers.append(self.optimizer_G)
                        self.optimizers.append(self.optimizer_D)
                    else:
                        raise NotImplementedError(f"GAN mode {self.gan_mode} not implemented")

                    """ Pixel + GAN + Spectral losses : """
                    # Pretrain G on Pixel loss, then fine-tune G on GAN + spectral loss
                elif self.l_type == 'prepixel_gan_spec':
                    """ GAN losses"""
                    if self.netD_type in self.netD_type in ['n_layers', 'basic']:
                        self.criterionGAN = GANLoss(self.gan_mode).to(self.device)
                    else:  # PixelDiscriminator
                        self.criterionGAN = GANLossImputeAGB(self.gan_mode).to(self.device)
                    self.criterionPixel = losses.AGBPixelLossImpute(opt.l1_l2)

                    """ Spectral loss """
                    self.criterionSpectral = losses.AGBSpectralLossImpute(self.spec_loss_name)

                    # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    if self.gan_mode in ['lsgan', 'vanilla']:
                        """Use the same hyperparameters as in pix2pix"""
                        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizers.append(self.optimizer_G)
                        self.optimizers.append(self.optimizer_D)
                    else:
                        raise NotImplementedError(f"GAN mode {self.gan_mode} not implemented")
        else:
            raise ValueError(f"Got arg weighted={self.volume}, but must be False to use this model.")

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        # Check if target comes with impute mask mask:
        self.num_input = len(input)
        if self.num_input == 4:
            """ No masks only input and target: """
            self.real_A = input['input'].to(self.device)
            self.real_B = input['target'].to(self.device)
            self.image_paths = input['input_paths']

            # Create a zero mask that ensure that impute_loss is 0 independent on value of self.impute_w
            self.real_B_gr_mask = torch.zeros(self.input_shp, self.input_shp).to(self.device)

        elif self.num_input == 5:
            """ Input, target + GR mask"""
            self.real_A = input['input'].to(self.device)
            self.real_B = input['target'].to(self.device)
            self.real_B_gr_mask = input['target_gr_mask_tensor'].to(self.device)
            self.image_paths = input['input_paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # self.net(A) i.e. the generator model

    def backward_D(self):
        """Calculate GAN loss for the discriminator, based on implementation in:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py

        For gan-loss need to fake the discriminator.
        """
        if self.netD_type in ['n_layers', 'basic']:
            # Apply Gr imputed mask on self.real_A, self.real_B and self.fake_B:
            self.real_A_gr = self.real_A * self.real_B_gr_mask
            self.real_B_gr = self.real_B * self.real_B_gr_mask
            self.fake_B_gr = self.fake_B * self.real_B_gr_mask

            # Fake; stop backprop to the generator by detaching fake_B
            # we use conditional GANs; we need to feed both input and output to the discriminator
            fake_AB_gr = torch.cat((self.real_A_gr, self.fake_B_gr), 1)  # For imputed gr mask

            pred_fake_gr = self.netD(fake_AB_gr.detach())

            # D on fake with imputed GR mask:
            self.loss_D_fake_gr = self.criterionGAN(pred_fake_gr, False)

            # Real
            real_AB_gr = torch.cat((self.real_A_gr, self.real_B_gr), 1)
            pred_real_gr = self.netD(real_AB_gr)

            self.loss_D_real_gr = self.criterionGAN(pred_real_gr, True)

            if self.gan_mode in ['lsgan', 'vanilla']:
                # combine loss and calculate gradients
                self.loss_D_tot = (self.loss_D_fake_gr + self.loss_D_real_gr) * 0.5 * self.impute_w
                self.loss_D_tot.backward()
            else:
                raise NotImplementedError(f"GAN mode name {self.gan_mode} is not implemented.")
        else:
            """ For Pixel GAN"""
            # Fake; stop backprop to the generator by detaching fake_B
            # we use conditional GANs; we need to feed both input and output to the discriminator
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB.detach())

            self.loss_D_fake = self.criterionGAN(pred_fake, False, self.real_B_gr_mask, self.impute_w)

            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)

            self.loss_D_real = self.criterionGAN(pred_real, True, self.real_B_gr_mask, self.impute_w)

            if self.gan_mode in ['lsgan', 'vanilla']:
                # combine loss and calculate gradients
                self.loss_D_tot = (self.loss_D_fake + self.loss_D_real) * 0.5
                self.loss_D_tot.backward()
            else:
                raise NotImplementedError(f"GAN mode name {self.gan_mode} is not implemented.")

    def backward_G(self):
        """
        Calcualte losses for the G network (the "generator")
        :return:
        """
        if self.isTrain and self.stage == 1:  # If train +pretrain mode:
            if self.l_type == 'pixel':

                self.loss_Pre_Pixel = self.criterionPretrainPixel(self.fake_B, self.real_B, self.real_B_gr_mask,
                                                                  self.impute_w, self.device)
                self.loss_G_tot = self.loss_Pre_Pixel

            elif self.l_type == 'gan':
                if self.netD_type in ['n_layers', 'basic']:
                    # Apply GR impute mask on self.real_A, self.real_B and self.fake_B:

                    self.real_A_gr = self.real_A * self.real_B_gr_mask
                    self.fake_B_gr = self.fake_B * self.real_B_gr_mask

                    # Fake; stop backprop to the generator by detaching fake_B
                    # we use conditional GANs; we need to feed both input and output to the discriminator
                    fake_AB_gr = torch.cat((self.real_A_gr, self.fake_B_gr), 1)  # For imputed gr mask

                    pred_fake_gr = self.netD(fake_AB_gr.detach())

                    # D on fake with imputed GR mask and multiply with impute_w:
                    self.loss_Pre_G_GAN = self.criterionGAN(pred_fake_gr, False) * self.impute_w

                else:  # PixelDiscriminator:
                    # For gan-loss need to fake the discriminator.
                    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                    pred_fake = self.netD(fake_AB)
                    self.loss_Pre_G_GAN = self.criterionGAN(pred_fake, True, self.real_B_gr_mask, self.impute_w)

                # Second, G(A) = B
                self.loss_Pre_G_Pixel = self.criterionPixel(self.fake_B, self.real_B, self.real_B_gr_mask,
                                                            self.impute_w, self.device)

                # Combine all generator losses with regularization into the G_tot loss
                self.loss_G_tot = self.alpha * self.loss_Pre_G_GAN + self.loss_Pre_G_Pixel

        elif self.isTrain and self.stage == 2:

            if self.l_type == 'prepixel_gan':
                # Pretrain G on Pixel loss, then fine-tune G on all gan losses

                """GAN losses """
                if self.netD_type in ['n_layers', 'basic']:
                    # Apply GR impute mask on self.real_A, self.real_B and self.fake_B:

                    self.real_A_gr = self.real_A * self.real_B_gr_mask
                    self.fake_B_gr = self.fake_B * self.real_B_gr_mask

                    # Fake; stop backprop to the generator by detaching fake_B
                    # we use conditional GANs; we need to feed both input and output to the discriminator
                    fake_AB_gr = torch.cat((self.real_A_gr, self.fake_B_gr), 1)  # For imputed gr mask

                    pred_fake_gr = self.netD(fake_AB_gr.detach())

                    # D on fake with imputed GR mask and multiply with impute_w:
                    self.loss_G_GAN = self.criterionGAN(pred_fake_gr, False) * self.impute_w

                else:  # PixelDiscriminator:
                    # For gan-loss need to fake the discriminator.
                    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                    pred_fake = self.netD(fake_AB)
                    self.loss_G_GAN = self.criterionGAN(pred_fake, True, self.real_B_gr_mask, self.impute_w)

                # Second, G(A) = B
                self.loss_G_Pixel = self.criterionPixel(self.fake_B, self.real_B, self.real_B_gr_mask, self.impute_w,
                                                        self.device)

                # Combine all generator losses with regularization into the G_tot loss
                self.loss_G_tot = self.alpha * self.loss_G_GAN + self.loss_G_Pixel

            elif self.l_type == 'prepixel_spec':
                # Pretrain G on Pixel loss, then fine-tune G on additional spectral loss

                """Pixel and Spectral loss """
                self.loss_G_Pixel = self.criterionPixel(self.fake_B, self.real_B, self.real_B_gr_mask, self.impute_w,
                                                        self.device)
                self.loss_G_Spec = self.criterionSpectral(self.fake_B, self.real_B, self.real_B_gr_mask, self.impute_w,
                                                          self.device)

                # Combine all generator losses with regularization into the G_tot loss
                self.loss_G_tot = self.loss_G_Pixel + self.gamma * self.loss_G_Spec

            elif self.l_type == 'pregan_spec':
                # Pretrain G with GAN fine-tune G on additional spectral loss

                """GAN losses """
                if self.netD_type in ['n_layers', 'basic']:
                    # Apply GR impute mask on self.real_A, self.real_B and self.fake_B:

                    self.real_A_gr = self.real_A * self.real_B_gr_mask
                    self.fake_B_gr = self.fake_B * self.real_B_gr_mask

                    # Fake; stop backprop to the generator by detaching fake_B
                    # we use conditional GANs; we need to feed both input and output to the discriminator
                    fake_AB_gr = torch.cat((self.real_A_gr, self.fake_B_gr), 1)  # For imputed gr mask

                    pred_fake_gr = self.netD(fake_AB_gr.detach())

                    # D on fake with imputed GR mask and multiply with impute_w:
                    self.loss_G_GAN = self.criterionGAN(pred_fake_gr, False) * self.impute_w

                else:  # PixelDiscriminator:
                    # For gan-loss need to fake the discriminator.
                    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                    pred_fake = self.netD(fake_AB)
                    self.loss_G_GAN = self.criterionGAN(pred_fake, True, self.real_B_gr_mask, self.impute_w)

                # Second, G(A) = B
                self.loss_G_Pixel = self.criterionPixel(self.fake_B, self.real_B, self.real_B_gr_mask, self.impute_w,
                                                        self.device)

                """Spectral loss """
                self.loss_G_Spec = self.criterionSpectral(self.fake_B, self.real_B, self.real_B_gr_mask, self.impute_w,
                                                          self.device)

                # Combine all generator losses losses into the generator loss:
                self.loss_G_tot = (self.alpha * self.loss_G_GAN + self.loss_G_Pixel) + self.gamma * self.loss_G_Spec

            elif self.l_type == 'prepixel_gan_spec':
                # Pretrain G on Pixel loss, then fine-tune G on GAN + spectral loss

                """Spectral loss """
                self.loss_G_Spec = self.criterionSpectral(self.fake_B, self.real_B, self.real_B_gr_mask, self.impute_w,
                                                          self.device)

                """GAN losses """
                if self.netD_type in ['n_layers', 'basic']:
                    # Apply GR impute mask on self.real_A, self.real_B and self.fake_B:

                    self.real_A_gr = self.real_A * self.real_B_gr_mask
                    self.fake_B_gr = self.fake_B * self.real_B_gr_mask

                    # Fake; stop backprop to the generator by detaching fake_B
                    # we use conditional GANs; we need to feed both input and output to the discriminator
                    fake_AB_gr = torch.cat((self.real_A_gr, self.fake_B_gr), 1)  # For imputed gr mask

                    pred_fake_gr = self.netD(fake_AB_gr.detach())

                    # D on fake with imputed GR mask and multiply with impute_w:
                    self.loss_G_GAN = self.criterionGAN(pred_fake_gr, False) * self.impute_w

                else:  # PixelDiscriminator:
                    # For gan-loss need to fake the discriminator.
                    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                    pred_fake = self.netD(fake_AB)
                    self.loss_G_GAN = self.criterionGAN(pred_fake, True, self.real_B_gr_mask, self.impute_w)

                # Second, G(A) = B
                self.loss_G_Pixel = self.criterionPixel(self.fake_B, self.real_B, self.real_B_gr_mask, self.impute_w,
                                                        self.device)

                # Combine all generator losses losses into the generator loss:
                self.loss_G_tot = (self.alpha * self.loss_G_GAN + self.loss_G_Pixel) + self.gamma * self.loss_G_Spec

        # Calculate gradients
        self.loss_G_tot.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # If have a gan-loss, need to update the discriminator.
        if self.l_type == 'gan' or self.l_type == 'prepixel_gan' or self.l_type == 'prepixel_gan_spec' or \
                self.l_type == 'pregan_spec':
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights

            # Update the generator (G)
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate gradients for G
            self.optimizer_G.step()  # update G's weights
        else:
            # Update the generator (G):
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate gradients for G
            self.optimizer_G.step()  # update G's weights


class Test_AGBForestModel(BasicModelModules):
    """ This TesteModel is inspired by the Testmodel() fromn pix2pix.
        It can be used to generate I2I results for only one direction.

        This model will check if opt.dataset_mode is either 'remotesensing' or 'agb', if not it will raise error as
        it doesn't accept both input and target data.

        See the test instruction for more details.
        """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time.
        """
        assert not is_train, 'TestModel cannot be used during training time'

        return parser

    def __init__(self, opt):
        """Initialize the AGBForestModel class for testing.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert (not opt.isTrain)
        BasicModelModules.__init__(self, opt)
        self.volume = opt.volume
        self.test_target_path = opt.test_target_path

        # To ensure that we don't call weighted model:
        if not self.volume:
            # Specify the training losses you want to print out. The training/test scripts  will call
            # <BasicModelModules.get_current_losses>
            self.l_names = []

            # Specify the images you want to save/display. The training/test scripts  will call
            # <BasicModelModules.get_current_visuals>
            if self.test_target_path is None:
                self.visual_names = ['real_A', 'fake_B']
            else:
                self.visual_names = ['real_A', 'fake_B', 'real_B']

            # specify the models you want to save to the disk. The training/test scripts will call
            # <BasicModelModules.save_networks> and <BasicModelModules.load_networks>
            self.model_names = ['G']  # only generator is needed.

            # Define_G from Forest model:
            # False = To not load imagenet weights.
            self.netG = define_G(opt.target_nc, opt.input_nc, opt.input_shp, opt.netG, opt.pretrainweights,
                                 opt.encoder_name, opt.enc_depth, opt.norm)

            if self.test_target_path is None:
                # Ensure that opt.dataset_mode is either 'remotesensing' or 'agb if there are no target path:
                if opt.dataset_mode not in ['remotesensing', 'agb']:
                    raise ValueError(
                        f"Got dataset={opt.dataset_mode} but test model only accepts dataset=['remotesensing', "
                        f"'agb'] if no target path is given")
            else:
                if opt.dataset_mode != 'rs_agb':
                    raise ValueError(
                        f"Got dataset={opt.dataset_mode} but test model with target path only accepts dataset='rs_agb'")



            # assigns the model to self.netG so that it can be loaded
            # please see <BaseModel.load_networks> # TODO update this one
            setattr(self, 'netG', self.netG)  # store netG in self.
        else:
            raise ValueError(f"Got arg weighted={self.volume}, but must be False to use this model.")

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        NB! In test phase never use GR mask
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        """
        if self.test_target_path is None:
            self.real_A = input['input'].to(self.device)
            self.image_paths = input['input_paths']
        else:
            self.real_A = input['input'].to(self.device)
            self.real_B = input['target'].to(self.device)
            self.image_paths = input['input_paths']

    def forward(self):
        """Run forward pass."""
        self.fake_B = self.netG(self.real_A)  # G(real_A)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass


class VolumeForestModel(BasicModelModules):
    """
    This class only use dataset which comes with a weighted mask. It also only uses the weighted losses.

    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training/trainval phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        By default, we use LSGAN loss, UNet without batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', dataset_mode='aligned')
        if is_train:
            # parser.set_defaults(pool_size=0, gan_mode='vanilla') # This is default
            parser.set_defaults(pool_size=0)

        return parser

    def __init__(self, opt):
        """

        :param opt:

        opt.stage: (int) # Stage 1= pretraining with pixel loss, stage=2 2nd part, training with additional losses
        """

        # # Set device:
        # # TODO: add as args in future.
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        BasicModelModules.__init__(self, opt)
        self.gan_mode = opt.gan_mode
        self.volume = opt.volume
        self.l_type = opt.l_type
        self.spec_loss_name = opt.spec_loss_name
        self.alpha = opt.alpha
        self.gamma = opt.gamma
        self.impute_w = opt.impute_w
        self.netD_type = opt.netD

        if self.l_type not in ['pixel', 'gan', 'prepixel_gan', 'prepixel_spec', 'prepixel_gan_spec', 'pregan_spec']:
            raise ValueError(f"Got l_type={self.l_type}, but need to be any of ['pixel', 'gan', 'prepixel_gan', "
                             f"'prepixel_spec', 'prepixel_gan_spec', 'pregan_spec'].")

        """ Use weighted losses and masks: """
        if self.volume:
            if self.gan_mode in ['lsgan', 'vanilla']:
                if self.isTrain and self.stage == 1:  # If train +pretrain mode:
                    # Check which loss names that are used:
                    if self.l_type == 'pixel':  # Only pixel loss
                        self.l_names = ['Pre_Pixel', 'G_tot']
                    elif self.l_type == 'gan':  # Baseline Pix2Pix G and D losses.
                        self.l_names = ['Pre_G_GAN', 'Pre_G_Pixel', 'G_tot', 'D_tot']
                    else:
                        raise NotImplementedError(f"l_type must be set to 'pixel' or 'gan' when self.stage={self.stage}.")
                elif self.isTrain and self.stage == 2:  # If train + 2nd stage of training:
                    if self.l_type == 'prepixel_gan':
                        # Pretrain G on Pixel loss, then fine-tune G on all gan losses
                        self.l_names = ['G_GAN', 'G_Pixel', 'G_tot', 'D_tot']
                    elif self.l_type == 'prepixel_spec':
                        # Pretrain G on Pixel loss, then fine-tune G on additional spectral loss
                        self.l_names = ['G_Pixel', 'G_Spec', 'G_tot']
                    elif self.l_type == 'prepixel_gan_spec':
                        # Pretrain G on Pixel loss, then fine-tune G on GAN + spectral loss
                        self.l_names = ['G_GAN', 'G_Pixel', 'G_Spec', 'G_tot', 'D_tot']
                    elif self.l_type == 'pregan_spec':
                        # Pretrain G with GAN fine-tune G on additional spectral loss
                        self.l_names = ['G_GAN', 'G_Pixel', 'G_Spec', 'G_tot', 'D_tot']

            # specify the images you want to save/display. The training/test scripts will
            # call <BaseModel.get_current_visuals>
            self.visual_names = ['real_A', 'fake_B', 'real_B']

            # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
            if self.isTrain and self.stage == 1:
                if self.l_type == 'gan':  # If train +pretrain mode + GAN:
                    self.model_names = ['G', 'D']
                else:  # If no gan loss or test time, only have G
                    self.model_names = ['G']
            if self.isTrain and self.stage == 2:
                # If use GAN-based losses also want to store the discriminator to the disc:
                if self.l_type == 'prepixel_gan' or self.l_type == 'prepixel_gan_spec' or self.l_type == 'pregan_spec':
                    self.model_names = ['G', 'D']
                else:  # If no gan loss or test time, only have G
                    self.model_names = ['G']

            # Define the ResNet-based generator for I2I generation of Forest prediction map images:
            self.netG = define_G(opt.target_nc, opt.input_nc, opt.input_shp, opt.netG,
                                 opt.pretrainweights, opt.encoder_name, opt.enc_depth, opt.norm)

            if self.isTrain:  # If train or validation and use GAN loss, also need discriminator.
                if self.l_type == 'gan' or self.l_type == 'prepixel_gan' or self.l_type == 'prepixel_gan_spec' or \
                        self.l_type == 'pregan_spec':
                    # Defining Discriminator without force target_nc=input_nc:
                    self.netD = define_D(opt.input_nc + opt.target_nc, opt.ndf, self.netD_type, self.gan_mode,
                                         opt.input_shp, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

            """ Pretraining: """
            if self.isTrain and self.stage == 1:  # If train +pretrain mode:

                if self.l_type == 'pixel':  # Only pixel loss

                    """ Pretraining with pixel loss"""
                    self.criterionPretrainPixel = losses.VolumePixelLossImpute(opt.l1_l2)
                    # Initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_G)

                if self.l_type == 'gan':  # Only GAN loss
                    """ Pretraining with GAN losses"""
                    if self.netD_type in self.netD_type in ['n_layers', 'basic']:
                       self.criterionGAN = GANLoss(self.gan_mode).to(self.device)
                    else:  # PixelDiscriminator
                        self.criterionGAN = GANLossImputeVolume(self.gan_mode).to(self.device)
                    self.criterionPixel = losses.VolumePixelLossImpute(l1_l2=opt.l1_l2)

                    # Initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    if self.gan_mode in ['lsgan', 'vanilla']:
                        """Use the same hyperparameters as in pix2pix"""
                        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizers.append(self.optimizer_G)
                        self.optimizers.append(self.optimizer_D)
                    else:
                        raise NotImplementedError(f"GAN mode {self.gan_mode} not implemented")

            elif self.isTrain and self.stage == 2:  # If pretrain +main training mode:

                """ Main training mode with additional losses: """
                # define loss functions

                """ Pixel + GAN losses: """
                # Pretrain G on Pixel loss, then fine-tune G on all gan losses
                if self.l_type == 'prepixel_gan':

                    """GAN losses:"""
                    if self.netD_type in self.netD_type in ['n_layers', 'basic']:
                        self.criterionGAN = GANLoss(self.gan_mode).to(self.device)
                    else:  # PixelDiscriminator
                        self.criterionGAN = GANLossImputeVolume(self.gan_mode).to(self.device)
                    self.criterionPixel = losses.VolumePixelLossImpute(l1_l2='l1')

                    # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    if self.gan_mode in ['lsgan', 'vanilla']:
                        """Use the same hyperparameters as in pix2pix"""
                        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizers.append(self.optimizer_G)
                        self.optimizers.append(self.optimizer_D)
                    else:
                        raise NotImplementedError(f"GAN mode {self.gan_mode} not implemented")

                    """ Pixel + Spectral losses: """
                    # Pretrain G on Pixel loss, then fine-tune G on additional spectral loss
                elif self.l_type == 'prepixel_spec':

                    """ Pixel loss"""
                    self.criterionPixel = losses.VolumePixelLossImpute(opt.l1_l2)

                    """ Spectral loss """
                    self.criterionSpectral = losses.VolumeSpectralLossImpute(self.spec_loss_name)

                    # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_G)

                    """ GAN + Spectral losses: """
                    # Pretrain G with GAN fine-tune G on additional spectral loss
                elif self.l_type == 'pregan_spec':

                    """ GAN losses"""
                    if self.netD_type in self.netD_type in ['n_layers', 'basic']:
                        self.criterionGAN = GANLoss(self.gan_mode).to(self.device)
                    else:  # PixelDiscriminator
                        self.criterionGAN = GANLossImputeVolume(self.gan_mode).to(self.device)
                    self.criterionPixel = losses.VolumePixelLossImpute(l1_l2='l1')

                    """ Spectral loss """
                    self.criterionSpectral = losses.VolumeSpectralLossImpute(self.spec_loss_name)

                    # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    if self.gan_mode in ['lsgan', 'vanilla']:
                        """Use the same hyperparameters as in pix2pix"""
                        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizers.append(self.optimizer_G)
                        self.optimizers.append(self.optimizer_D)
                    else:
                        raise NotImplementedError(f"GAN mode {self.gan_mode} not implemented")

                    """ Pixel + GAN + Spectral losses or GAN + Pixel + Spectral losses: """
                    # Pretrain G on Pixel loss, then fine-tune G on GAN + spectral loss
                elif self.l_type == 'prepixel_gan_spec':
                    """ GAN losses"""
                    if self.netD_type in self.netD_type in ['n_layers', 'basic']:
                        self.criterionGAN = GANLoss(self.gan_mode).to(self.device)
                    else:  # PixelDiscriminator
                        self.criterionGAN = GANLossImputeVolume(self.gan_mode).to(self.device)
                    self.criterionPixel = losses.VolumePixelLossImpute(opt.l1_l2)

                    """ Spectral loss """
                    self.criterionSpectral = losses.VolumeSpectralLossImpute(self.spec_loss_name)

                    # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                    if self.gan_mode in ['lsgan', 'vanilla']:
                        """Use the same hyperparameters as in pix2pix"""
                        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                        self.optimizers.append(self.optimizer_G)
                        self.optimizers.append(self.optimizer_D)
                    else:
                        raise NotImplementedError(f"GAN mode {self.gan_mode} not implemented")
        else:
            raise ValueError(f"Got arg weighted={self.volume}, but must be True to use this model.")

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # Check if target comes with impute mask mask:
        self.num_input = len(input)
        if self.num_input == 5:
            """ Input, target + forest mask"""
            self.real_A = input['input'].to(self.device)
            self.real_B = input['target'].to(self.device)
            self.real_B_mask = input['target_vol_mask_tensor'].to(self.device)
            self.image_paths = input['input_paths']

            # TODO: check if this is needed:
            # # Create a zero mask that ensure that impute_loss is 0 independent on value of self.impute_w
            # self.real_B_gr_mask = torch.zeros(self.input_shp, self.input_shp).to(self.device)

        elif self.num_input == 6:
            """ Input, target + forest mask + GR mask"""
            self.real_A = input['input'].to(self.device)
            self.real_B = input['target'].to(self.device)
            self.real_B_mask = input['target_vol_mask_tensor'].to(self.device)
            self.real_B_gr_mask = input['target_gr_mask_tensor'].to(self.device)

            self.image_paths = input['input_paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # TODO: Check if mask also should be passed to forward with real_A
        self.fake_B = self.netG(self.real_A)  # self.net(A) i.e. the generator model

    def backward_D(self):
        """Calculate GAN loss for the discriminator, based on implementation in:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
        """
        if self.netD_type in ['n_layers', 'basic']:
            # Apply volume mask on self.real_A, self.real_B and self.fake_B:
            self.real_A_vol = self.real_A * self.real_B_mask
            self.real_B_vol = self.real_B * self.real_B_mask
            self.fake_B_vol = self.fake_B * self.real_B_mask

            # Apply GR imputed mask on self.real_A, self.real_B and self.fake_B:
            self.real_A_gr = self.real_A * self.real_B_gr_mask
            self.real_B_gr = self.real_B * self.real_B_gr_mask
            self.fake_B_gr = self.fake_B * self.real_B_gr_mask

            # Fake; stop backprop to the generator by detaching fake_B
            # we use conditional GANs; we need to feed both input and output to the discriminator
            fake_AB_vol = torch.cat((self.real_A_vol, self.fake_B_vol), 1)  # For timber volume mask
            fake_AB_gr = torch.cat((self.real_A_gr, self.fake_B_gr), 1)  # For imputed gr mask

            pred_fake_vol = self.netD(fake_AB_vol.detach())
            pred_fake_gr = self.netD(fake_AB_gr.detach())

            # D on fake with volume mask:
            self.loss_D_fake_vol = self.criterionGAN(pred_fake_vol, False)
            # D on fake with imputed GR mask:
            self.loss_D_fake_gr = self.criterionGAN(pred_fake_gr, False)

            # Real
            real_AB_vol = torch.cat((self.real_A_vol, self.real_B_vol), 1)
            real_AB_gr = torch.cat((self.real_A_gr, self.real_B_gr), 1)
            pred_real_vol = self.netD(real_AB_vol)
            pred_real_gr = self.netD(real_AB_gr)

            # D on real with volume mask:
            self.loss_D_real_vol = self.criterionGAN(pred_real_vol, False)
            # D on real with imputed GR mask:
            self.loss_D_real_gr = self.criterionGAN(pred_real_gr, False)

            # Combine volume and GR losses separately and include impute_w
            self.loss_D_fake = self.loss_D_fake_vol + self.loss_D_fake_gr * self.impute_w
            self.loss_D_real = self.loss_D_real_vol + self.loss_D_real_gr * self.impute_w

            if self.gan_mode in ['lsgan', 'vanilla']:
                # combine loss and calculate gradients
                self.loss_D_tot = (self.loss_D_fake + self.loss_D_real) * 0.5
                self.loss_D_tot.backward()
            else:
                raise NotImplementedError(f"GAN mode name {self.gan_mode} is not implemented.")

        else:
            """ For Pixel GAN"""
            # Fake;
            # stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.netD(fake_AB.detach())

            self.loss_D_fake = self.criterionGAN(pred_fake, False, self.real_B_mask, self.real_B_gr_mask, self.impute_w)

            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True, self.real_B_mask, self.real_B_gr_mask, self.impute_w)

            if self.gan_mode in ['lsgan', 'vanilla']:
                # combine loss and calculate gradients
                self.loss_D_tot = (self.loss_D_fake + self.loss_D_real) * 0.5
                self.loss_D_tot.backward()
            else:
                raise NotImplementedError(f"GAN mode name {self.gan_mode} is not implemented.")

    def backward_G(self):
        """
        Calcualte losses for the G network (the "generator")
        :return:
        """
        if self.isTrain and self.stage == 1:  # If train +pretrain mode:
            if self.l_type == 'pixel':
                self.loss_Pre_Pixel = self.criterionPretrainPixel(self.fake_B, self.real_B, self.real_B_mask,
                                                                  self.real_B_gr_mask, self.impute_w, self.device)
                self.loss_G_tot = self.loss_Pre_Pixel

            elif self.l_type == 'gan':
                if self.netD_type in ['n_layers', 'basic']:
                    # Apply volume mask and GR impute mask on self.real_A and self.fake_B:
                    self.real_A_vol = self.real_A * self.real_B_mask
                    self.fake_B_vol = self.fake_B * self.real_B_mask

                    # Apply GR imputed mask on self.real_A, self.real_B and self.fake_B:
                    self.real_A_gr = self.real_A * self.real_B_gr_mask
                    self.fake_B_gr = self.fake_B * self.real_B_gr_mask

                    # Fake; stop backprop to the generator by detaching fake_B
                    # we use conditional GANs; we need to feed both input and output to the discriminator
                    fake_AB_vol = torch.cat((self.real_A_vol, self.fake_B_vol), 1)  # For timber volume mask
                    fake_AB_gr = torch.cat((self.real_A_gr, self.fake_B_gr), 1)  # For imputed gr mask

                    pred_fake_vol = self.netD(fake_AB_vol.detach())
                    pred_fake_gr = self.netD(fake_AB_gr.detach())

                    # D on fake with volume mask:
                    self.loss_Pre_G_GAN_vol = self.criterionGAN(pred_fake_vol, False)
                    # D on fake with imputed GR mask:
                    self.loss_Pre_G_GAN_gr = self.criterionGAN(pred_fake_gr, False)

                    # Combine volume and GR loss and include impute_w
                    self.loss_Pre_G_GAN = self.loss_Pre_G_GAN_vol + self.loss_Pre_G_GAN_gr * self.impute_w

                else: # PixelDiscriminator:
                    # For gan-loss need to fake the discriminator.
                    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                    pred_fake = self.netD(fake_AB)

                    self.loss_Pre_G_GAN = self.criterionGAN(pred_fake, True, self.real_B_mask, self.real_B_gr_mask,
                                                            self.impute_w)

                # Second, G(A) = B
                self.loss_Pre_G_Pixel = self.criterionPixel(self.fake_B, self.real_B, self.real_B_mask,
                                                            self.real_B_gr_mask, self.impute_w, self.device)

                # Combine all generator losses with regularization into the G_tot loss
                self.loss_G_tot = self.alpha * self.loss_Pre_G_GAN + self.loss_Pre_G_Pixel

        elif self.isTrain and self.stage == 2:

            if self.l_type == 'prepixel_gan':
                # Pretrain G on Pixel loss, then fine-tune G on all gan losses

                """GAN loss """
                if self.netD_type in ['n_layers', 'basic']:
                    # Apply volume mask and GR impute mask on self.real_A and self.fake_B:
                    self.real_A_vol = self.real_A * self.real_B_mask
                    self.fake_B_vol = self.fake_B * self.real_B_mask

                    # Apply GR imputed mask on self.real_A, self.real_B and self.fake_B:
                    self.real_A_gr = self.real_A * self.real_B_gr_mask
                    self.fake_B_gr = self.fake_B * self.real_B_gr_mask

                    # Fake; stop backprop to the generator by detaching fake_B
                    # we use conditional GANs; we need to feed both input and output to the discriminator
                    fake_AB_vol = torch.cat((self.real_A_vol, self.fake_B_vol), 1)  # For timber volume mask
                    fake_AB_gr = torch.cat((self.real_A_gr, self.fake_B_gr), 1)  # For imputed gr mask

                    pred_fake_vol = self.netD(fake_AB_vol.detach())
                    pred_fake_gr = self.netD(fake_AB_gr.detach())

                    # D on fake with volume mask:
                    self.loss_G_GAN_vol = self.criterionGAN(pred_fake_vol, False)
                    # D on fake with imputed GR mask:
                    self.loss_G_GAN_gr = self.criterionGAN(pred_fake_gr, False)

                    # Combine volume and GR loss and include impute_w
                    self.loss_G_GAN = self.loss_G_GAN_vol + self.loss_G_GAN_gr * self.impute_w

                else:  # PixelDiscriminator:
                    # For gan-loss need to fake the discriminator.
                    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                    pred_fake = self.netD(fake_AB)

                    self.loss_G_GAN = self.criterionGAN(pred_fake, True, self.real_B_mask, self.real_B_gr_mask,
                                                        self.impute_w)

                # Second, G(A) = B
                self.loss_G_Pixel = self.criterionPixel(self.fake_B, self.real_B, self.real_B_mask, self.real_B_gr_mask,
                                                        self.impute_w, self.device)

                # Combine all generator losses with regularization into the G_tot loss
                self.loss_G_tot = self.alpha * self.loss_G_GAN + self.loss_G_Pixel

            elif self.l_type == 'prepixel_spec':

                """Pixel and Spectral loss """
                self.loss_G_Pixel = self.criterionPixel(self.fake_B, self.real_B, self.real_B_mask, self.real_B_gr_mask,
                                                        self.impute_w, self.device)
                self.loss_G_Spec = self.criterionSpectral(self.fake_B, self.real_B, self.real_B_mask,
                                                          self.real_B_gr_mask, self.impute_w, self.device)

                # Combine all generator losses losses into the AE generator loss:
                self.loss_G_tot = self.loss_G_Pixel + self.gamma * self.loss_G_Spec

            elif self.l_type == 'pregan_spec':

                """GAN losses """
                if self.netD_type in ['n_layers', 'basic']:
                    # Apply volume mask and GR impute mask on self.real_A and self.fake_B:
                    self.real_A_vol = self.real_A * self.real_B_mask
                    self.fake_B_vol = self.fake_B * self.real_B_mask

                    # Apply GR imputed mask on self.real_A, self.real_B and self.fake_B:
                    self.real_A_gr = self.real_A * self.real_B_gr_mask
                    self.fake_B_gr = self.fake_B * self.real_B_gr_mask

                    # Fake; stop backprop to the generator by detaching fake_B
                    # we use conditional GANs; we need to feed both input and output to the discriminator
                    fake_AB_vol = torch.cat((self.real_A_vol, self.fake_B_vol), 1)  # For timber volume mask
                    fake_AB_gr = torch.cat((self.real_A_gr, self.fake_B_gr), 1)  # For imputed gr mask

                    pred_fake_vol = self.netD(fake_AB_vol.detach())
                    pred_fake_gr = self.netD(fake_AB_gr.detach())

                    # D on fake with volume mask:
                    self.loss_G_GAN_vol = self.criterionGAN(pred_fake_vol, False)
                    # D on fake with imputed GR mask:
                    self.loss_G_GAN_gr = self.criterionGAN(pred_fake_gr, False)

                    # Combine volume and GR loss and include impute_w
                    self.loss_G_GAN = self.loss_G_GAN_vol + self.loss_G_GAN_gr * self.impute_w

                else:  # PixelDiscriminator:
                    # For gan-loss need to fake the discriminator.
                    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                    pred_fake = self.netD(fake_AB)

                    self.loss_G_GAN = self.criterionGAN(pred_fake, True, self.real_B_mask, self.real_B_gr_mask,
                                                        self.impute_w)

                # Second, G(A) = B
                self.loss_G_Pixel = self.criterionPixel(self.fake_B, self.real_B, self.real_B_mask, self.real_B_gr_mask,
                                                        self.impute_w, self.device)

                """Spectral loss """
                self.loss_G_Spec = self.criterionSpectral(self.fake_B, self.real_B, self.real_B_mask,
                                                          self.real_B_gr_mask, self.impute_w, self.device)

                # Combine all generator losses losses into the generator loss:
                self.loss_G_tot = (self.alpha * self.loss_G_GAN + self.loss_G_Pixel) + self.gamma * self.loss_G_Spec

            elif self.l_type == 'prepixel_gan_spec':
                # Pretrain G on Pixel loss, then fine-tune G on GAN + spectral loss

                """Spectral loss """
                self.loss_G_Spec = self.criterionSpectral(self.fake_B, self.real_B, self.real_B_mask,
                                                          self.real_B_gr_mask, self.impute_w, self.device)

                """GAN losses """
                if self.netD_type in ['n_layers', 'basic']:
                    # Apply volume mask on self.real_A, self.real_B and self.fake_B:
                    self.real_A_vol = self.real_A * self.real_B_mask
                    self.fake_B_vol = self.fake_B * self.real_B_mask

                    # Apply GR imputed mask on self.real_A, self.real_B and self.fake_B:
                    self.real_A_gr = self.real_A * self.real_B_gr_mask
                    self.fake_B_gr = self.fake_B * self.real_B_gr_mask

                    # Fake; stop backprop to the generator by detaching fake_B
                    # we use conditional GANs; we need to feed both input and output to the discriminator
                    fake_AB_vol = torch.cat((self.real_A_vol, self.fake_B_vol), 1)  # For timber volume mask
                    fake_AB_gr = torch.cat((self.real_A_gr, self.fake_B_gr), 1)  # For imputed gr mask

                    pred_fake_vol = self.netD(fake_AB_vol.detach())
                    pred_fake_gr = self.netD(fake_AB_gr.detach())

                    # D on fake with volume mask:
                    self.loss_G_GAN_vol = self.criterionGAN(pred_fake_vol, False)
                    # D on fake with imputed GR mask:
                    self.loss_G_GAN_gr = self.criterionGAN(pred_fake_gr, False)

                    # Combine volume and GR loss and include impute_w
                    self.loss_G_GAN = self.loss_G_GAN_vol + self.loss_G_GAN_gr * self.impute_w

                else:  # PixelDiscriminator:
                    # For gan-loss need to fake the discriminator.
                    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                    pred_fake = self.netD(fake_AB)

                    self.loss_G_GAN = self.criterionGAN(pred_fake, True, self.real_B_mask, self.real_B_gr_mask,
                                                        self.impute_w)

                # Second, G(A) = B
                self.loss_G_Pixel = self.criterionPixel(self.fake_B, self.real_B, self.real_B_mask, self.real_B_gr_mask,
                                                        self.impute_w, self.device)

                # Combine all generator losses losses into the generator loss:
                self.loss_G_tot = (self.alpha * self.loss_G_GAN + self.loss_G_Pixel) + self.gamma * self.loss_G_Spec

        # Calculate gradients
        self.loss_G_tot.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # If have a gan-loss, need to update the discriminator.
        if self.l_type == 'gan' or self.l_type == 'prepixel_gan' or self.l_type == 'prepixel_gan_spec' or \
                self.l_type == 'pregan_spec':
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights

            # Update the generator (G):
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate gradients for G
            self.optimizer_G.step()  # update G's weights

        else:
            # Update the generator:
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate gradients for G
            self.optimizer_G.step()  # update G's weights


class Test_VolumeForestModel(BasicModelModules):
    """ This TesteModel is inspired by the Testmodel() fromn pix2pix.
        It can be used to generate I2I results for only one direction.

        This model will check if opt.dataset_mode is either 'remotesensing' or 'agb', if not it will raise error as
        it doesn't accept both input and target data.

        See the test instruction for more details.
        """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time.
        """
        assert not is_train, 'TestModel cannot be used during training time'

        return parser

    def __init__(self, opt):
        """Initialize the ForestModel class for testing.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert (not opt.isTrain)
        BasicModelModules.__init__(self, opt)
        self.volume = opt.volume
        self.test_target_path = opt.test_target_path

        # To ensure that we don't call weighted model:
        if self.volume:
            # Specify the training losses you want to print out. The training/test scripts  will call
            # <BasicModelModules.get_current_losses>
            self.l_names = []

            # Specify the images you want to save/display. The training/test scripts  will call
            # <BasicModelModules.get_current_visuals>
            if self.test_target_path is None:
                self.visual_names = ['real_A', 'fake_B']
            else:
                self.visual_names = ['real_A', 'fake_B', 'real_B']

            # specify the models you want to save to the disk. The training/test scripts will call
            # <BasicModelModules.save_networks> and <BasicModelModules.load_networks>
            self.model_names = ['G']  # only generator is needed.

            # Define_G from Forest model:
            self.netG = define_G(opt.target_nc, opt.input_nc, opt.input_shp, opt.netG, opt.pretrainweights,
                                 opt.encoder_name, opt.enc_depth, opt.norm)

            if self.test_target_path is None:
                # Ensure that opt.dataset_mode is either 'remotesensing' or 'agb if there are no target path:
                if opt.dataset_mode not in ['remotesensing', 'agb']:
                    raise ValueError(
                        f"Got dataset={opt.dataset_mode} but test model only accepts dataset=['remotesensing', "
                        f"'agb'] if no target path is given")
            else:
                if opt.dataset_mode != 'rs_agb':
                    raise ValueError(
                        f"Got dataset={opt.dataset_mode} but test model with target path only accepts dataset='rs_agb'")

            # assigns the model to self.netG so that it can be loaded
            # please see <BaseModel.load_networks> # TODO update this one
            setattr(self, 'netG', self.netG)  # store netG in self.
        else:
            raise ValueError(f"Got arg weighted={self.volume}, but must be False to use this model.")

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        if self.test_target_path is None:
            self.real_A = input['input'].to(self.device)
            self.image_paths = input['input_paths']
        else:
            self.real_A = input['input'].to(self.device)
            self.real_B = input['target'].to(self.device)
            self.real_B_mask = input['target_vol_mask_tensor'].to(self.device)
            self.image_paths = input['input_paths']

    def forward(self):
        """Run forward pass."""
        self.fake_B = self.netG(self.real_A)  # G(real_A)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass