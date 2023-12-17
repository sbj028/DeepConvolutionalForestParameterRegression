import argparse
import os
from util import util
import torch


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--name_finetune', type=str, default='finetune_name',
                            help='Only used if do fine-tuning and want to store experiment with new name.')
        parser.add_argument('--use_wandb', default=True, help='use wandb')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--image_dir', type=str, default='./images', help='Images are saved here')
        # model parameters
        parser.add_argument('--input_nc', type=int, default=3,
                            help='# of input image channels: 3 for RGB and 1 for grayscale. Assumption; train and '
                                 'validation datasets has the same number of channels.')
        parser.add_argument('--target_nc', type=int, default=1,
                            help='# of target (output) image channels: 3 for RGB and 1 for grayscale. Assumption; train'
                                 'and validation datasets has the same number of channels.')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        # parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic',
                            help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a '
                                 '70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='customUnet', help='For the moment, "customUnet" is the only '
                                                                           'option.')
        parser.add_argument('--encoder_name', type=str, default='resnet34',
                            help='Check https://segmentation-modelspytorch.readthedocs.io/en/latest/docs/api.html#linknet '
                                 'for possibilities')
        parser.add_argument('--enc_depth', type=int, default=4, help='3,4 or 5 are probably good choices.')
        parser.add_argument('--n_blocks', type=int, default=6, help='Number of ResNet blocks in bottleneck.')

        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | layer | none]')
        parser.add_argument('--volume', default=False,
                            help='If true, will use dataloader which load input, target and mask + use volume losses.'
                                 'If false, use regular dataloader and losses.')
        parser.add_argument('--pretrainweights', default=None,
                            help='Can be: | True | None |. If True, will load weights from pretrained ResNet'
                                 '(ImageNet weights) else initialize weights randomly. Only used in training, during'
                                 'inference stored weights overwrite these.')
        # parser.add_argument('--init_type', type=str, default='normal',
        # help='network initialization [normal | xavier | kaiming | orthogonal]')
        # parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # TODO: If enable no_dropout arg has to disable it for test and validation generator.
        # parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='rs_agb',
                            help='chooses if single input (remotesensing) or target (agb) data, or combined '
                                 'input/targets should be loaded. [remotesensing | agb | rs_agb]')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=5, help='input batch size')
        # parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more '
                                 'than max_dataset_size, only a subset is loaded.')
        # Normalization/Preprocess args:
        parser.add_argument('--preprocess', type=str, default='none',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | '
                                 'scale_width_and_crop | none]')
        parser.add_argument('--convert', default=True,
                            help="If True: Do ToTensor and Normalize in get_transform() else, don't do these steps in "
                                "get_transform(). If False require that input images are converted to tensors manually")
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--crop_size', type=int, default=64, help='then crop to this size')
        parser.add_argument('--norm_mean', type=float, nargs='+', default=[0.485, 0.456, 0.406],
                            help='Channel-wise mean to normalize input before model. Default is ImageNet  mean, can '
                                 'only be used with input of three channels.')
        parser.add_argument('--norm_std', type=float, nargs='+', default=[0.229, 0.224, 0.225],
                            help='Channel-wise std to normalize input before model. Default is ImageNet std, can '
                                 'only be used with input of three channels.')
        # # additional parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0',
                            help='which iteration to load? if load_iter > 0, the code will load models by '
                                 'iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--stage', type=int, default=1, help='Set to 1 if pretrain generator with pixel loss, set '
                                                                 'to 2 if are in stage 2 when pixel+gan, pixel+spec or'
                                                                 'pixel+gan+spec loss are used to train.')
        # paramters to store images as .tiff
        parser.add_argument('--extension', default='.tiff', type=str, help='can be .tiff, tif')
        parser.add_argument('--mode_out', default='F', type=str,
                            help='For PIL.Image.fromarray() can set the mode, i.e. dtype of output image. '
                                 'See https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes'
                                 'for possibilities.')
        parser.add_argument('--input_shp', default='64', type=int, help='Input shape of image, i.e. width/height. '
                                                                        'Assume that the image is a square, where '
                                                                        'width = height.')
        parser.add_argument('--relu', default=True, help='If True=reLu output, i.e. no scaling of input/output. '
                                                         'If False= other output activation function, i.e. scaling'
                                                         'of output/input')
        # Loss regularization paramters
        parser.add_argument('--alpha', type=float, default='0.01',
                            help='Regularization coefficient in range [0, 0.01, 0.1, 1]  Used to balanse loss terms in '
                                 'pix2pix G loss.')
        parser.add_argument('--gamma', type=float, default='1e-8', help='Regularization coefficient for FFT-loss.')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        """ Create checkpoint dirs for pretraining, finetune or finetune + sweeps"""
        # save to the disk
        if opt.stage == 1:
            expr_dir = os.path.join((opt.checkpoints_dir + '/pretrain'), opt.name)
            util.mkdirs(expr_dir)
        elif opt.stage == 2:
            if opt.sweep == 'True' or opt.sweep == 'true' or opt.sweep == True:  # if use wandb sweep:
                expr_dir_finetune_sweep = os.path.join((opt.checkpoints_dir + '/finetune_sweeps'), opt.name_finetune)
                util.mkdirs(expr_dir_finetune_sweep)
            else:
                expr_dir_finetune = os.path.join((opt.checkpoints_dir + '/finetune'), opt.name_finetune)
                util.mkdirs(expr_dir_finetune)

        if opt.stage == 1:
            file_name = os.path.join(expr_dir, 'opt_pretrain.txt')
        elif opt.stage == 2:
            if opt.sweep == 'True' or opt.sweep == 'true' or opt.sweep == True:  # if use wandb sweep:
                file_name = os.path.join(expr_dir_finetune_sweep, f'opt_finetune.txt')
            else:
                file_name = os.path.join(expr_dir_finetune, 'opt_finetune.txt')
        else:
            raise ValueError(f"Got stage={opt.stage}, but must be eiter 1 or 2")

        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
