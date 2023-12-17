from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=400,
                            help='frequency of showing training results on screen')
        parser.add_argument('--update_wandb_freq', type=int, default=1000,
                            help='frequency of saving training results to wandb')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--l_type', type=str, default='pixel',
                            help='Type of loss(es) used, can be: | pixel | gan | prepixel_gan | prepixel_spec '
                                 '|prepixel_gan_spec |pregan_spec')
        parser.add_argument('--l1_l2', type=str, default='l2', help='Type of pixel loss used, can be | l1| l2 |.')
        parser.add_argument('--impute_w', type=int, default='100',
                            help='Weight for imputed data loss, for Tanzania test: [50, 100, 200, 300]')
        # Train data paths:
        parser.add_argument('--train_input_path', required=True,
                            help='Path to training images, input path. I.e: python/datasets/input/path/fold0/A/train')
        parser.add_argument('--train_target_path', required=True,
                            help='Path to training images, target path. I.e: python/datasets/target/path/fold0/B/train')
        # Validation data paths:
        parser.add_argument('--val_input_path',
                            help='Path to validation images, input path. I.e: python/datasets/input/path/fold0/A/val')
        parser.add_argument('--val_target_path',
                            help='Path to validation images, target path. I.e: python/datasets/target/path/fold0/B/val')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--continue_train_finetune', action='store_true',
                            help='continue finetune training: load the latest finetuned model and continue finetuning')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, '
                                 '<epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train',
                            help='train, trainval, test. Trainval is used to perfrom validation during training')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan]. Vanilla GAN loss is the cross-entropy '
                                 'objective used in the original GAN paper.')
        # wandb:
        parser.add_argument('--sweep', default=False,
                            help='False= use wandb_project and wandb_l_type args, True= Do not initialize wandb with '
                                 'name and project')
        parser.add_argument('--wandb_project', type=str, default='AE_agb_tests',
                            help='Name of project, e.g. pretrain_pixel if run pretrain models.')
        parser.add_argument('--wandb_name', type=str, default='experiment_name', help='Experiment name used in wandb, '
                                                                                      'use same as experiment name.')
        # Parameters for frequency awareness and spectral loss:
        parser.add_argument('--spec_loss_name', type=str, default='fft',
                            help='What spectral loss to use, default fft-loss.')
        self.isTrain = True
        return parser
