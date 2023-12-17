import os
import torch
from torch.nn import init
from torch.optim import lr_scheduler


class BasicModelModules:
    """
    Class consisting of several basic module used by the models.
    """

    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.stage = opt.stage
        # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        if opt.preprocess != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.l_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = None  # used for learning rate policy 'plateau'

        # Save all the checkpoints to save_dir
        if self.stage == 1:
            self.save_dir_checkp = os.path.join((opt.checkpoints_dir + '/pretrain'), opt.name)
        elif self.stage == 2:
            if opt.sweep == 'True' or opt.sweep == 'true' or opt.sweep == True:  # if use wandb sweep:
                self.save_dir_checkp = os.path.join((opt.checkpoints_dir + '/finetune_sweeps'), opt.name_finetune)
            elif opt.sweep == 'False' or opt.sweep == 'false' or opt.sweep == False:
                self.save_dir_checkp = os.path.join((opt.checkpoints_dir + '/finetune'), opt.name_finetune)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(BasicModelModules, self).__init__()

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = {}
        for name in self.l_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def update_learning_rate(self, opt):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print(f"learning rate {old_lr:.7} -> {lr:.7}")

    def setup(self, opt):
        """
        Inspired by similar function in pix2pix, see models/base_model and setup()
        Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            if opt.continue_train:
                """
                To continue train a model, i.e --continue_train is given (=True)
                """
                load_suffix = f"iter_{opt.load_iter}" if opt.load_iter > 0 else opt.epoch
                self.load_networks_continue_training_or_test(load_suffix, opt)
            elif opt.continue_train_finetune:
                """
                To continue train a model fine-tuned model, --continue_train is given (=True)
                """
                load_suffix = f"iter_{opt.load_iter}"  if opt.load_iter > 0 else opt.epoch
                self.load_networks_finetune(load_suffix, opt)
        if not self.isTrain:
            if opt.phase == 'test':
                """
                To load a model for testing. I.e. self.isTrain = False 
                """
                load_suffix = f"iter_{opt.load_iter}"  if opt.load_iter > 0 else opt.epoch
                self.load_networks_continue_training_or_test(load_suffix, opt)
        self.print_networks(opt.verbose)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass
    
    def train(self):
        """ Turn model to train mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        """
        with torch.no_grad():
            self.forward()

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def load_networks_continue_training_or_test(self, epoch, opt):
        """
        Inspired by similar function in pix2pix, see models/base_model
        Load all the networks from the disk.
        
        This module is called if we either:
        - Want to continue training a pretrained model or a fine-tuned model. I.e. this module is not called when
         going from pre-training to fine-tuning.
        - Want to run the test-phase and load a pretrained or a finetuned model
        
        Parameters:
            epoch (int) -- current epoch; used in the file name f"'"{epoch}_net_{name}.pth"'"
            opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
        """
        for name in self.model_names:
            if isinstance(name, str):

                if self.stage == 1:
                    load_filename = f"{epoch}_pretrain_net_{name}.pth"
                    load_path = os.path.join(self.save_dir_checkp, load_filename)
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    if opt.phase == 'test':
                        print(f"loading the model from {load_path} to test")
                    else:
                        print(f"loading the model from {load_path} to continue pretraining")
                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    net.load_state_dict(state_dict)
                    
                elif self.stage == 2:
                    # Don't load finetune sweep checkpoint for continue training, just regular finetune model:
                    if opt.sweep == 'False' or opt.sweep == 'false' or opt.sweep == False:
                        load_filename = f"{epoch}_finetune_net_{name}.pth"
                        load_path = os.path.join(self.save_dir_checkp, load_filename)
                        net = getattr(self, 'net' + name)
                        if isinstance(net, torch.nn.DataParallel):
                            net = net.module
                        if opt.phase == 'test':
                            print(f"loading the model from {load_path} to test")
                        else:
                            print(f"loading the model from {load_path} continue fine-tuning")
                        # if you are using PyTorch newer than 0.4 (e.g., built from
                        # GitHub source), you can remove str() on self.device
                        state_dict = torch.load(load_path, map_location=str(self.device))
                        if hasattr(state_dict, '_metadata'):
                            del state_dict._metadata

                        # patch InstanceNorm checkpoints prior to 0.4
                        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                        net.load_state_dict(state_dict)
                else: 
                    raise ValueError(f"Got stage={self.stage}, but must be eiter 1 or 2, and for stage 2 sweep models "
                                     f"are not loaded to continued training")

    def load_networks_finetune(self, epoch, opt):
        """
        Inspired by similar function in pix2pix, see models/base_model
        Load all the networks from the disk.

        This module is only called if we want to fine-tune a pre-trained model. Load model from checkpoint with
        pre-trained models.
         
        Parameters:
            epoch (int) -- current epoch; used in the file name f"{epoch}_net_{net}.pth"
            opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
        """
        for name in self.model_names:
            if isinstance(name, str):
                if self.stage == 2:
                    # Load pre-trained model to do fine-tuning on:
                    load_filename = f"{epoch}_pretrain_net_{name}.pth"
                    # Make sure to load checkpoints from pretrain dir:
                    pretrain_checkpoint_dir = os.path.join((opt.checkpoints_dir + '/pretrain'), opt.name)
                    load_path = os.path.join(pretrain_checkpoint_dir, load_filename)
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print(f"loading the model from {load_path} to fine-tune a pre-trained model.")

                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device

                    """ 
                    Since a model pre-trained on pixel loss doesn't have a pre-trained D model, trying to load it 
                    during fine-tuning with pixel+GAN loss will cause errors like "FileNotFoundError: 
                    [Errno 2] No such file or directory: '/pth/pretrain_pixel/latest_net_D.pth'.  
                    Thus before calling: state_dict = torch.load(load_path, map_location=str(self.device)), need to 
                    check if the path exist. If not, have to initialize the model.
                    """
                    if os.path.isfile(load_path):
                        state_dict = torch.load(load_path, map_location=str(self.device))

                        if hasattr(state_dict, '_metadata'):
                            del state_dict._metadata

                        # patch InstanceNorm checkpoints prior to 0.4
                        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                        net.load_state_dict(state_dict)

                    elif not os.path.isfile(load_path):
                        print(f"Pretrained network for {name} doesn't exists, initializing a new network.")
                        self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
                else:
                    raise ValueError(f"Got stage={self.stage}, but must be 2 to call this  module.")

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """
        Copy from Pix2Pix code
        Fix InstanceNorm checkpoints incompatibility (prior to 0.4)
        """
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def save_networks(self, epoch, opt):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name f"{epoch}_net_{net}.pth"
            opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
        """
        for name in self.model_names:
            if isinstance(name, str):
                """Save pre-training model"""
                if self.stage == 1:
                    save_filename = f"{epoch}_pretrain_net_{name}.pth"
                    save_path = os.path.join(self.save_dir_checkp, save_filename)
                    net = getattr(self, 'net' + name)
                elif self.stage == 2:
                    if opt.sweep == 'False' or opt.sweep == 'false' or opt.sweep == False:  # if don't use wandb sweep:
                        """Save fine-tuning model"""
                        save_filename = f"{epoch}_finetune_net_{name}.pth"
                        save_path = os.path.join(self.save_dir_checkp, save_filename)
                        net = getattr(self, 'net' + name)

                    elif opt.sweep == 'True' or opt.sweep == 'true' or opt.sweep == True:  # if use wandb sweep:
                        """Save fine-tuning model"""
                        save_filename = f"{epoch}_finetune_net_{name}.pth"
                        save_path = os.path.join(self.save_dir_checkp, save_filename)
                        net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def print_networks(self, verbose):
        """
        Inspired by similar function in pix2pix, see models/base_model

        Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        print(f"model_names in print_networks is: {self.model_names}")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(f"[Network {name}] Total number of parameters : {(num_params/1e6):.3f} M")
        print("-----------------------------------------------")

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with wandb"""
        visual_ret = {}
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """
    Inspired by similar function in pix2pix, see models/networks and get_scheduler()

    Return a learning rate scheduler

    Parameters:
        optimizer (str)         -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError(f"learning rate policy [{opt.lr_policy}] is not implemented")
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"initialization method {init_type} is not implemented")
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)  # apply the initialization function <init_func>
