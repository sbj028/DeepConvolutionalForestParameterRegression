import numpy as np
import os
import ntpath
import time

import matplotlib.pyplot as plt
from PIL import Image
from tifffile import imread, imwrite
from . import util

"""
This module contains a combination of modules retreived visualizer.py from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/visualizer.py
and modifications to it.
"""

# Check if wandb is available:
try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


class write_output:
    """This class allows us to write images to a outputdir.

     It consists of functions such as  <save> (save to the disk).

    """

    def __init__(self, output_dir):
        """Initialize the write_output classes

        Parameters:
            output_dir (str) -- a directory to write images to.
        """
        self.output_dir = output_dir
        self.img_dir = os.path.join(self.output_dir, 'images')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir


def save_images(output, visuals, image_path, extension, opt, aspect_ratio=1.0, use_wandb=False):
    """Save images to the disk.

    :param output (the write_output class) -- the write_output class that stores these images
    :param visuals (dict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
    :param image_path (str)         -- the string is used to create image paths
    :param extension (str)         -- if want to write .png, .tiff, .tif
    :param opt                      -- parameters from Baseoptions, train-options or test.options
    :param aspect_ratio (float)     -- the aspect ratio of saved images
    :param use_wandb (bool)         -- if True log data to wandb

    This function will save images stored in 'visuals' to the output_dir specified by 'output'.
    """
    image_dir = output.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    ims_dict = {}
    for label, im_data in visuals.items():
        if extension == '.png':
            im = util.tensor2im(im_data, opt)
            image_name = f"{name}_{label}.png"
            save_path = os.path.join(image_dir, image_name)
        elif extension == '.tiff' or extension == '.tif':
            im = util.tensor2im(im_data, opt, imtype=np.float32)
            image_name = f"{name}_{label}.tiff"
            save_path = os.path.join(image_dir, image_name)
        else:
            print('Output file format not implemented')
            break

        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imread(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imread(im, (int(h / aspect_ratio), w), interp='bicubic')
        # Write image to disc:
        imwrite(save_path, im)

        if use_wandb:
            ims_dict[label] = wandb.Image(im)
        if use_wandb:
            wandb.log(ims_dict)


class Visualizer:
    """
    This class includes several functions that can display/save images and print/save logging information.
    """
    def __init__(self, opt):
        """
        Initialize the Visualizer class
        :param opt   -- parameters from Baseoptions, train-options or test.options
        """

        self.opt = opt  # cache the option
        self.relu = opt.relu
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.current_epoch = 0

        # To store images as either '.png' or '.tiff':
        self.extension = opt.extension  # To store images as either '.png' or '.tiff'

        # Initialize wandb if not already done:
        if self.use_wandb:
            if opt.sweep == 'False' or opt.sweep == 'false' or opt.sweep == False:  # if don't use wandb sweep:
                self.wandb_run = wandb.init(project=opt.wandb_project, name=opt.wandb_name,
                                            config=opt) if not wandb.run else wandb.run
                self.wandb_run._label(repo=opt.wandb_project)
            elif opt.sweep == 'True' or opt.sweep == 'true' or opt.sweep == True:  # if use wandb sweep:
                self.wandb_run = wandb.init(config=opt) if not wandb.run else wandb.run

        # create a logging file to store training losses in the corresponding checkpoint dirs:
        if opt.stage == 1:
            self.log_name = os.path.join((opt.checkpoints_dir + '/pretrain'), opt.name, 'loss_log_pretrain.txt')
        elif opt.stage == 2:
            if opt.sweep == 'True' or opt.sweep == 'true' or opt.sweep == True:  # if use wandb sweep:
                self.log_name = os.path.join((opt.checkpoints_dir + '/finetune_sweeps'),
                                             opt.name_finetune, 'loss_log_finetune_sweep.txt')
            else:
                self.log_name = os.path.join((opt.checkpoints_dir + '/finetune'),
                                             opt.name_finetune, 'loss_log_finetune.txt')
        else:
            raise ValueError(f"Got stage={opt.stage}, but must be eiter 1 or 2")

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch):
        """Display current results on wandb;
        save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
        """
        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, 'epoch')
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                if self.extension == '.tiff' or self.extension == '.tif':
                    """
                    Normalize images (real_A, fake_B and real_B to [0, 255] and set imtype to np.uint8 
                    prior visualisation in wandb:
                    """
                    im = util.tensor2im(image, relu=self.relu, imtype=np.uint8, normalize=True, rounded=True,
                                        n_range=[0, 255])
                    image_pil = Image.fromarray(im, mode='RGB')
                    wandb_image = wandb.Image(image_pil)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            self.wandb_run.log(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                self.wandb_run.log({"Result": result_table})

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = f'(epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, data: {t_data:.3f}) '
        if self.use_wandb:
            for k, v in losses.items():
                message += f'{k}: {v:.3f} '
                self.wandb_run.log({'epoch': epoch, k: v})
        else:
            for k, v in losses.items():
                message += f'{k}: {v:.3f} '

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def wandb_sweep(self, metrics):
        self.wandb_run.log(metrics)

    # losses: same format as |losses| of plot_current_losses
    def plot_combined_train_val_losses(self, epoch, loss_train, loss_val):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            loss_train (float) -- current training loss
            loss_val (float) -- current validation loss
        """
        if self.use_wandb:

            plt.figure()
            plt.plot(epoch, loss_train, )
            plt.plot(epoch, loss_val)
            plt.legend(['train', 'val'])

            wandb.log({'plot': plt})

    # losses: same format as |losses| of plot_current_losses
    def plot_separate_train_val_losses(self, epoch, losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        if self.use_wandb:
            self.wandb_run.log({'G_loss_train': losses['G_loss_train'],
                                'G_loss_val': losses['G_loss_val'], 'epoch': epoch})
