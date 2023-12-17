"""
Train procedure for SV model:

"""
import time
import torch
from evaluate import evaluate
from util.visualizer import Visualizer
from modules.forest_models import VolumeForestModel
from options.train_options import TrainOptions  # If cluster
from load_data.datasetloader import create_dataset


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    # Parse training options:
    opt = TrainOptions().parse()

    # Check length of norm_mean and norm_std list so that it corresponds with input_nc:
    len_norm_mean = len(opt.norm_mean)
    len_norm_std = len(opt.norm_std)

    if len_norm_mean != opt.input_nc or len_norm_std != opt.input_nc:
        raise ValueError(f"Number of mean and std values for normalization != number of input_nc")

    # Create model given options:
    model = VolumeForestModel(opt)

    # Load and print network, create scheduler:
    model.setup(opt)

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0

    """ Training procedure: """
    # Read dataset:
    if opt.phase == 'train':

        # Read dataset:
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)
        print(f"The number of training images = {dataset_size}")

        # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()  # timpt.epoch_count,
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()  # reset the visualizer

            for idx, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                """ Display images in wandb"""
                if total_iters % opt.display_freq == 0:
                    save_result = total_iters % opt.update_wandb_freq == 0

                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                """ Print training losses to console and Wandb. Save logging information to the disk"""
                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    """Printing current losses in terminal, log-file and wandb: """
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                # cache our latest model every <save_latest_freq> iterations
                if total_iters % opt.save_latest_freq == 0:
                    print(f"Saving the latest model (epoch {epoch} total_iters {total_iters})")
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix, opt)

                iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print(f"Saving the model at the end of epoch {epoch}, iters {total_iters}")

                model.save_networks('latest', opt)
                model.save_networks(epoch, opt)

            print(f"End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: "
                 f"{time.time() - epoch_start_time} sec")
            model.update_learning_rate(opt)  # update learning rates in the beginning of every epoch.

    elif opt.phase == 'trainval':

        """ Training with validation procedure: """
        # Read dataset:
        dataset_train, dataset_val = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_train_size = len(dataset_train)
        dataset_val_size = len(dataset_val)
        print(f"Number of training images = {dataset_train_size} and validation images = {dataset_val_size}")

        ##################
        ### TRAIN LOOP ###
        ##################
        # set the model to train mode

        model.train()

        loss_train = []
        loss_val = []
        epoch_list = []

        losses = {}

        # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()  # timpt.epoch_count,
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()  # reset the visualizer

            for idx_train, data_train in enumerate(dataset_train):  # inner loop within one training epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                model.set_input(data_train)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                if not opt.sweep:  # If not do sweep, visualize patches:
                    """ Display images in wandb"""
                    if total_iters % opt.display_freq == 0:
                        save_result = total_iters % opt.update_wandb_freq == 0
                        # model.compute_visuals()
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                """ Print training losses to console and Wandb. Save logging information to the disk"""
                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    train_losses = model.get_current_losses()

                    t_comp = (time.time() - iter_start_time) / opt.batch_size

                    """Printing current losses in terminal, log-file and wandb: """
                    visualizer.print_current_losses(epoch, epoch_iter, train_losses, t_comp, t_data)

                # cache our latest model every <save_latest_freq> iterations:
                if total_iters % opt.save_latest_freq == 0:
                    print(f"Saving the latest model (epoch {epoch} total_iters {total_iters})")
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'

                    model.save_networks(save_suffix, opt)

                iter_data_time = time.time()

            ########################################
            ### VALIDATION LOOP after each epoch ###
            ########################################
            val_losses = evaluate(model, dataset_val)

            loss_train.append(train_losses['G_tot'])
            loss_val.append(val_losses['G_tot'])
            epoch_list.append(epoch)

            # Append train and val loss to loss_dict
            losses['G_loss_train'] = train_losses['G_tot']
            losses['G_loss_val'] = val_losses['G_tot']

            visualizer.plot_separate_train_val_losses(epoch, losses)

            metrics = {'G_loss_train': train_losses['G_tot'],
                       'G_loss_val': val_losses['G_tot']}

            visualizer.wandb_sweep(metrics)

            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print(f"Saving the model at the end of epoch {epoch}, iters {total_iters}")

                model.save_networks('latest', opt)
                model.save_networks(epoch, opt)

            print(f"End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: "
                  f"{time.time() - epoch_start_time} sec")
            model.update_learning_rate(opt)  # update learning rates in the beginning of every epoch.

        visualizer.plot_combined_train_val_losses(epoch_list, loss_train, loss_val)
