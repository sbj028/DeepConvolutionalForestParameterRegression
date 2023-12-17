"""General-purpose test script for image-to-image translation.

Once you have trained your model with either train_agb.py or train_volume.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images.

The results will be saved at ./results/.
Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

See options/base_options.py and options/test_options.py for more test options.

"""
import os
from options.test_options import TestOptions
from load_data.datasetloader import create_dataset
from modules.forest_models import Test_AGBForestModel, Test_VolumeForestModel
from util.visualizer import save_images, write_output

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.

    # Check length of norm_mean and norm_std list so that it corresponds with input_nc:
    len_norm_mean = len(opt.norm_mean)
    len_norm_std = len(opt.norm_std)

    if len_norm_mean != opt.input_nc or len_norm_std != opt.input_nc:
        raise ValueError(f"Number of mean and std values for normalization != number of input_nc")

    # Create model given options:
    if not opt.volume:
        model = Test_AGBForestModel(opt) # Load regular model
        print(f"Load AGBForest test model")
    if opt.volume:
        model = Test_VolumeForestModel(opt) # Load regular model
        print(f"Load VOlumeForest test model")
    # Load and print network, create scheduler:
    model.setup(opt)

    # Create output dir and write images to it:
    if opt.stage == 1:
        """Test a pretrain/baseline model: """
        output_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')  # define the output dir
    elif opt.stage == 2:
        """Test a finetuned model: """
        output_dir = os.path.join(opt.results_dir,  opt.name_finetune, f'{opt.phase}_{opt.epoch}')  # define the output dir

    output = write_output(output_dir)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print(f"The number of test images is {dataset_size}")

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results

        img_path = model.get_image_paths()     # get image paths

        if i % 5 == 0:  # save images to output dir
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(output, visuals, img_path, opt.extension, opt, opt.aspect_ratio)