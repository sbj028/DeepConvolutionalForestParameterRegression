"""
Inspired by  https://towardsdatascience.com/custom-dataset-in-pytorch-part-1-images-2df3152895:

"""
from python.load_data.transforms import unnormalize
import matplotlib.pyplot as plt
import numpy as np
import copy
#######################################################
#                  Visualize Dataset
#         Images are plotted after augmentation
#######################################################


def visualize_patch(opt, dataset, samples=10, cols=5, random_img=False):
    dataset = copy.deepcopy(dataset)
    rows = samples // cols

    # Draw one sample to check if dataset contain one set of images or a tuple
    test_tensor = dataset[0]

    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    if not isinstance(test_tensor, tuple):
        """ If dataset is a single dataset i.e. input or target: """
        for i in range(0, samples):
            if random_img:
                # To extract 10 random images from the dataset:
                idx = np.random.randint(1, len(opt.input_path))
            image = dataset[idx]

            #Redo normalization of the data:
            invTrans = unnormalize()
            inv_image = invTrans(image)
            input_arr = inv_image.cpu().detach().numpy()

            # Plotting:
            ax.ravel()[i].imshow(np.transpose(inv_image, (1,2,0)), vmin=np.min(input_arr), vmax=np.max(input_arr))
            ax.ravel()[i].set_axis_off()
            ax.ravel()[i].set_title(f'Training patch {idx}')
        plt.tight_layout(pad=1)
        plt.show()

        return input_arr

    elif isinstance(test_tensor, tuple):
        """
        If dataset is a tuple, i.e. both input and target plot input samples on first row and corresponding target 
        samples on second row. Plot only half of the samples to fit both inputs and targets: 
        """
        samples = int(samples/2)
        for i in range(0, samples):
            if random_img:
                # To extract 10 random images from the dataset:
                idx = np.random.randint(1, len(opt.input_path))
            image = dataset[idx]
            input_tensor = image[0]
            target_tensor = image[1]

            # Redo normalization of the input data:
            invTrans = unnormalize()
            inv_image = invTrans(input_tensor)

            input_arr = inv_image.cpu().detach().numpy() # To numpy
            target_arr = target_tensor.cpu().detach().numpy()

            # Plotting:
            # Input:
            ax.ravel()[i].imshow(np.transpose(inv_image, (1, 2, 0)), vmin=np.min(input_arr), vmax=np.max(input_arr))
            ax.ravel()[i].set_axis_off()
            ax.ravel()[i].set_title(f'Train input patch {idx}')

            # Target:
            # If shape of target HxW:
            if len(np.shape(target_arr)) == 2:
                # If shape of target is CxhxW:
                ax.ravel()[samples + i].imshow(target_arr, vmin=np.min(target_arr), vmax=np.max(target_arr))
                ax.ravel()[samples + i].set_axis_off()
                ax.ravel()[samples + i].set_title(f'Train target patch {idx}')
            elif len(np.shape(target_arr)) == 3:
                # If shape of target is CxhxW:
                ax.ravel()[samples +i].imshow(np.transpose(target_arr, (1, 2, 0)), vmin=np.min(target_arr),
                                              vmax=np.max(target_arr))
                ax.ravel()[samples +i].set_axis_off()
                ax.ravel()[samples +i].set_title(f'Train target patch {idx}')
        plt.tight_layout(pad=1)
        plt.show()

        return input_arr, target_arr

