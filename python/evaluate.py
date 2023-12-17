import torch


def evaluate(net, dataset_val):
    net.eval()

    # turn off gradients for validation
    with torch.no_grad():
        for idx_val, data_val in enumerate(dataset_val):  # inner loop within one training epoch

            # Forward pass without gradients:
            net.set_input(data_val)  # unpack data from dataset and apply preprocessing
            net.test()  # Run inference, don't do optimization or backward

            val_losses = net.get_current_losses()

    # Set model back to train:
    net.train()

    return val_losses
