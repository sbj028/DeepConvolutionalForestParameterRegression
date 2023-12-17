import torch
from torch.utils.data import Dataset, DataLoader
from .dataset import RemoteSensingDataset, AGBDataset, RS_AGBDataset


def find_dataset_using_name(dataset_mode):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "python.load_data.dataset"

    dataset = None
    if dataset_mode == 'remotesensing':
        # Only input data:
        dataset = RemoteSensingDataset
    elif dataset_mode == 'agb':
        # Only target data:
        dataset = AGBDataset
    elif dataset_mode == 'rs_agb':
        # Combined input and output data:
        dataset = RS_AGBDataset

    if dataset is None:
        raise NotImplementedError(f"In {dataset_filename}.py, there should be a Dataset class name that matches "
                                  f"{dataset_mode} in lower case letters.")
    return dataset


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        from data import create_dataset
        dataset = create_dataset(opt)
    """
    if opt.phase == 'train':
        data_loader = CustomTrainDatasetDataLoader(opt)
        train_dataset = data_loader.load_data()
        return train_dataset
    elif opt.phase == 'trainval':
        train_data_loader = CustomTrainDatasetDataLoader(opt)
        val_data_loader = CustomValDatasetDataLoader(opt)
        train_dataset = train_data_loader.load_data()
        val_dataset = val_data_loader.load_data()
        return train_dataset, val_dataset
    elif opt.phase == 'test':
        data_loader = CustomTestDatasetDataLoader(opt)
        test_dataset = data_loader.load_data()
        return test_dataset
    else:
        raise ValueError(f"Got phase: {opt.phase}, but must be train, trainval or test.")


class CustomTrainDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading

    Modiefied, from pix2pix implementation. """

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)

        if opt.dataset_mode == 'rs_agb':
            self.dataset = dataset_class(opt, opt.train_input_path, opt.train_target_path)
        elif opt.dataset_mode == 'remotesensing':
            self.dataset = dataset_class(opt, opt.train_input_path)
        elif opt.dataset_mode == 'agb':
            self.dataset = dataset_class(opt, opt.train_target_path)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data


class CustomValDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading

    Modified, from pix2pix implementation. """

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)

        if opt.dataset_mode == 'rs_agb':
            self.dataset_val = dataset_class(opt, opt.val_input_path, opt.val_target_path)  # val
        elif opt.dataset_mode == 'remotesensing':
            self.dataset_val = dataset_class(opt, opt.val_input_path)       # val
        elif opt.dataset_mode == 'agb':
            self.dataset_val = dataset_class(opt, opt.val_target_path)      # val

        self.dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=opt.batch_size,  # Use same batch size as for training
            shuffle=False,             # Don't shuffle validation data
            num_workers=int(opt.num_threads))  # Use same num_workers as for training

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the training dataset"""
        return min(len(self.dataset_val), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of validation data"""
        for j, val_data in enumerate(self.dataloader_val):
            if j * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield val_data


class CustomTestDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading

    Modiefied, from pix2pix implementation. """

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)

        if opt.dataset_mode == 'rs_agb':
            self.dataset_test = dataset_class(opt, opt.test_input_path, opt.test_target_path)  # test
        elif opt.dataset_mode == 'remotesensing':
            self.dataset_test = dataset_class(opt, opt.test_input_path)       # test
        elif opt.dataset_mode == 'agb':
            self.dataset_test = dataset_class(opt, opt.test_target_path)      # test

        self.dataloader_test = torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=1,  # Use same batch size as for training
            shuffle=False,             # Don't shuffle test data
            num_workers=0)   # For testing

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the training dataset"""
        return min(len(self.dataset_test), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of validation data"""
        for j, test_data in enumerate(self.dataloader_test):
            if j * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield test_data
