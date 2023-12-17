"""
Dataset loader, only normalize input data. No normalization on output data.

"""
import os

IMG_EXTENSIONS = ['.tif', '.TIF', '.tiff', '.TIFF']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), f" {dir} is not a valid directory"

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
