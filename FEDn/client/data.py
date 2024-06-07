import os
from math import floor

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)



def load_data(data_path, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/clients/1/mnist.pt")

    data = torch.load(data_path)

    if is_train:
        X = data["x_train"]
        y = data["y_train"]
    else:
        X = data["x_test"]
        y = data["y_test"]

    # Normalize
    X = X / 255

    return X, y


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return result


def split(out_dir="data"):
    n_splits = 5

    # Make dir
    if not os.path.exists(f"{out_dir}/clients"):
        os.mkdir(f"{out_dir}/clients")

    # Make splits
    for i in range(n_splits):
        client_path =f'../data/client{i}'
        x_train = np.load(os.path.join(client_path, 'trainx.pyp'), allow_pickle=True)
        y_train = np.load(os.path.join(client_path, 'trainy.pyp'), allow_pickle=True)
        x_test = np.load(os.path.join(client_path, 'testx.pyp'), allow_pickle=True)
        y_test = np.load(os.path.join(client_path, 'testy.pyp'), allow_pickle=True)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        def apply_transforms(data):
            return torch.stack([transform(img) for img in data])
        x_train = apply_transforms(x_train)
        x_test = apply_transforms(x_test)
        
        y_train = torch.argmax(torch.tensor(y_train, dtype=torch.float32), dim=1)
        y_test = torch.argmax(torch.tensor(y_test, dtype=torch.float32), dim=1)

        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        torch.save(
            {
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
            },
            f"{subdir}/mnist.pt",
        )


if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        split()
