
import torch
import os
import numpy as np
from PIL import Image

class Load_numpy_data():

    def __init__(self, root, transform = None):

        self.transform = transform

        self.train_data, self.label = torch.load(root)

        self.train_data = torch.from_numpy((self.train_data).numpy().transpose((0, 2, 3, 1)).astype(np.uint8))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_g = self.train_data[index]
        img_g = Image.fromarray(img_g.numpy())
        img_g = self.transform(img_g)

        label = self.label[index]

        return img_g, label

    def __len__(self):
        return self.train_data.size()[0]

import h5py as h5
import torch
import torch.utils.data as data


class ILSVRC_HDF5(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 load_in_mem=True, train=True, download=False, validate_seed=0,
                 val_split=0, **kwargs):  # last four are dummies

        self.root = root
        self.num_imgs = len(h5.File(root, 'r')['labels'])

        # self.transform = transform
        self.target_transform = target_transform

        # Set the transform here
        self.transform = transform

        # load the entire dataset into memory?
        self.load_in_mem = load_in_mem

        # If loading into memory, do so now
        if self.load_in_mem:
            print('Loading %s into memory...' % root)
            with h5.File(root, 'r') as f:
                self.data = f['imgs'][:]
                self.labels = f['labels'][:]

        images = self.data
        labels = self.labels
        chosen_test = np.ndarray.tolist(np.random.choice(images.shape[0], int(images.shape[0] * 0.2), replace=False))
        chosen_train = list(set(np.ndarray.tolist(np.arange(images.shape[0]))) - set(chosen_test))
        chosen_test = np.asarray(chosen_test)
        chosen_train = np.asarray(chosen_train)

        print(chosen_test.shape, chosen_train.shape)

        torch.save([images[chosen_train], labels[chosen_train]], 'original.pt')

        torch.save([images[chosen_test], labels[chosen_test]], 'test.pt')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # If loaded the entire dataset in RAM, get image from memory
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]

        # Else load it from disk
        else:
            with h5.File(self.root, 'r') as f:
                img = f['imgs'][index]
                target = f['labels'][index]

        # if self.transform is not None:
        # img = self.transform(img)
        # Apply my own transform
        img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, int(target)

    def __len__(self):
        return self.num_imgs
