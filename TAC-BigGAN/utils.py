#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Utilities file
This file contains utility functions for bookkeeping, logging, and data loading.
Methods which directly affect training should either go in layers, the model,
or train_fns.py.
'''

from __future__ import print_function
import sys
import os
import numpy as np
import time
import datetime
import json
import pickle
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import datasets as dset


def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = ArgumentParser(description=usage)

    ### Dataset/Dataloader stuff ###
    parser.add_argument(
        '--dataset', type=str, default='I128_hdf5',
        help='Which Dataset to train on, out of I128, I256, C10, C100;'
             'Append "_hdf5" to use the hdf5 version for ISLVRC '
             '(default: %(default)s)')
    parser.add_argument(
        '--augment', action='store_true', default=False,
        help='Augment with random crops and flips (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of dataloader workers; consider using less for HDF5 '
             '(default: %(default)s)')
    parser.add_argument(
        '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
        help='Pin data into memory through dataloader? (default: %(default)s)')
    parser.add_argument(
        '--shuffle', action='store_true', default=False,
        help='Shuffle the data (strongly recommended)? (default: %(default)s)')
    parser.add_argument(
        '--load_in_mem', action='store_true', default=False,
        help='Load all data into memory? (default: %(default)s)')
    parser.add_argument(
        '--use_multiepoch_sampler', action='store_true', default=False,
        help='Use the multi-epoch sampler for dataloader? (default: %(default)s)')



    parser.add_argument(
        '--loss_type', type=str, default='Twin_AC',
        help='Loss function,"Twin_AC","AC","Projection"')
    parser.add_argument(
        '--AC', action='store_true', default=False,
        help='Whether use auxilary classifier loss, turn on for "Twin_AC" and "AC" ')
    parser.add_argument(
        '--AC_weight', type=float, default=1.0,
        help='The weight for auxilary classifier loss')
    parser.add_argument(
        '--Pac', action='store_true', default=False,
        help='Extra pac loss to improve GAN performance')
    parser.add_argument(
        '--G_param', type=str, default='SN',
        help='Parameterization style to use for G, spectral norm (SN) or SVD (SVD)'
             ' or None (default: %(default)s)')
    parser.add_argument(
        '--D_param', type=str, default='SN',
        help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)'
             ' or None (default: %(default)s)')
    parser.add_argument(
        '--G_ch', type=int, default=64,
        help='Channel multiplier for G (default: %(default)s)')
    parser.add_argument(
        '--D_ch', type=int, default=64,
        help='Channel multiplier for D (default: %(default)s)')
    parser.add_argument(
        '--G_depth', type=int, default=1,
        help='Number of resblocks per stage in G? (default: %(default)s)')
    parser.add_argument(
        '--D_depth', type=int, default=1,
        help='Number of resblocks per stage in D? (default: %(default)s)')
    parser.add_argument(
        '--D_thin', action='store_false', dest='D_wide', default=True,
        help='Use the SN-GAN channel pattern for D? (default: %(default)s)')
    parser.add_argument(
        '--G_shared', action='store_true', default=False,
        help='Use shared embeddings in G? (default: %(default)s)')
    parser.add_argument(
        '--shared_dim', type=int, default=0,
        help='G''s shared embedding dimensionality; if 0, will be equal to dim_z. '
             '(default: %(default)s)')
    parser.add_argument(
        '--dim_z', type=int, default=128,
        help='Noise dimensionality: %(default)s)')
    parser.add_argument(
        '--z_var', type=float, default=1.0,
        help='Noise variance: %(default)s)')
    parser.add_argument(
        '--hier', action='store_true', default=False,
        help='Use hierarchical z in G? (default: %(default)s)')
    parser.add_argument(
        '--mybn', action='store_true', default=False,
        help='Use my batchnorm (which supports standing stats?) %(default)s)')
    parser.add_argument(
        '--G_nl', type=str, default='relu',
        help='Activation function for G (default: %(default)s)')
    parser.add_argument(
        '--D_nl', type=str, default='relu',
        help='Activation function for D (default: %(default)s)')
    parser.add_argument(
        '--G_attn', type=str, default='64',
        help='What resolutions to use attention on for G (underscore separated) '
             '(default: %(default)s)')
    parser.add_argument(
        '--D_attn', type=str, default='64',
        help='What resolutions to use attention on for D (underscore separated) '
             '(default: %(default)s)')
    parser.add_argument(
        '--norm_style', type=str, default='bn',
        help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
             'ln [layernorm], gn [groupnorm] (default: %(default)s)')

    ### Model init stuff ###
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use; affects both initialization and '
             ' dataloading. (default: %(default)s)')
    parser.add_argument(
        '--G_init', type=str, default='ortho',
        help='Init style to use for G (default: %(default)s)')
    parser.add_argument(
        '--D_init', type=str, default='ortho',
        help='Init style to use for D(default: %(default)s)')
    parser.add_argument(
        '--skip_init', action='store_true', default=False,
        help='Skip initialization, ideal for testing when ortho init was used '
             '(default: %(default)s)')

    ### Optimizer stuff ###
    parser.add_argument(
        '--G_lr', type=float, default=5e-5,
        help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_lr', type=float, default=2e-4,
        help='Learning rate to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B1', type=float, default=0.0,
        help='Beta1 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B1', type=float, default=0.0,
        help='Beta1 to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B2', type=float, default=0.999,
        help='Beta2 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B2', type=float, default=0.999,
        help='Beta2 to use for Discriminator (default: %(default)s)')

    ### Batch size, parallel ###
    parser.add_argument(
        '--batch_size', type=int, default=50,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--G_batch_size', type=int, default=0,
        help='Batch size to use for G; if 0, same as D (default: %(default)s)')
    parser.add_argument(
        '--num_G_steps', type=int, default=2,
        help='Number of D steps per G step (default: %(default)s)')
    parser.add_argument(
        '--num_G_accumulations', type=int, default=1,
        help='Number of passes to accumulate G''s gradients over '
             '(default: %(default)s)')
    parser.add_argument(
        '--num_D_steps', type=int, default=2,
        help='Number of D steps per G step (default: %(default)s)')
    parser.add_argument(
        '--num_D_accumulations', type=int, default=1,
        help='Number of passes to accumulate D''s gradients over '
             '(default: %(default)s)')
    parser.add_argument(
        '--split_D', action='store_true', default=False,
        help='Run D twice rather than concatenating inputs? (default: %(default)s)')
    parser.add_argument(
        '--num_epochs', type=int, default=100,
        help='Number of epochs to train for (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Train with multiple GPUs (default: %(default)s)')

    parser.add_argument(
        '--accumulate_stats', action='store_true', default=False,
        help='Accumulate "standing" batchnorm stats? (default: %(default)s)')
    parser.add_argument(
        '--num_standing_accumulations', type=int, default=16,
        help='Number of forward passes to use in accumulating standing stats? '
             '(default: %(default)s)')

    ### Bookkeping stuff ###
    parser.add_argument(
        '--G_eval_mode', action='store_true', default=False,
        help='Run G in eval mode (running/standing stats?) at sample/test time? '
             '(default: %(default)s)')
    parser.add_argument(
        '--save_every', type=int, default=2000,
        help='Save every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_save_copies', type=int, default=2,
        help='How many copies to save (default: %(default)s)')
    parser.add_argument(
        '--num_best_copies', type=int, default=2,
        help='How many previous best checkpoints to save (default: %(default)s)')
    parser.add_argument(
        '--which_best', type=str, default='IS',
        help='Which metric to use to determine when to save new "best"'
             'checkpoints, one of IS or FID (default: %(default)s)')
    parser.add_argument(
        '--no_fid', action='store_true', default=False,
        help='Calculate IS only, not FID? (default: %(default)s)')
    parser.add_argument(
        '--test_every', type=int, default=5000,
        help='Test every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_inception_images', type=int, default=50000,
        help='Number of samples to compute inception metrics with '
             '(default: %(default)s)')
    parser.add_argument(
        '--base_root', type=str, default='../Biggan_result',
        help='Default location to store all weights, samples, data, and logs '
             ' (default: %(default)s)')
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--weights_root', type=str, default='weights',
        help='Default location to store weights (default: %(default)s)')
    parser.add_argument(
        '--logs_root', type=str, default='logs',
        help='Default location to store logs (default: %(default)s)')
    parser.add_argument(
        '--samples_root', type=str, default='samples',
        help='Default location to store samples (default: %(default)s)')
    parser.add_argument(
        '--name_suffix', type=str, default='',
        help='Suffix for experiment name for loading weights for sampling '
             '(consider "best0") (default: %(default)s)')
    parser.add_argument(
        '--experiment_name', type=str, default='',
        help='Optionally override the automatic experiment naming with this arg. '
             '(default: %(default)s)')
    parser.add_argument(
        '--config_from_name', action='store_true', default=False,
        help='Use a hash of the experiment name instead of the full config '
             '(default: %(default)s)')

    ### EMA Stuff ###
    parser.add_argument(
        '--ema', action='store_true', default=False,
        help='Keep an ema of G''s weights? (default: %(default)s)')
    parser.add_argument(
        '--ema_decay', type=float, default=0.9999,
        help='EMA decay rate (default: %(default)s)')
    parser.add_argument(
        '--use_ema', action='store_true', default=False,
        help='Use the EMA parameters of G for evaluation? (default: %(default)s)')
    parser.add_argument(
        '--ema_start', type=int, default=0,
        help='When to start updating the EMA weights (default: %(default)s)')

    parser.add_argument(
        '--toggle_grads', action='store_true', default=True,
        help='Toggle D and G''s "requires_grad" settings when not training them? '
             ' (default: %(default)s)')


    ### Resume training stuff
    parser.add_argument(
        '--load_weights', type=str, default='',
        help='Suffix for which weights to load (e.g. best0, copy0) '
             '(default: %(default)s)')
    parser.add_argument(
        '--resume', action='store_true', default=True,
        help='Resume training? (default: %(default)s)')

    ### Log stuff ###
    parser.add_argument(
        '--logstyle', type=str, default='%3.3e',
        help='What style to use when logging training metrics?'
             'One of: %#.#f/ %#.#e (float/exp, text),'
             'pickle (python pickle),'
             'npz (numpy zip),'
             'mat (MATLAB .mat file) (default: %(default)s)')

    return parser


# Arguments for sample.py; not presently used in train.py
def add_sample_parser(parser):
    parser.add_argument(
        '--sample_npz', action='store_true', default=False,
        help='Sample "sample_num_npz" images and save to npz? '
             '(default: %(default)s)')
    parser.add_argument(
        '--sample_num_npz', type=int, default=50000,
        help='Number of images to sample when sampling NPZs '
             '(default: %(default)s)')
    parser.add_argument(
        '--sample_sheets', action='store_true', default=False,
        help='Produce class-conditional sample sheets and stick them in '
             'the samples root? (default: %(default)s)')
    parser.add_argument(
        '--sample_interps', action='store_true', default=False,
        help='Produce interpolation sheets and stick them in '
             'the samples root? (default: %(default)s)')
    parser.add_argument(
        '--sample_sheet_folder_num', type=int, default=-1,
        help='Number to use for the folder for these sample sheets '
             '(default: %(default)s)')
    parser.add_argument(
        '--sample_random', action='store_true', default=False,
        help='Produce a single random sheet? (default: %(default)s)')
    parser.add_argument(
        '--sample_trunc_curves', type=str, default='',
        help='Get inception metrics with a range of variances?'
             'To use this, specify a startpoint, step, and endpoint, e.g. '
             '--sample_trunc_curves 0.2_0.1_1.0 for a startpoint of 0.2, '
             'endpoint of 1.0, and stepsize of 1.0.  Note that this is '
             'not exactly identical to using tf.truncated_normal, but should '
             'have approximately the same effect. (default: %(default)s)')
    parser.add_argument(
        '--sample_inception_metrics', action='store_true', default=False,
        help='Calculate Inception metrics with sample.py? (default: %(default)s)')
    return parser


# Convenience dicts
dset_dict = {'I32': dset.ImageFolder, 'I64': dset.ImageFolder,'VGGFACE':dset.ILSVRC_HDF5,
             'I128': dset.ImageFolder, 'I256': dset.ImageFolder,
             'I32_hdf5': dset.ILSVRC_HDF5, 'I64_hdf5': dset.ILSVRC_HDF5,
             'I128_hdf5': dset.ILSVRC_HDF5, 'I256_hdf5': dset.ILSVRC_HDF5,
             'C10': dset.CIFAR10, 'C100': dset.CIFAR100}
imsize_dict = {'I32': 32, 'I32_hdf5': 32,'VGGFACE':64,
               'I64': 64, 'I64_hdf5': 64,
               'I128': 128, 'I128_hdf5': 128,
               'I256': 256, 'I256_hdf5': 256,
               'C10': 32, 'C100': 32}
root_dict = {'I32': 'ImageNet', 'I32_hdf5': 'ILSVRC32.hdf5','VGGFACE':'VGGFACE64.hdf5',
             'I64': 'ImageNet', 'I64_hdf5': 'ILSVRC64.hdf5',
             'I128': 'ImageNet', 'I128_hdf5': 'ILSVRC128.hdf5',
             'I256': 'ImageNet', 'I256_hdf5': 'ILSVRC256.hdf5',
             'C10': 'cifar', 'C100': 'cifar'}
nclass_dict = {'I32': 1000, 'I32_hdf5': 1000,'VGGFACE':2000,
               'I64': 1000, 'I64_hdf5': 1000,
               'I128': 1000, 'I128_hdf5': 1000,
               'I256': 1000, 'I256_hdf5': 1000,
               'C10': 10, 'C100': 100}
# Number of classes to put per sample sheet
classes_per_sheet_dict = {'I32': 50, 'I32_hdf5': 50,'VGGFACE':100,
                          'I64': 50, 'I64_hdf5': 50,
                          'I128': 20, 'I128_hdf5': 20,
                          'I256': 20, 'I256_hdf5': 20,
                          'C10': 10, 'C100': 100}
activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True), }


class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """

    def __call__(self, img):
        """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class RandomCropLongEdge(object):
    """Crops the given PIL Image on the long edge with a random start point.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """

    def __call__(self, img):
        """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
             else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
             else np.random.randint(low=0, high=img.size[1] - size[1]))
        return transforms.functional.crop(img, i, j, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


# multi-epoch Dataset sampler to avoid memory leakage and enable resumption of
# training from the same sample regardless of if we stop mid-epoch
class MultiEpochSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly over multiple epochs

  Arguments:
      data_source (Dataset): dataset to sample from
      num_epochs (int) : Number of times to loop over the dataset
      start_itr (int) : which iteration to begin from
  """

    def __init__(self, data_source, num_epochs, start_itr=0, batch_size=128):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.num_epochs = num_epochs
        self.start_itr = start_itr
        self.batch_size = batch_size

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))

    def __iter__(self):
        n = len(self.data_source)
        # Determine number of epochs
        num_epochs = int(np.ceil((n * self.num_epochs
                                  - (self.start_itr * self.batch_size)) / float(n)))
        # Sample all the indices, and then grab the last num_epochs index sets;
        # This ensures if we're starting at epoch 4, we're still grabbing epoch 4's
        # indices
        out = [torch.randperm(n) for epoch in range(self.num_epochs)][-num_epochs:]
        # Ignore the first start_itr % n indices of the first epoch
        out[0] = out[0][(self.start_itr * self.batch_size % n):]
        # if self.replacement:
        # return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        # return iter(.tolist())
        output = torch.cat(out).tolist()
        print('Length dataset output is %d' % len(output))
        return iter(output)

    def __len__(self):
        return len(self.data_source) * self.num_epochs - self.start_itr * self.batch_size


# Convenience function to centralize all data loaders
def get_data_loaders(dataset, data_root, batch_size=50,
                     num_workers=8, shuffle=True, load_in_mem=False, hdf5=False,
                     pin_memory=True, drop_last=True, start_itr=0,
                     num_epochs=500, use_multiepoch_sampler=False,
                     **kwargs):
    # Append /FILENAME.hdf5 to root if using hdf5
    print(data_root, dataset, root_dict[dataset])
    data_root += '/%s' % root_dict[dataset]
    print('Using dataset root location %s' % data_root)

    which_dataset = dset_dict[dataset]
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    image_size = imsize_dict[dataset]
    # For image folder datasets, name of the file where we store the precomputed
    # image locations to avoid having to walk the dirs every time we load.
    dataset_kwargs = {'index_filename': '%s_imgs.npz' % dataset}

    # HDF5 datasets have their own inbuilt transform, no need to train_transform
    if 'hdf5' in dataset:
        train_transform = None
    else:
        if dataset in ['C10', 'C100']:
            train_transform = []
        else:
            train_transform = [CenterCropLongEdge(), transforms.Resize(image_size)]
        train_transform = transforms.Compose(train_transform + [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])
    train_set = which_dataset(root=data_root, transform=train_transform,
                              load_in_mem=load_in_mem, **dataset_kwargs)

    # Prepare loader; the loaders list is for forward compatibility with
    # using validation / test splits.
    loaders = []
    if use_multiepoch_sampler:
        print('Using multiepoch sampler from start_itr %d...' % start_itr)
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
        sampler = MultiEpochSampler(train_set, num_epochs, start_itr, batch_size)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=sampler, **loader_kwargs)
    else:
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                         'drop_last': drop_last}  # Default, drop last incomplete batch
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=shuffle, **loader_kwargs)
    loaders.append(train_loader)
    return loaders


# Utility file to seed rngs
def seed_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


# Utility to peg all roots to a base root
# If a base root folder is provided, peg all other root folders to it.
def update_config_roots(config):
    if config['base_root']:
        print('Pegging all root folders to base root %s' % config['base_root'])
        for key in ['data', 'weights', 'logs', 'samples']:
            config['%s_root' % key] = '%s/%s' % (config['base_root'], key)
    return config


# Utility to prepare root folders if they don't exist; parent folder must exist
def prepare_root(config):
    for key in ['weights_root', 'logs_root', 'samples_root']:
        if not os.path.exists(config[key]):
            print('Making directory %s for %s...' % (config[key], key))
            os.mkdir(config[key])

        if not os.path.exists(config[key]+'/' + config['dataset']):
            print('Making directory %s for %s...' % (config[key], key))
            os.mkdir(config[key]+'/' + config['dataset'])


# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class ema(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
                # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                                 + self.source_dict[key].data * (1 - decay))


# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off


# Function to join strings or ignore them
# Base string is the string to link "strings," while strings
# is a list of strings or Nones.
def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])


# Save a model's weights, optimizer, and the state_dict
def save_weights(G, D, state_dict, weights_root, experiment_name,
                 name_suffix=None, G_ema=None):
    root = '/'.join([weights_root, experiment_name])
    if not os.path.exists(root):
        os.mkdir(root)
    if name_suffix:
        print('Saving weights to %s/%s...' % (root, name_suffix))
    else:
        print('Saving weights to %s...' % root)
    torch.save(G.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['G', name_suffix])))
    torch.save(G.optim.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix])))
    torch.save(D.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['D', name_suffix])))
    torch.save(D.optim.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix])))
    torch.save(state_dict,
               '%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))
    if G_ema is not None:
        torch.save(G_ema.state_dict(),
                   '%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix])))


def load_weights(G, D, state_dict, weights_root, experiment_name,
                 name_suffix=None, G_ema=None, strict=True, load_optim=True):
    root = '/'.join([weights_root, experiment_name])
    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, root))
    else:
        print('Loading weights from %s...' % root)
    if G is not None:
        G.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G', name_suffix]))),
            strict=strict)
        if load_optim:
            G.optim.load_state_dict(
                torch.load('%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix]))))
    if D is not None:
        D.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['D', name_suffix]))),
            strict=strict)
        if load_optim:
            D.optim.load_state_dict(
                torch.load('%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix]))))
    # Load state dict
    for item in state_dict:
        state_dict[item] = torch.load('%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))[item]
    if G_ema is not None:
        G_ema.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix]))),
            strict=strict)


''' MetricsLogger originally stolen from VoxNet source code.
    Used for logging inception metrics'''


class MetricsLogger(object):
    def __init__(self, fname, reinitialize=False):
        self.fname = fname
        self.reinitialize = reinitialize
        if os.path.exists(self.fname):
            if self.reinitialize:
                print('{} exists, deleting...'.format(self.fname))
                os.remove(self.fname)

    def log(self, record=None, **kwargs):
        """
    Assumption: no newlines in the input.
    """
        if record is None:
            record = {}
        record.update(kwargs)
        record['_stamp'] = time.time()
        with open(self.fname, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')


# Logstyle is either:
# '%#.#f' for floating point representation in text
# '%#.#e' for exponent representation in text
# 'npz' for output to npz # NOT YET SUPPORTED
# 'pickle' for output to a python pickle # NOT YET SUPPORTED
# 'mat' for output to a MATLAB .mat file # NOT YET SUPPORTED
class MyLogger(object):
    def __init__(self, fname, reinitialize=False, logstyle='%3.3f'):
        self.root = fname
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.reinitialize = reinitialize
        self.metrics = []
        self.logstyle = logstyle  # One of '%3.3f' or like '%3.3e'

    # Delete log if re-starting and log already exists
    def reinit(self, item):
        if os.path.exists('%s/%s.log' % (self.root, item)):
            if self.reinitialize:
                # Only print the removal mess
                if 'sv' in item:
                    if not any('sv' in item for item in self.metrics):
                        print('Deleting singular value logs...')
                else:
                    print('{} exists, deleting...'.format('%s_%s.log' % (self.root, item)))
                os.remove('%s/%s.log' % (self.root, item))

    # Log in plaintext; this is designed for being read in MATLAB(sorry not sorry)
    def log(self, itr, **kwargs):
        for arg in kwargs:
            if arg not in self.metrics:
                if self.reinitialize:
                    self.reinit(arg)
                self.metrics += [arg]
            if self.logstyle == 'pickle':
                print('Pickle not currently supported...')
                # with open('%s/%s.log' % (self.root, arg), 'a') as f:
                # pickle.dump(kwargs[arg], f)
            elif self.logstyle == 'mat':
                print('.mat logstyle not currently supported...')
            else:
                with open('%s/%s.log' % (self.root, arg), 'a') as f:
                    f.write('%d: %s\n' % (itr, self.logstyle % kwargs[arg]))


# Write some metadata to the logs directory
def write_metadata(logs_root, experiment_name, config, state_dict):
    with open(('%s/%s/metalog.txt' %
               (logs_root, experiment_name)), 'w') as writefile:
        writefile.write('datetime: %s\n' % str(datetime.datetime.now()))
        writefile.write('config: %s\n' % str(config))
        writefile.write('state: %s\n' % str(state_dict))


"""
Very basic progress indicator to wrap an iterable in.

Author: Jan Schlüter
Andy's adds: time elapsed in addition to ETA, makes it possible to add
estimated time to 1k iters instead of estimated time to completion.
"""


def progress(items, desc='', total=None, min_delay=0.1, displaytype='s1k'):
    """
  Returns a generator over `items`, printing the number and percentage of
  items processed and the estimated remaining processing time before yielding
  the next item. `total` gives the total number of items (required if `items`
  has no length), and `min_delay` gives the minimum time in seconds between
  subsequent prints. `desc` gives an optional prefix text (end with a space).
  """
    total = total or len(items)
    t_start = time.time()
    t_last = 0
    for n, item in enumerate(items):
        t_now = time.time()
        if t_now - t_last > min_delay:
            print("\r%s%d/%d (%6.2f%%)" % (
                desc, n + 1, total, n / float(total) * 100), end=" ")
            if n > 0:

                if displaytype == 's1k':  # minutes/seconds for 1000 iters
                    next_1000 = n + (1000 - n % 1000)
                    t_done = t_now - t_start
                    t_1k = t_done / n * next_1000
                    outlist = list(divmod(t_done, 60)) + list(divmod(t_1k - t_done, 60))
                    print("(TE/ET1k: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")
                else:  # displaytype == 'eta':
                    t_done = t_now - t_start
                    t_total = t_done / n * total
                    outlist = list(divmod(t_done, 60)) + list(divmod(t_total - t_done, 60))
                    print("(TE/ETA: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")

            sys.stdout.flush()
            t_last = t_now
        yield item
    t_total = time.time() - t_start
    print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))


# Sample function for use with inception metrics
def sample(G, z_, y_, config):
    with torch.no_grad():
        z_.sample_()
        y_.sample_()
        if config['parallel']:
            G_z = nn.parallel.data_parallel(G, (z_, G.shared(y_)))
        else:
            G_z = G(z_, G.shared(y_))
        return G_z, y_


# Sample function for sample sheets
def sample_sheet(G, classes_per_sheet, num_classes, samples_per_class, parallel,
                 samples_root, experiment_name, folder_number, z_=None):
    # Prepare sample directory
    if not os.path.isdir('%s/%s' % (samples_root, experiment_name)):
        os.mkdir('%s/%s' % (samples_root, experiment_name))
    if not os.path.isdir('%s/%s/%d' % (samples_root, experiment_name, folder_number)):
        os.mkdir('%s/%s/%d' % (samples_root, experiment_name, folder_number))
    # loop over total number of sheets
    for i in range(num_classes // classes_per_sheet):
        ims = []
        y = torch.arange(i * classes_per_sheet, (i + 1) * classes_per_sheet, device='cuda')
        for j in range(samples_per_class):
            z_ = torch.randn(classes_per_sheet, G.dim_z, device='cuda')
            with torch.no_grad():
                if parallel:
                    o = nn.parallel.data_parallel(G, (z_[:classes_per_sheet], G.shared(y)))
                else:
                    o = G(z_[:classes_per_sheet], G.shared(y))

            ims += [o.data.cpu()]
        # This line should properly unroll the images
        out_ims = torch.stack(ims, 1).view(-1, ims[0].shape[1], ims[0].shape[2],
                                           ims[0].shape[3]).data.float().cpu()
        # The path for the samples
        image_filename = '%s/%s/%d/samples%d.jpg' % (samples_root, experiment_name,
                                                     folder_number, i)
        torchvision.utils.save_image(out_ims, image_filename,
                                     nrow=samples_per_class, normalize=True)



# Name an experiment based on its config
def name_from_config(config):
    name = '_'.join([
        item for item in [
            config['dataset']+'/',
            config['loss_type'],
            'Pac' if config['Pac'] else None,
            'AC_weight%3.1f' % config['AC_weight']
        ]
        if item is not None])
    return name



# Get GPU memory, -i is the index
def query_gpu(indices):
    os.system('nvidia-smi -i 0 --query-gpu=memory.free --format=csv')


# Convenience function to count the number of parameters in a module
def count_parameters(module):
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in module.parameters()])))


# Convenience function to sample an index, not actually a 1-hot
def sample_1hot(batch_size, num_classes, device='cuda'):
    return torch.randint(low=0, high=num_classes, size=(batch_size,),
                         device=device, dtype=torch.int64, requires_grad=False)


# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


# Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda',
                fp16=False, z_var=1.0):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=z_var)
    z_ = z_.to(device, torch.float16 if fp16 else torch.float32)

    if fp16:
        z_ = z_.half()

    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses)
    y_ = y_.to(device, torch.int64)
    return z_, y_



