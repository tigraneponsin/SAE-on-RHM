from itertools import *
import warnings
import copy

import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import dec2bin, dec2base

def sample_rules( v, n, m, s, L, seed=42):
        """
        Sample random rules for a random hierarchy model.

        Args:
            v: The number of values each variable can take (vocabulary size, int).
            n: The number of classes (int).
            m: The number of synonymic lower-level representations (multiplicity, int).
            s: The size of lower-level representations (int).
            L: The number of levels in the hierarchy (int).
            seed: Seed for generating the rules.

        Returns:
            A dictionary containing the rules for each level of the hierarchy.
        """
        random.seed(seed)
        tuples = list(product(*[range(v) for _ in range(s)]))

        rules = {}
        rules[0] = torch.tensor(
                random.sample( tuples, n*m)
        ).reshape(n,m,-1)
        for i in range(1, L):
            rules[i] = torch.tensor(
                    random.sample( tuples, v*m)
            ).reshape(v,m,-1)

        return rules


def sample_data_from_generator_classes(g, y, rules, return_tree_structure=False):
    """
    Create data of the Random Hierarchy Model starting from its rules, a seed and a set of class labels.

    Args:
        g: A torch.Generator object.
        y: A tensor of size [batch_size, 1] containing the class labels.
        rules: A dictionary containing the rules for each level of the hierarchy.
        return_tree_structure: If True, return the tree structure of the hierarchy as a dictionary.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    labels = copy.deepcopy(y)

    if return_tree_structure:
        x_st = (
            {}
        )  # Initialize the dictionary to store the hidden variables
        x_st[0] = y
        for i in range(L):  # Loop over the levels of the hierarchy
            chosen_rule = torch.randint(
                low=0, high=rules[i].shape[1], size=x_st[i].shape, generator=g
            )  # Choose a random rule for each variable in the current level
            x_st[i + 1] = rules[i][x_st[i], chosen_rule].flatten(
                start_dim=1
            )  # Apply the chosen rule to each variable in the current level
        return x_st, labels
    else:
        x = y
        for i in range(L):
            chosen_rule = torch.randint(
                low=0, high=rules[i].shape[1], size=x.shape, generator=g
            )
            x = rules[i][x, chosen_rule].flatten(start_dim=1)
        return x, labels
    

def sample_with_replacement(train_size, test_size, seed_sample, rules):

    n = rules[0].shape[0]  # Number of classes

    if train_size == -1:
        warnings.warn(
            "Whole dataset (train_size=-1) not available with replacement! Using train_size=1e6.",
            RuntimeWarning,
        )
        train_size = 1000000

    g = torch.Generator()
    g.manual_seed(seed_sample)

    y = torch.randint(low=0, high=n, size=(train_size + test_size,), generator=g)
    features, labels = sample_data_from_generator_classes(g, y, rules)

    return features, labels


def sample_data_from_indices(samples, rules, n, m, s, L, return_tree_structure=False):
    """
    Create data of the Random Hierarchy Model starting from a set of rules and the sampled indices.

    Args:
        samples: A tensor of size [batch_size, I], with I from 0 to max_data-1, containing the indices of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    max_data = n * m ** ((s**L-1)//(s-1))
    data_per_hl = max_data // n 	# div by num_classes to get number of data per class

    high_level = samples.div(data_per_hl, rounding_mode='floor')	# div by data_per_hl to get class index (run in range(n))
    low_level = samples % data_per_hl					# compute remainder (run in range(data_per_hl))

    labels = high_level	# labels are the classes (features of highest level)
    features = labels		# init input features as labels (rep. size 1)
    size = 1

    if return_tree_structure:
        features_dict = (
            {}
        )  # Initialize the dictionary to store the hidden variables
        features_dict[0] = copy.deepcopy(features)
        for l in range(L):

            choices = m**(size)
            data_per_hl = data_per_hl // choices	# div by num_choices to get number of data per high-level feature

            high_level = low_level.div( data_per_hl, rounding_mode='floor')	# div by data_per_hl to get high-level feature index (1 index in range(m**size))
            high_level = dec2base(high_level, m, length=size).squeeze()	# convert to base m (size indices in range(m), squeeze needed if index already in base m)

            features = rules[l][features, high_level]	        		# apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
            features = features.flatten(start_dim=1)				# flatten to tensor of size (batch_size, size*s)
            features_dict[l+1] = copy.deepcopy(features)
            size *= s								# rep. size increases by s at each level

            low_level = low_level % data_per_hl				# compute remainder (run in range(data_per_hl))

        return features_dict, labels

    else:
        for l in range(L):

            choices = m**(size)
            data_per_hl = data_per_hl // choices	# div by num_choices to get number of data per high-level feature

            high_level = low_level.div( data_per_hl, rounding_mode='floor')	# div by data_per_hl to get high-level feature index (1 index in range(m**size))
            high_level = dec2base(high_level, m, length=size).squeeze()	# convert to base m (size indices in range(m), squeeze needed if index already in base m)

            features = rules[l][features, high_level]	        		# apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
            features = features.flatten(start_dim=1)				# flatten to tensor of size (batch_size, size*s)
            size *= s								# rep. size increases by s at each level

            low_level = low_level % data_per_hl				# compute remainder (run in range(data_per_hl))

        return features, labels


def sample_without_replacement(max_data, train_size, test_size, seed_sample, rules):

    L = len(rules)  # Number of levels in the hierarchy
    n = rules[0].shape[0]  # Number of classes
    m = rules[0].shape[1]  # Number of synonymic lower-level representations
    s = rules[0].shape[2]  # Size of lower-level representations

    if train_size == -1:
        samples = torch.arange(max_data)
    else:
        test_size = min(test_size, max_data - train_size)

        random.seed(seed_sample)
        samples = torch.tensor(random.sample(range(max_data), train_size + test_size))

    features, labels = sample_data_from_indices(samples, rules, n, m, s, L)

    return features, labels


class RandomHierarchyModel(Dataset):
    """
    Implement the Random Hierarchy Model (RHM) as a PyTorch dataset.
    """

    def __init__(
            self,
            num_features=8,
            num_classes=2,
            num_synonyms=2,
            tuple_size=2,	# size of the low-level representations
            num_layers=2,
            seed_rules=0,
            seed_sample=1,
            train_size=-1,
            test_size=0,
            input_format='onehot',
            whitening=0,
            transform=None,
            replacement=False,
    ):

        self.num_features = num_features
        self.num_synonyms = num_synonyms 
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.tuple_size = tuple_size

        rules = sample_rules( num_features, num_classes, num_synonyms, tuple_size, num_layers, seed=seed_rules)
 
        max_data = num_classes * num_synonyms ** ((tuple_size ** num_layers - 1) // (tuple_size - 1))
        assert train_size >= -1, "train_size must be greater than or equal to -1"

        if max_data > 1e19 and not replacement:
            print(
                "Max dataset size cannot be represented with int64! Using sampling with replacement."
            )
            warnings.warn(
                "Max dataset size cannot be represented with int64! Using sampling with replacement.",
                RuntimeWarning,
            )
            replacement = True

        if not replacement:
            self.features, self.labels = sample_without_replacement(
                max_data, train_size, test_size, seed_sample, rules
            )
        else:
            self.features, self.labels = sample_with_replacement(
                train_size, test_size, seed_sample, rules
            )

        if 'onehot' not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

	# TODO: implement one-hot encoding of s-tuples
        if 'onehot' in input_format:

            self.features = F.one_hot(
                self.features.long(),
                num_classes=num_features if 'tuples' not in input_format else num_features ** tuple_size
            ).float()
            
            if whitening:

                inv_sqrt_norm = (1.-1./num_features) ** -.5
                self.features = (self.features - 1./num_features) * inv_sqrt_norm

            self.features = self.features.permute(0, 2, 1)

        elif 'long' in input_format:
            self.features = self.features.long() + 1

        else:
            raise ValueError

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
        	idx: sample index

        Returns:
            Feature-label pairs at index            
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y

    def get_rules(self):
        return self.rules
