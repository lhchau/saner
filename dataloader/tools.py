import torch
import numpy as np
from math import inf
from PIL import Image

import torch.nn.functional as F

import random
from collections import defaultdict
from torch.utils.data import Sampler


class ClassBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, samples_per_class=2, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last

        # Map class labels to indices
        self.class_to_indices = defaultdict(list)
        for idx, (_, label, _) in enumerate(dataset):
            self.class_to_indices[label].append(idx)

        self.labels = list(self.class_to_indices.keys())
        self.num_classes_per_batch = self.batch_size // samples_per_class

        # Compute max number of full batches possible
        self.num_batches = self._compute_num_batches()

    def _compute_num_batches(self):
        # How many full batches can we make before any class runs out?
        max_batches_per_class = {
            cls: len(indices) // self.samples_per_class
            for cls, indices in self.class_to_indices.items()
        }

        # Total batches limited by the lowest among all classes
        max_per_class = min(max_batches_per_class.values())
        total_classes = len(self.labels)
        return (max_per_class * total_classes) // self.num_classes_per_batch

    def __iter__(self):
        # Copy and shuffle indices per class
        class_indices = {cls: indices.copy() for cls, indices in self.class_to_indices.items()}
        for idx_list in class_indices.values():
            random.shuffle(idx_list)

        batches_yielded = 0
        while batches_yielded < self.num_batches:
            # Sample classes for the current batch
            selected_classes = random.sample(self.labels, self.num_classes_per_batch)
            batch = []

            for cls in selected_classes:
                batch.extend(class_indices[cls][:self.samples_per_class])
                del class_indices[cls][:self.samples_per_class]

            if len(batch) == self.batch_size:
                yield batch
                batches_yielded += 1
            elif not self.drop_last and len(batch) > 0:
                yield batch
                break
            else:
                break

    def __len__(self):
        return self.num_batches

class DependentLabelGenerator:
    def __init__(self, num_classes, feature_size, transform):
        self.W = torch.FloatTensor(np.random.randn(num_classes, feature_size, num_classes))
        self.num_classes = num_classes
        self.transform = transform
    def generate_dependent_labels(self, data, target):
        # 1*m *  m*10 = 1*10
        img = Image.fromarray(data)
        img = self.transform(img)
        A = img.view(1, -1).mm(self.W[target]).squeeze(0)
        A[target] = -inf
        A = F.softmax(A, dim=0)

        new_label = int(np.random.choice(list(range(self.num_classes)), p=A.cpu().numpy()))
        return new_label