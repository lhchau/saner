import os
import torch
import torchvision.transforms as transforms
from .cutout import Cutout
from torchvision import datasets
import numpy as np
from PIL import Image


class TINYIMAGENETNoisy(datasets.ImageFolder):
    def __init__(self, root, transform=None, noise_type='symmetric', target_transform=None, noise_rate=0.25):
        super(TINYIMAGENETNoisy, self).__init__(root, transform=transform, target_transform=target_transform)
        self.noise_rate = noise_rate
        self.num_classes = 200

        self.noisy_labels = self.targets.copy()  # Copy the original labels
        self._apply_noise(noise_type)

    def _apply_noise(self, noise_type):
        num_samples = len(self.noisy_labels)
        num_noisy = int(self.noise_rate * num_samples)
        noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)

        self.flip_labels = torch.zeros(num_samples, dtype=torch.bool)
        self.flip_labels[noisy_indices] = True

        for idx in noisy_indices:
            current_label = self.noisy_labels[idx]
            if noise_type == 'symmetric':
                new_label = np.random.choice([x for x in range(self.num_classes) if x != current_label])
            self.noisy_labels[idx] = new_label
            
    def __getitem__(self, index):
        img, target, flip_label = self.samples[index][0], self.noisy_labels[index], self.flip_labels[index]

        img = self.loader(img)
        
        # Apply the transformations if any
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, flip_label
    
def get_tiny_imagenet(
    batch_size,
    num_workers,
    noise,
    noise_type
):
    transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    
    data_train = TINYIMAGENETNoisy(os.path.join('.', 'data', 'tiny-imagenet-200', 'train'), transform_train, noise_type=noise_type, noise_rate=noise)
    data_test = datasets.ImageFolder (os.path.join('.', 'data', 'tiny-imagenet-200', 'test'), transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    
    
    return train_dataloader, test_dataloader, 200