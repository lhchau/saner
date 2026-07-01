import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from .cutout import Cutout
from .tools import DependentLabelGenerator
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


class CIFAR10Noisy(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, noise_type='symmetric', transform=None, target_transform=None, download=False, noise_rate=0.2, data_size=1):
        super(CIFAR10Noisy, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.noise_rate = noise_rate
        self.num_classes = len(self.classes)
        
        self.data, self.targets = self.get_smaller_dataset(data_size)
        
        self.noisy_labels = self.targets.copy()  # Copy the original labels
        
        if noise_type == 'dependent':
            self.noise_label_gen = DependentLabelGenerator(self.num_classes, 32 * 32 * 3, transform) 
        elif noise_type == 'asymmetric':
            self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
        if self.train:
            self._apply_noise(noise_type)

    def get_smaller_dataset(self, data_size):
        if data_size == 1:
            return self.data, self.targets  # return full dataset if data_size is 100%

        if not (0 < data_size <= 1):
            raise ValueError("data_size should be a float between 0 and 1")

        # Use train_test_split to get a subset of the data, stratify ensures class balance
        X_small, _, y_small, _ = train_test_split(
            self.data, 
            self.targets, 
            train_size=data_size, 
            stratify=self.targets,  # Ensures each class is proportionally represented
            random_state=42  # Ensure reproducibility
        )

        return X_small, y_small

    def _apply_noise(self, noise_type):
        num_samples = len(self.noisy_labels)
        if noise_type == 'real':
            self.noisy_labels = get_real_world_cifar("cifar10")

            self.flip_labels = torch.zeros(num_samples, dtype=torch.bool)
            for idx, (target, noisy_target) in enumerate(zip(self.targets, self.noisy_labels)):
                if target != noisy_target:
                    self.flip_labels[idx] = True
        else:
            num_noisy = int(self.noise_rate * num_samples)
            noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)

            self.flip_labels = torch.zeros(num_samples, dtype=torch.bool)
            self.flip_labels[noisy_indices] = True

            for idx in noisy_indices:
                current_label = self.noisy_labels[idx]
                if noise_type == 'symmetric':
                    new_label = np.random.choice([x for x in range(self.num_classes) if x != current_label])
                elif noise_type == 'asymmetric':
                    new_label = self.transition[current_label]
                    if new_label == current_label:
                        self.flip_labels[idx] = False
                elif noise_type == 'dependent':
                    new_label = self.noise_label_gen.generate_dependent_labels(self.data[idx], current_label)
                self.noisy_labels[idx] = new_label
                
    def __getitem__(self, index):
        img = self.data[index]
        target = self.noisy_labels[index]
        flip_label = self.flip_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # IMPORTANT: return index for PLC/DivideMix
        return img, target, flip_label, index

    def set_label(self, index, new_label: int):
        self.noisy_labels[index] = int(new_label)

    def get_label(self, index):
        return int(self.noisy_labels[index])

    
def get_real_world_cifar(dataset="cifar10"):
    if dataset == "cifar10":
        noise_label = torch.load('./data/CIFAR-10_human.pt')
        noisy_label = noise_label['worse_label']
    else:
        noise_label = torch.load('./data/CIFAR-100_human.pt')
        noisy_label = noise_label['noisy_label']
    return noisy_label
 
def get_cifar10_index(
    batch_size=128,
    num_workers=4,
    noise=0.25,
    noise_type='symmetric',
    data_augmentation="standard", 
    data_size=1,
    use_val=False):
    if data_augmentation == "standard":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # Cutout()
        ])
    elif data_augmentation == "off":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if use_val:
        data_train = CIFAR10Noisy(root='./data', train=True, download=True, transform=transform_train, noise_rate=noise, noise_type=noise_type, data_size=data_size)
        
        train_size = int(0.9 * len(data_train))
        val_size = len(data_train) - train_size
        data_train, data_val = random_split(data_train, [train_size, val_size])
        
        data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_dataloader, val_dataloader, test_dataloader, len(data_test.classes)
    else:
        data_train = CIFAR10Noisy(root='./data', train=True, download=True, transform=transform_train, noise_rate=noise, noise_type=noise_type, data_size=data_size)
        data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_dataloader, test_dataloader, len(data_test.classes)