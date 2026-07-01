import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from .cutout import Cutout
from .tools import DependentLabelGenerator, ClassBalancedBatchSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


class CIFAR100Noisy(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, noise_type='symmetric', target_transform=None, download=False, noise_rate=0.2, data_size=1):
        super(CIFAR100Noisy, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.noise_rate = noise_rate
        self.num_classes = len(self.classes)
        
        self.data, self.targets = self.get_smaller_dataset(data_size)
        
        self.noisy_labels = self.targets.copy()  # Copy the original labels
        
        if noise_type == 'dependent':
            self.noise_label_gen = DependentLabelGenerator(self.num_classes, 32 * 32 * 3, transform) 
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
            self.noisy_labels = get_real_world_cifar("cifar100")

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
                    new_label = (current_label + 1) % self.num_classes
                elif noise_type == 'dependent':
                    new_label = self.noise_label_gen.generate_dependent_labels(self.data[idx], current_label)
                self.noisy_labels[idx] = new_label

    def __getitem__(self, index):
        img, target, flip_label = self.data[index], self.noisy_labels[index], self.flip_labels[index]

        img = Image.fromarray(img)
        
        # Apply the transformations if any
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, flip_label

def get_real_world_cifar(dataset="cifar10"):
    import torch
    if dataset == "cifar10":
        noise_label = torch.load('./data/CIFAR-10_human.pt')
        noisy_label = noise_label['worse_label']
    else:
        noise_label = torch.load('./data/CIFAR-100_human.pt')
        noisy_label = noise_label['noisy_label']
    return noisy_label
 

def get_cifar100(
    batch_size=128,
    num_workers=4,
    noise=0.25,
    noise_type='symmetric',
    data_augmentation="standard", 
    data_size=1,
    use_val=False,
    samples_per_class=None):
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
        data_train = CIFAR100Noisy(root='./data', train=True, download=True, transform=transform_train, noise_rate=noise, noise_type=noise_type, data_size=data_size)
        
        train_size = int(0.9 * len(data_train))
        val_size = len(data_train) - train_size
        data_train, data_val = random_split(data_train, [train_size, val_size])
        
        data_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        
        train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_dataloader, val_dataloader, test_dataloader, len(data_test.classes)
    else:
        if samples_per_class:
            data_train = CIFAR100Noisy(root='./data', train=True, download=True, transform=transform_train, noise_rate=noise, noise_type=noise_type, data_size=data_size)
            train_sampler = ClassBalancedBatchSampler(data_train, batch_size, samples_per_class, drop_last=True)
            train_dataloader = torch.utils.data.DataLoader(data_train, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=True)
            
            data_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
            test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
            
            return train_dataloader, test_dataloader, len(data_test.classes)
        
        data_train = CIFAR100Noisy(root='./data', train=True, download=True, transform=transform_train, noise_rate=noise, noise_type=noise_type, data_size=data_size)
        data_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        
        train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_dataloader, test_dataloader, len(data_test.classes)