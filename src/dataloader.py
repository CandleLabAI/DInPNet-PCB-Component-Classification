# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Data preprocessing and loader for training and testing
"""

import os
from torchvision import datasets, transforms
import torch
import config

def image_common_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    """
    Helper function for RGB normalization
    """
    preprocess = image_preprocess_transforms()
    common_transforms = transforms.Compose([
        preprocess,
        transforms.Normalize(mean, std)
    ])
    return common_transforms

def data_augmentation_preprocess():
    """
    Helper function for applying augmentations
    """
    data_augmented_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20, fill=(0,0,0)),
    ])
    return data_augmented_transforms

def data_loader(data_root, collate, transform, batch_size=16, shuffle=False, num_workers=2):
    """
    Helper function to return object of pytorch dataloader
    """
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         collate_fn = collate,
                                         num_workers=num_workers,
                                         shuffle=shuffle)
    return loader

def image_preprocess_transforms():
    """
    Helper function for resizing and normalizing image
    """
    preprocess = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor()
        ])

    return preprocess

def get_mean_std(data_root):
    """
    Helper function for finding RGB Mean and STD values.
    """
    transform = image_preprocess_transforms()
    loader = data_loader(data_root, transform = transform, collate = None)
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    print(f'mean: {mean}, std: {std}')
    return mean, std

def get_data(batch_size, data_root, num_workers=4):
    """
    Create train and test dataloaders
    """
    train_data_path = os.path.join(data_root, 'Train')
    mean, std = get_mean_std(data_root=train_data_path)

    def collate(batch):

        trans = data_augmentation_preprocess()
        train_transforms = transforms.Compose([
            trans,
            transforms.Resize(config.IMAGE_SIZE),
            # this re-scales image tensor values between 0-1. image_tensor /= 255
            transforms.ToTensor(),
            # subtract mean and divide by variance.
            transforms.Normalize(mean, std)
        ])

        imgs, labels = zip(*batch)
        imgs = [train_transforms(img) for img in imgs]
        labels = [torch.tensor(label) for label in labels]
        return torch.stack(imgs), torch.stack(labels)

    common_transforms = image_common_transforms(mean, std)

    # train dataloader
    train_loader = data_loader(train_data_path,
                               collate = collate,
                               transform = None,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers
                               )

    # test dataloader
    test_data_path = os.path.join(data_root, 'Test')
    test_loader = data_loader(test_data_path,
                              collate = None,
                              transform = common_transforms,
                              batch_size=1,
                              shuffle=False,
                              num_workers=num_workers,
                              )

    return train_loader, test_loader
