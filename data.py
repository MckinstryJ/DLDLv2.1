import os
import csv
import math
import torch
import torch.nn.functional as F
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torch.utils.data import dataloader, ConcatDataset


class Data:
    def __init__(self, args, trans=None):
        self.args = args
        transform_list = [
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        transform = transforms.Compose(transform_list)
        self.train_dataset = Dataset(args, transform)

        if trans:
            for t in trans:
                transform_list = [
                    transforms.Resize((args.height, args.width)),
                    transforms.RandomHorizontalFlip(p=0.4),
                    transforms.GaussianBlur(kernel_size=(5, 7), sigma=(0.1, 2.0)),
                    transforms.RandomAdjustSharpness(sharpness_factor=2),
                    transforms.RandomAutocontrast(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
                transform = transforms.Compose(transform_list)
                ds_ = Dataset(args, transform)
                self.train_dataset = ConcatDataset([self.train_dataset, ds_])

        self.train_loader = dataloader.DataLoader(self.train_dataset,
                                                  shuffle=True,
                                                  batch_size=args.train_batch_size,
                                                  num_workers=args.nThread)


class Dataset(dataset.Dataset):
    def __init__(self, args, transform):
        self.root = args.train_img
        self.transform = transform
        self.labels = args.labels
        self.loader = default_loader

    def __getitem__(self, index):
        name, age = self.labels[index]
        if self.root:
            img = self.loader(os.path.join(self.root, name))
        else:
            img = self.loader(name)
        age = int(age)
        label = [normal_sampling(age, i) for i in range(101)]
        label = torch.Tensor(label)
        label = F.normalize(label, p=1, dim=0)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, age

    def __len__(self):
        return len(self.labels)


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)
