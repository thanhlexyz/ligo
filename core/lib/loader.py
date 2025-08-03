from torch.utils.data import DataLoader
from torchvision import datasets

import torchvision.transforms as transforms
import os

def create(args):
    # Transform to apply to the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])
    # Load the FashionMNIST dataset
    train_dataset = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    test_dataset = datasets.FashionMNIST(root=args.data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())
    return train_loader, test_loader
