from torch.utils.data import DataLoader
from torchvision import datasets

import torchvision.transforms as transforms

def create(args):
    # Transform to apply to the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])
    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader
