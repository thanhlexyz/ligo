import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import lib
import os

def create(args):
    return Trainer(args)

class Trainer:

    def __init__(self, args):
        # save args
        self.args = args
        self.monitor = lib.monitor.create(args)
        self.train_loader, self.test_loader = lib.loader.create(args)
        self.model = lib.model.create(args)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        # weights transfer
        initializer = lib.initializer.create(args)
        pretrain_model = self.load_pretrain()
        initializer.init(pretrain_model, self.model, self.train_loader)

    def train_epoch(self):
        loader, model, criterion, optimizer, args = self.train_loader, self.model, self.criterion, self.optimizer, self.args
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        info = {'train_loss': np.mean(losses)}
        return info

    def test_epoch(self):
        loader, model, args = self.test_loader, self.model, self.args
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(args.device)
                y = y.to(args.device)
                y_hat = model(x)
                _, y_pred = torch.max(y_hat.data, 1)
                total += y_pred.size(0)
                correct += (y_pred == y).sum().item()
        info = {'test_acc': 100 * correct / total}
        return info

    def train(self):
        args, monitor = self.args, self.monitor
        for epoch in range(args.n_epoch):
            info = {'epoch': epoch}
            info.update(self.train_epoch())
            info.update(self.test_epoch())
            monitor.step(info)
            monitor.export_csv()

    def save(self):
        args, model, monitor = self.args, self.model, self.monitor
        path = os.path.join(args.checkpoint_dir, f'{monitor.label}.pth')
        print('[+] saving model')
        torch.save(self.model.to('cpu'), path)
        print(f'    - saved to {path}')

    def load_pretrain(self):
        args, monitor = self.args, self.monitor
        path = os.path.join(args.checkpoint_dir, f'{args.pretrain_model}.pth')
        print('[+] loading pretrain model')
        if os.path.exists(path):
            model = torch.load(path, map_location=args.device, weights_only=False)
            print(f'    - loaded from {path}')
            return model
