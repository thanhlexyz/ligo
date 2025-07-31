import torch.optim as optim
import torch.nn as nn

class Initializer:

    def __init__(self, args):
        pass

    def init(self, pretrain_model, model, loader):
        # step 1:
        # construct trainable depth expansion matrix
        # step 2:
        # construct trainable width expansion matrix
        # step 3:
        # optimize expansion matrices
        print('[+] optimize LiGO weight transfer matrices')
        criterion = nn.MSELoss()
        optimizer = optim.Adam(..., lr=args.lr)
        for i, (x, _) in enumerate(loader):
            optimizer.zero_grad()
            y_hat_pretrain = pretrain_model(x) # may be need to modify here
            y_hat = model(x)
            loss = criterion(y_hat_pretrain, y_hat)
            loss.backward()
            optimizer.step()
            print(f'    - {i=} {loss.item()=:0.6f}')
        # step 4:
        # copy weight to model for return
        return model
