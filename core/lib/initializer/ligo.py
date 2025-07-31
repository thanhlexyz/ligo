import torch.optim as optim
import torch.nn as nn
import torch

from . import util

class LiGO(nn.Module):

    def __init__(self, w, A, B):
        super().__init__()
        self.w = w
        self.A = A
        self.B = B
        self.L2, self.L1 = w.shape
        for i, a in enumerate(A):
            self.register_parameter(f'a_{i}', a)
        for i, b in enumerate(B):
            self.register_parameter(f'b_{i}', b)

    def forward(self, W1):
        # extract args
        L2, L1 = self.w.shape
        w, A, B = self.w, self.A, self.B
        # depth expansion
        W1_ = []
        return W2

class Initializer:

    def __init__(self, args):
        pass

    def init(self, model1, model2, loader):
        # model1 (small) -> model2 (large)
        # step 1: extract pretrain
        W1, L1 = util.decode(model1)
        W2, L2 = util.decode(model2)
        # step 2: construct trainable depth expansion matrix
        w = util.get_depth_expansion_matrix(L1, L2)
        A, B = util.get_width_expansion_matrices(W1, W2, L1, L2)
        ligo_model = LiGO(w, A, B)
        exit()
        # step 3:
        # optimize expansion matrices
        print('[+] optimize LiGO weight transfer matrices')
        criterion = nn.MSELoss()
        optimizer = optim.Adam(ligo_model, lr=args.lr)
        for i, (x, _) in enumerate(loader):
            optimizer.zero_grad()
            W2 = ligo_model(W1)
            util.encode(W2, model2)
            loss = criterion(model2(x), model1(x))
            loss.backward()
            optimizer.step()
            print(f'    - {i=} {loss.item()=:0.6f}')
        return model2
