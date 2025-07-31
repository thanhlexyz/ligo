import torch.optim as optim
import torch.nn as nn
import torch

from . import util

class LiGO(nn.Module):

    def __init__(self, w, A_weight, B_weight, B_bias):
        super().__init__()
        self.w = w
        self.A_weight = A_weight
        self.B_weight = B_weight
        self.B_bias = B_bias
        for i, a in enumerate(A_weight):
            self.register_parameter(f'a_weight_{i}', a)
        for i, b in enumerate(B_weight):
            self.register_parameter(f'b_weight_{i}', b)
        for i, b in enumerate(B_bias):
            self.register_parameter(f'b_bias_{i}', b)

    def forward(self, W1, B1):
        # extract args
        L2, L1 = self.w.shape
        w, A_weight, B_weight, B_bias = self.w, self.A_weight, self.B_weight, self.B_bias
        # width expansion
        W1_, B1_ = [], []
        for l in range(L1):
            # extract linear transformation
            a_w, b_w, b_b = A_weight[l], B_weight[l], B_bias[l]
            # extract input
            w1, b1 = W1[l], B1[l]
            # weight width expansion
            w1_ = b_w @ w1 @ a_w.T
            # bias width expansion
            b1_ = b_b @ b1
            # print(f'{l=} {w1.shape=} {w1_.shape=} {b1.shape=} {b1_.shape=}')
            # store
            W1_.append(w1_); B1_.append(b1_)
        # depth expansion
        W2, B2 = [], []
        for l in range(L2):
            pass
        return W2, B2

class Initializer:

    def __init__(self, args):
        self.args = args

    def init(self, model1, model2, loader):
        # extract args
        args = self.args
        # model1 (small) -> model2 (large)
        # step 1: extract pretrain
        W1, B1 = util.decode(model1)
        W2, B2 = util.decode(model2)
        L1, L2 = len(W1), len(W2)
        # step 2: construct trainable depth expansion matrix
        w = util.get_depth_expansion_matrix(L1, L2)
        A_weight, B_weight = util.get_weight_width_expansion_matrices(W1, W2)
        B_bias = util.get_bias_width_expansion_matrices(B1, B2)
        ligo_model = LiGO(w, A_weight, B_weight, B_bias).to(args.device)
        # step 3:
        # optimize expansion matrices
        print('[+] optimize LiGO weight transfer matrices')
        criterion = nn.MSELoss()
        optimizer = optim.Adam(ligo_model.parameters(), lr=args.lr)
        for i, (x, _) in enumerate(loader):
            optimizer.zero_grad()
            W2, B2 = ligo_model(W1, B1)
            util.encode(W2, B2, model2)
            loss = criterion(model2(x), model1(x))
            loss.backward()
            optimizer.step()
            print(f'    - {i=} {loss.item()=:0.6f}')
        return model2
