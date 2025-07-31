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

    def forward(self, W1, B1, W20, B20):
        # extract args
        L2, L1 = self.w.shape
        w, A_weight, B_weight, B_bias = self.w, self.A_weight, self.B_weight, self.B_bias
        # width expansion
        W1_, B1_ = [], []
        for l1 in range(L1):
            # extract linear transformation
            a_w, b_w, b_b = A_weight[l1], B_weight[l1], B_bias[l1]
            # extract input
            w1, b1 = W1[l1], B1[l1]
            # weight width expansion
            w1_ = b_w @ w1 @ a_w.T
            # bias width expansion
            b1_ = b_b @ b1
            # store
            W1_.append(w1_); B1_.append(b1_)
        # depth expansion
        W2, B2 = [], []
        for l2 in range(L2):
            # fuse W1_ -> W2, B1_ -> B2
            w20, b20 = W20[l2], B20[l2]
            w2, b2 = torch.zeros_like(w20), torch.zeros_like(b20)
            for l1 in range(L1):
                w1_, b1_ = W1_[l1], B1[l1]
                if w1_.shape == w20.shape:
                    w2 += w[l2, l1] * w1_[l1]
                if b1_.shape == b20.shape:
                    b2 += w[l2, l1] * b1_[l1]
            W2.append(w2); B2.append(b2)
        return W2, B2

class Initializer:

    def __init__(self, args):
        self.args = args

    def init(self, model1, model2, loader):
        # extract args
        args = self.args
        model1 = model1.to(args.device)
        model2 = model2.to(args.device)
        # model1 (small) -> model2 (large)
        # step 1: extract pretrain
        W1, B1 = util.decode(model1)
        W20, B20 = util.decode(model2)
        L1, L2 = len(W1), len(W20)
        # step 2: construct trainable depth expansion matrix
        w = util.get_depth_expansion_matrix(L1, L2)
        A_weight, B_weight = util.get_weight_width_expansion_matrices(W1, W20)
        B_bias = util.get_bias_width_expansion_matrices(B1, B20)
        ligo_model = LiGO(w, A_weight, B_weight, B_bias).to(args.device)
        # step 3:
        # optimize expansion matrices
        print('[+] optimize LiGO weight transfer matrices')
        print(f'    - {util.get_model_size(ligo_model)=}')
        criterion = nn.MSELoss()
        optimizer = optim.Adam(ligo_model.parameters(), lr=args.lr)
        for i, (x, _) in enumerate(loader):
            optimizer.zero_grad()
            W2, B2 = ligo_model(W1, B1, W20, B20)
            util.encode(W2, B2, model2)
            x = x.to(args.device)
            loss = criterion(model2(x), model1(x))
            loss.backward()
            optimizer.step()
            print(f'    - {i=} {loss.item()=:0.6f}')
            print(ligo_model.w)
            print()
        exit()
        return model2
