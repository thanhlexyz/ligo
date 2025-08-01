import torch.optim as optim
import torch.nn as nn
import torch

from . import util

class LiGO(nn.Module):

    def __init__(self, L1, L2, W1, B1, W20, B20):
        super().__init__()
        w = nn.Parameter(torch.rand(L2, L1) / L1)
        self.register_parameter('w', w)
        
        A_weight, B_weight = util.get_weight_width_expansion_matrices(W1, W20)
        B_bias = util.get_bias_width_expansion_matrices(B1, B20)
        
        for i, a in enumerate(A_weight):
            self.register_parameter(f'a_w_{i}', a)
        for i, b in enumerate(B_weight):
            self.register_parameter(f'b_w_{i}', b)
        for i, b in enumerate(B_bias):
            self.register_parameter(f'b_b_{i}', b)
            
    def print_trainable_parameters(self):
        for name, p in self.named_parameters():
            print(name, p.mean().item())

    def forward(self, x, W1, B1, W20, B20):
        # extract args
        L2, L1 = self.w.shape
        w = getattr(self, 'w')
        # width expansion
        W1_, B1_ = [], []
        for l1 in range(L1):
            # extract linear transformation
            a_w, b_w, b_b = getattr(self, f'a_w_{l1}'), getattr(self, f'b_w_{l1}'), getattr(self, f'b_b_{l1}')
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
                w1_, b1_ = W1_[l1], B1_[l1]
                if w1_.shape == w20.shape:
                    # print(f'Weight: {l1=}->{l2=}, shapes match: {w1_.shape}')
                    w2 += w[l2, l1] * w1_
                if b1_.shape == b20.shape:
                    # print(f'Bias: {l1=}->{l2=}, shapes match: {b1_.shape}')
                    b2 += w[l2, l1] * b1_
            W2.append(w2); B2.append(b2)
            
            
        x = x.reshape(x.shape[0], -1)
        for l in range(L2-1):
            x = torch.nn.functional.relu(x @ W2[l].T + B2[l])
        x = x @ W2[L2-1].T + B2[L2-1]
        return x, W2, B2

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
        ligo_model = LiGO(L1, L2, W1, B1, W20, B20).to(args.device)
        # step 3:
        # optimize expansion matrices
        print('[+] optimize LiGO weight transfer matrices')
        print(f'    - {util.get_model_size(ligo_model)=}')
        criterion = nn.MSELoss()
        optimizer = optim.Adam(ligo_model.parameters(), lr=args.lr)
        # for name, p in ligo_model.named_parameters():
            # print(name, p.shape)
        for epoch in range(20):
            for i, (x, _) in enumerate(loader):
                optimizer.zero_grad()
                x = x.to(args.device)
                x_hat, W2, B2 = ligo_model(x, W1, B1, W20, B20)
                y_hat = model1(x).detach()
                loss = criterion(x_hat, y_hat)
                loss.backward()
                # print(ligo_model.w.grad)
                optimizer.step()
                # ligo_model.print_trainable_parameters()
                # if i > 2:
                #     break
            print(f'{epoch=} {loss.item()=:0.6f}')
        util.encode(W2, B2, model2)
        return model2
