
class Initializer:

    def __init__(self, args):
        pass

    def init(self, pretrain_model, model):
        #
        print('[+] pretrain_model:')
        for name, p in pretrain_model.named_parameters():
            print(f'{name=} {p.shape=}')
        print('[+] model:')
        for name, p in model.named_parameters():
            print(f'{name=} {p.shape=}')
        exit()
        return model
