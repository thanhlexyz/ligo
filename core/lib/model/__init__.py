from . import fc, fc2

def create(args):
    return eval(args.model).Model().to(args.device)
