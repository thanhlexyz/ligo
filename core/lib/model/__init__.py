from . import fc

def create(args):
    return eval(args.model).Model()
