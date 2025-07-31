from . import scratch, ligo

def create(args):
    return eval(args.initializer).Initializer(args)
