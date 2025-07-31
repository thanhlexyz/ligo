from . import scratch

def create(args):
    return eval(args.initializer).Initializer(args)
