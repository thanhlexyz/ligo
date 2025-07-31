import lib

def main(args):
    trainer = lib.trainer.create(args)
    trainer.train()
    trainer.save()
