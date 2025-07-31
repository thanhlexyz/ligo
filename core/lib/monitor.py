import pandas as pd
import numpy as np
import torch
import tqdm
import os

def get_label(args):
    return f'{args.initializer}'

def create(args):
    return Monitor(args)

class Monitor:

    def __init__(self, args):
        # save
        self.args = args
        # initialize progress bar
        T = args.n_epoch
        self.bar = tqdm.tqdm(range(T))
        # initialize writer
        self.csv_data = {}
        self.global_step = 0

    def __update_time(self):
        self.bar.update(1)

    def __update_description(self, **kwargs):
        _kwargs = {}
        for key in kwargs:
            if 'acc' in key:
                _kwargs[key] = f'{kwargs[key]:0.3f}'
            elif 'loss' in key:
                _kwargs[key] = f'{kwargs[key]:0.3f}'
            elif 'epoch' in key:
                _kwargs[key] = f'{kwargs[key]:d}'
        self.bar.set_postfix(**_kwargs)

    def __display(self):
        self.bar.display()
        print()

    def step(self, info):
        # update progress bar
        self.__update_time()
        self.__update_description(**info)
        self.__display()
        # log to csv
        self.__update_csv(info)
        self.global_step += 1

    @property
    def label(self):
        return get_label(self.args)

    def __update_csv(self, info):
        for key in info.keys():
            if key not in self.csv_data:
                self.csv_data[key] = [float(info[key])]
            else:
                self.csv_data[key].append(float(info[key]))

    def export_csv(self):
        args = self.args
        path = os.path.join(args.csv_dir, f'{self.label}.csv')
        df = pd.DataFrame(self.csv_data)
        df.to_csv(path, index=None)
