from copy import copy
import os
import numpy as np
import pandas as pd

from tqdm import tqdm

from prog_utils import *


def main_split(args):
    """
    Split the data into one segment per training instance
    """
    data = pd.read_pickle(args.data_path)
    splitted = []
    print("Before split, {} instances".format(data.shape[0]))
    for i in tqdm(range(data.shape[0])):
        row = data.loc[i]
        features = row.FEATURES[:args.obs_len]
        for j in range(len(row.PROG)):
            splitted.append([row.SEQUENCE, features, row.CANDIDATE_CENTERLINES, [row.PROG[j]]])
            xy = get_xy(features)
            cl, n_step, v = row.PROG[j]
            next_seg = exec_prog(xy, row.CANDIDATE_CENTERLINES[cl], n_step, v)
            for p in next_seg:
                new_step = copy(features[-1])
                new_step[0] = str(float(new_step[0]) + 0.1)
                new_step[3:5] = p
                features = np.append(features, np.expand_dims(new_step, 0), axis=0)

    result = pd.concat([pd.DataFrame([entry], columns=data.columns) for entry in splitted], ignore_index=True)

    print("After split, {} instances".format(result.shape[0]))
    ss = args.data_path[:-4]
    new_path = ss+'_split.pkl'
    result.to_pickle(new_path)

def main():
    args = parse_arguments()
    if args.mode == 'split':
        main_split(args)
    else:
        main(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
