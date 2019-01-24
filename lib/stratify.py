""" Utility functions for generating stratified datasets in different sizes and folds. """

import os
import sys
import csv
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import KFold
import random
#random.seed(1)

def get_samples(df, nClusters):
    model = AgglomerativeClustering(n_clusters=nClusters)
    model.fit(df)

    labels_idx = {}
    for i in range(0, len(model.labels_)):
        curr_label = model.labels_[i]
        if curr_label in labels_idx:
            labels_idx[curr_label].append(i)
        else:
            labels_idx[curr_label] = [i]

    idx_picked = []
    for key, value in labels_idx.items():
        idx_picked.append(np.random.choice(value))
    return idx_picked

def generate(dir_path):
    strFilename1 = '{!s}/processed_NonTFs.tsv'.format(dir_path)
    strFilename2 = '{!s}/processed_TFs.tsv'.format(dir_path)
    strFilename3 = '{!s}/processed_KO.tsv'.format(dir_path)

    df1 = pd.read_csv(strFilename1, sep="\t")
    df2 = pd.read_csv(strFilename2, sep="\t")
    df3 = pd.read_csv(strFilename3, sep="\t")

    df = pd.concat([df1, df2, df3], axis=1)

    num_samples_min = 10
    num_samples_max = 11
    num_samples_step = 10
    num_folds = 5

    folds_dir_path = '{!s}/folds'.format(dir_path)
    if not os.path.exists(folds_dir_path):
        os.makedirs(folds_dir_path)

    with open('{!s}/stratified_sample_ids.csv'.format(dir_path), 'w', newline='') as ids_csvfile:
            ids_csv_writer = csv.writer(ids_csvfile, delimiter=',')
            for i in range(num_samples_min, num_samples_max, num_samples_step):
                idx_picked = get_samples(df, i)
                ids_csv_writer.writerow(idx_picked)
                print('.', end='', flush=True)
                
                kfold = KFold(n_splits=num_folds)
                f = 0
                for trainIdx, testIdx in kfold.split(idx_picked):
                    f+=1
                    with open('{!s}/n{:d}_f{:d}.txt'.format(folds_dir_path, i, f ), 'w') as fold_txtfile:
                        fold_txtfile.write(','.join(str(x) for x in np.array(idx_picked)[trainIdx]) + "\n")
                        fold_txtfile.write(','.join(str(x) for x in np.array(idx_picked)[testIdx]))
