""" Utility functions for reading and writing data """

import pandas as pd
import numpy as np
import os

def readFoldsIdx(fold_filepath):
    separator = ','
    with open(fold_filepath, 'r') as bf:
        idx_train = list(map(int,
                         bf.readline().rstrip().split(separator)))
        idx_test = list(map(int,
                        bf.readline().rstrip().split(separator)))
    
    return idx_train, idx_test


def readCsvRows(filename, idx_rows):
    df_all = pd.read_csv(filename, sep="\t")
    df_selected = df_all.iloc[idx_rows]

    return df_selected.sort_index().reset_index(drop=True)
    
def loadData(data_filepaths, fold_filepath):
    idx_train, idx_test = readFoldsIdx(fold_filepath)

    df_train_nonTFs = readCsvRows(data_filepaths['NonTFs'], idx_train)
    df_train_TFs = readCsvRows(data_filepaths['TFs'], idx_train)
    df_train_KOs = readCsvRows(data_filepaths['KOs'], idx_train)

    df_test_nonTFs = readCsvRows(data_filepaths['NonTFs'], idx_test)
    df_test_TFs = readCsvRows(data_filepaths['TFs'], idx_test)
    df_test_KOs = readCsvRows(data_filepaths['KOs'], idx_test)

    return pd.concat([df_train_TFs, df_train_KOs], axis=1), df_train_nonTFs,\
           pd.concat([df_test_TFs, df_test_KOs], axis=1), df_test_nonTFs

def savePreds(df_preds, dir_path, fold_filename, layers_config, alpha, filePrefix):
    file_prefix = '{:s}_pred_a={:.4f}_'.format(filePrefix, alpha) + 'H'.join(str(x) for x in np.array(layers_config))
    pred_filepath = '{!s}/folds/{!s}_{!s}.csv'.format(dir_path, file_prefix, fold_filename[:-4])
    df_preds.to_csv(pred_filepath, sep='\t', index=False)
    print("saved: {!s}".format(pred_filepath))

def get_ge_range(filename):
    if not os.path.exists(filename):
        print("Note: file \'{!s}\'does not exist (this maybe fine)".format(filename))
        return None
        
    print("apply range using: {!s}".format(filename))
    df = pd.read_csv(filename)
    range_dic = {}
    for __, row in df.iterrows():
        range_dic[row['gene']] = [row['min'], row['max']]

    return range_dic
