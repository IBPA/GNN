import os
import sys
import shutil
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
from sklearn import preprocessing
import graph_util as gu
import impact_module

def get_min_max_scaled(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns)
    return df

def save_mr_ge(df, mr_list, output_dir):
    df_mr = df[mr_list]
    #df_mr=(df_mr - df_mr.min())/(df_mr.max() - df_mr.min())
    #df_mr = get_min_max_scaled(df_mr)

    filepath = "{!s}/processed_TFs.tsv".format(output_dir)
    df_mr.to_csv(filepath, sep="\t", index=False)
    print("saved: {!s}".format(filepath))
    
def save_nmr_ge(df, mr_list, output_dir):
    df_mr = df[mr_list]
    #df_mr=(df_mr - df_mr.min())/(df_mr.max() - df_mr.min())
    #df_mr = get_min_max_scaled(df_mr)

    filepath = "{!s}/processed_NonTFs.tsv".format(output_dir)
    df_mr.to_csv(filepath, sep="\t", index=False)
    print("saved: {!s}".format(filepath))

def get_ko_vector(gt_str, nmr_list):
    ko_vector = np.ones(len(nmr_list))
    
    if type(gt_str) is str and len(gt_str)>0 :
        gt_str_parts = gt_str.split("/")

        for gname in gt_str_parts:
            if gname in nmr_list:
                gidx = nmr_list.index(gname)
                ko_vector[gidx] = 0
            
    return ko_vector

def save_ko(df, nmr_list, output_dir):
    df_gt = df[['Genotype']]
    
    ko_vectors = []
    for index, row in df_gt.iterrows():
        curr_vector = get_ko_vector(row[0], nmr_list)
        ko_vectors.append(curr_vector)
    
    ko_vectors = np.array(ko_vectors)

    filepath = "{!s}/processed_KO.tsv".format(output_dir)
    np.savetxt(filepath, ko_vectors, fmt='%1.1f', delimiter='\t', header = '\t'.join(nmr_list), comments='',)
    print("saved: {!s}".format(filepath))

def generate(gen_filenames, df, net_file):
    # 1) get mr, nmr from .tsv graph
    impact_m = impact_module.OutWImpact(gen_filenames['mr_score'])
    graph = gu.DirGraphReal(net_file, impact_m)
    mr_list = graph.get_mr()
    nmr_list = graph.get_nmr()

    # 2) save data
    output_dir = "{!s}_gnw_data/d_1/".format(net_file[:-4])
    os.makedirs(output_dir, exist_ok=True)

    save_mr_ge(df, mr_list, output_dir)
    save_nmr_ge(df, nmr_list, output_dir)
    save_ko(df, nmr_list, output_dir)

    # 3) save .dep
    dep_filepath = "{!s}.dep".format(net_file[:-4])
    graph.save_as_linear_dep_graph(mr_list, dep_filepath)
    shutil.copy2(dep_filepath, "{!s}_gnw_data/net.dep".format(net_file[:-4]))

    return output_dir