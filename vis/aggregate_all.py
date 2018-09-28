#!/usr/bin/env python3

import os, sys
import pandas as pd

sizes = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
base_dir = sys.argv[1] if len(sys.argv) > 1 else '{!s}/mygithub/GRNN_Clean/data/dream5_ecoli/expr1/modules_aggr/'.format(os.environ['HOME'])

df=pd.DataFrame(columns=["prefix","MSE","MAE","PCC","nDataSize"])
for size in sizes:
    filename='{!s}/size-{:d}/aggr.csv'.format(base_dir, size)
    df_curr=pd.read_csv(filename, sep=',', index_col=0)
    df_curr['netsize']=size

    df = pd.concat([df, df_curr], axis=0, ignore_index=True)

filename = "{!s}/aggr_all.csv".format(base_dir)
df.to_csv(filename)
print("saved to {!s}".format(filename))

