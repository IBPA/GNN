#!/usr/bin/env python3

# Run Command: ./script_name target_dir
# Description: randomize gene names inside the GE profiles to simulate randomized TRNs
import sys, os, random
import pandas as pd

class GNameRandomizer:
    def __init__(self, filename):
        with open(filename) as fh:
            line = fh.readline()
            self.header = line.rstrip()
            self.names_orig = self.header.split("\t")
            self.names_new = self.names_orig.copy()

            # shuffle
            random.seed(0.5)
            random.shuffle(self.names_new)

            # save correspondance
            self.dic_orig_to_new = {}
            for i in range(0, len(self.names_orig)):
                self.dic_orig_to_new[self.names_orig[i]] = self.names_new[i]

    def reorder_header(self, filename):
        # Read
        lines = []
        with open(filename) as fh:
            lines.extend(fh.readlines())

        # Replace
        lines[0] = "{!s}\n".format('\t'.join(self.names_new))

        # Write
        with open(filename, mode="w") as fh:
            fh.writelines(lines)
        
        print("saved: {!s}".format(filename))

    def reorder_ge_range(self, filename):
        df = pd.read_csv(filename, sep=',')

        for orig_name, new_name in self.dic_orig_to_new.items():
            idx = df.index[df['gene'] == orig_name]
            df.at[idx[0], 'gene'] = new_name

        df.to_csv(filename, sep=',', index=False)
        print("saved: {!s}".format(filename))

target_dir = sys.argv[1] if len(sys.argv) > 1 else '{!s}/mygithub/GRNN_Clean/data/dream5_ecoli/expr2_random_TRNs//modules_aggr/size-10/top_edges-1_gnw_data/'.format(os.environ['HOME'])

nonTFs_filename = '{!s}/d_1/processed_NonTFs.tsv'.format(target_dir)
ko_filename = '{!s}/d_1/processed_KO.tsv'.format(target_dir)
ge_range_filename = '{!s}/ge_range.csv'.format(target_dir)

randomizer = GNameRandomizer(nonTFs_filename)
randomizer.reorder_header(nonTFs_filename)
randomizer.reorder_header(ko_filename)
randomizer.reorder_ge_range(ge_range_filename)
