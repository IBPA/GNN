import csv

def load_ge_std_dic(filepath, genes=[]):
    ge_std_dic = {}
    with open(filepath, "r") as f_in:
        reader = csv.reader(f_in, delimiter = ",")
        next(reader) # skip header

        for row in reader:
            gname = row[0]
            if (len(genes) > 0 and gname in genes) or (len(genes) == 0):
                gstd = float(row[1])
                ge_std_dic[gname] = gstd
    return ge_std_dic
    