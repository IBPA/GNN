#!/usr/bin/env Rscript
# Gene Regulatory Network Inference using GENIE3
library("GENIE3")
library("foreach")
library("fBasics")
library("argparse")

parser <- ArgumentParser(description='Gene Network Inference using GENIE3')
parser$add_argument("-i", "--input", type="character", dest="input", default="dataset.csv", 
                    help="Gene expression dataset",
                    metavar="dataset.csv")

parser$add_argument("-o", "--output", type="character", dest="output", default="net_GENIE3.tsv",
                    help="Gene regulatory network inferred by GENIE3",
                    metavar="net_GENIE3.tsv")

args <- parser$parse_args()

df <- read.csv(args$input, header = TRUE)
df$KO <- NULL # remove KO column
ge_mat <- data.matrix(t(df))
net_weights <- GENIE3(ge_mat, nCores=2, verbose=TRUE)
net_edges <- getLinkList(net_weights, reportMax=(ncol(net_weights)*2))

write.table(net_edges, args$output, row.names=FALSE, quote=FALSE, sep="\t")
sprintf("save inferred net into '%s'", args$output)