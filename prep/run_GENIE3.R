#!/usr/bin/env Rscript
# Gene Regulatory Network Inference using GENIE3
library("GENIE3")
library("foreach")
library("fBasics")

getSampleData <- function(){
  exprMatr <- matrix(sample(1:10, 100, replace=TRUE), nrow=20)
  rownames(exprMatr) <- paste("Gene", 1:20, sep="")
  colnames(exprMatr) <- paste("Sample", 1:5, sep="")
  return( list("matrix" = exprMatr, "tfs" = c(rownames(exprMatr))) )
}

getActualData <- function(base_dir){
  df <- read.csv(file.path(base_dir, "data.tsv"), sep = "\t", header = FALSE)
  gene_names_df <- read.csv(file.path(base_dir, "gene_names.tsv"), sep = "\t", header = FALSE)
  rownames(df) <- gene_names_df$V1
  
  tf_names_df <-  read.csv(file.path(base_dir, "tf_names.tsv"), sep = "\t", header = FALSE)
  exprMatr <- data.matrix(df)
  return( list("matrix" = exprMatr, "tfs" = c(tf_names_df$V1)) )
}

args=commandArgs(TRUE)

if (length(args)==0) {
  base_dir <- file.path(Sys.getenv("HOME"), "/mygithub/GRNN_Clean/data/dream5_ecoli/s2/")
} else {
  base_dir <- file.path(args[1])
}
set.seed(123)

#dataObj <- getSampleData()
#weightMat <- GENIE3(dataObj$matrix, regulators=dataObj$tfs, verbose=TRUE)
dataObj <- getActualData(base_dir)
weightMat <- GENIE3(dataObj$matrix, regulators=dataObj$tfs, nCores=24, verbose=TRUE)
linkList <- getLinkList(weightMat, reportMax=(ncol(weightMat)*2))
write.table(linkList, file.path(base_dir, "edges_inferred.tsv"), sep="\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
print(paste0("Saved to:", file.path(base_dir, "edges_inferred.tsv")))

