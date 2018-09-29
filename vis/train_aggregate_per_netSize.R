#!/usr/bin/Rscript
library(reshape)
source("../vis/lib_modelEval.R")

getUniqPrefixes <- function(strDir, prefix, nSize){
  fnames <- list.files(strDir, pattern = sprintf("^%s.*_n%d_f[0-9]*.csv", prefix, nSize),
                       full.names=FALSE)
  prefixes <- lapply(fnames, function(x) unlist(strsplit(x, "_n"))[1])
  return (unique(prefixes))
}

args=(commandArgs(TRUE))

#strDir <- file.path(args[1])
strDir <- file.path(Sys.getenv("HOME"), "/mygithub/GRNN_Clean/data/dream5_ecoli/expr1/modules_aggr/")
best_archs_filename <- file.path(Sys.getenv("HOME"), "/mygithub/GRNN_Clean/data/dream5_ecoli/expr1/modules_aggr/aggr_all.csv.hpc1")
df <- read.csv(file=best_archs_filename, header=TRUE, sep = ",")
nDataSize=10

df_all <- data.frame(netsize=numeric(), prefix=character(), MSE=numeric(), MAE=numeric(), PCC=numeric(), stringsAsFactors=FALSE )
for (i in 1:nrow(df)){
   c_row=df[i,]
   strNetSizeDir=paste0(strDir, "size-", c_row$netsize)
   dirList <- paste0(list.dirs(path = strNetSizeDir, full.names = TRUE, recursive = FALSE), "/d_1/folds/")

   for(strCurrDir in dirList){
#      summary <- getDfDirPrefixSize(strCurrDir, c_row$prefix, nDataSize, "train_")
      summary <- getDfDirPrefixSize(strCurrDir, c_row$prefix, nDataSize, "train_" )
      if (nrow(summary)<1){
         print(sprintf("****** %s ****** %s", strCurrDir, c_row$prefix))
      } else{
         summary$netsize <- c_row$netsize
         df_all <- rbind(df_all, summary)
      }
   }
}

df_aggr <- aggregate(df_all[,c("MSE", "MAE", "PCC" )], by = list(df_all$netsize, df_all$prefix), mean)
colnames(df_aggr) <- c("netsize", "prefix", "MSE", "MAE", "PCC")
file_path <- file.path(strDir, "train_aggr_all.csv")
write.table(df_aggr, file=file_path, sep=",")
print(sprintf("saved to %s", file_path))
