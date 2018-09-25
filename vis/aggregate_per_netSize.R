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

strDir <- file.path(args[1])
#strDir <- "/home/eetemadi/mygithub/grnnPipeline/data/dream5/modules_gnw_a/size-10/"
dirList <- paste0(list.dirs(path = strDir, full.names = TRUE, recursive = FALSE), "/d_1/folds/")
dataSizeList <- c(10)
#dataSizeList <- c(10)
#prefixList <- c("grnn_pred", "biRnn_pred_a", "kmlp_pred_a", "rnn_pred_a", "avg_pred_a")
prefixList <- c("grnn_pred")

df_best_mean <- data.frame(prefix=character(), nDataSize=numeric(), MSE=numeric(), MAE=numeric(), PCC=numeric(), stringsAsFactors=FALSE )
for(nDataSize in dataSizeList){
   for(strPrefix in prefixList){
      prefixes_uniq <- getUniqPrefixes(dirList[1], strPrefix, nDataSize)

      strNetSizeDir <- dirList[1]
      df <- data.frame(prefix=character(), MSE=numeric(), MAE=numeric(), PCC=numeric(), stringsAsFactors=FALSE )
      for(strNetSizeDir in dirList){
         for (strPrefix in prefixes_uniq) {
               print( paste0(strNetSizeDir,"|", strPrefix, "|", nDataSize))
               summary <- getDfDirPrefixSize(strNetSizeDir, strPrefix, nDataSize)
               df <- rbind(df, summary)
         }
      }

      df_aggr <- aggregate(df[,c("MSE", "MAE", "PCC" )], list(df$prefix), mean)
      colnames(df_aggr) <- c("prefix", "MSE", "MAE", "PCC")
      df_curr_best <- df_aggr[which.max(df_aggr[,c("PCC")]),]
      df_curr_best$nDataSize <- nDataSize
      df_best_mean <- rbind(df_best_mean, df_curr_best)
   }
}

file_path <- file.path(strDir, "aggr.csv")
write.table(df_best_mean, file=file_path, sep=",")
print(sprintf("saved to %s", file_path))
