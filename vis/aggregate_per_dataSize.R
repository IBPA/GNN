library(reshape2)

get_files_for_size <- function(base_dir, nMinD, nMaxD, arch, nDataSize){
  files_all <- c()
  for(i in c(nMinD: nMaxD)){
    curr_dir <- sprintf("%s/d_%d/folds/", base_dir, i)
    file_pattern <- sprintf("^%s.*_n%d_f[1-5].csv", arch, nDataSize)
    files <- list.files(curr_dir, pattern = file_pattern, full.names=TRUE)
    files_all <- c(files_all, files)
  }
  
  return (files_all)
}

evaluate_vs_actual <- function(filepath){
  
  # 0) Find "actual" filename
  filename <- basename(filepath)
  dirpath <- dirname(filepath)
  arch <- str_split_fixed(filename, "_n[1-9][0-9]*_f[1-5]", 2)[1]
  suffix <- str_split_fixed(filename,arch, 2)[2]
  filepath_actual <- sprintf("%s/actual%s", dirpath, suffix)
  
  # 1) Read files
  df_pred <- read.csv(file=filepath,header=TRUE, sep = "\t")
  df_actual <- read.csv(file=filepath_actual,header=TRUE, sep = "\t")
  
  # 2) Melt
  df_pred_flat <- melt(df_pred, id.vars=1)
  df_actual_flat <- melt(df_actual, id.vars=1)
  
  if (nrow(df_pred_flat) ==0){
    print("LOOK!!!")
  }
  
  # 3) Combine
  df <- cbind.data.frame(df_pred_flat$variable, df_pred_flat$value, df_actual_flat$value)
  colnames(df) <- c("name", "pred", "actual")
  
  # 4) Evaluate
  summary <- data.frame(prefix=arch, 
                        MSE=mean((df$actual-df$pred)^2), 
                        MAE=mean(abs(df$actual-df$pred)), 
                        PCC=cor(df$actual, df$pred), 
                        stringsAsFactors=FALSE )
  
  return(summary)
}

get_best_prefix <- function(df){
  df_aggr <- aggregate(df[,c("PCC")], list(df$prefix), mean)
  colnames(df_aggr) <- c("prefix", "PCC")
  df_best <- df_aggr[which.max(df_aggr[,c("PCC")]),]
  
  return (df_best$prefix)
}

get_top_performance <- function(base_dir, arch, nMinD, nMaxD, nDataSize){
  # 0) Find all files
  arch_files <- get_files_for_size(base_dir, nMinD, nMaxD, arch, nDataSize)
  
  # 1) Evaluation and combine all evaluations
  df_eval_all <- data.frame()
  for (file in arch_files){
    df_eval <- evaluate_vs_actual(file)
    df_eval$nDataSize <- nDataSize
    df_eval_all <- rbind(df_eval_all, df_eval)
  }
  
  # 2) Find best one
  best_prefix <- get_best_prefix(df_eval_all)
  df_eval_best <- df_eval_all[df_eval_all$prefix == best_prefix,]
  
  return(df_eval_best)
}

nDataSize <- 10
nMinD <- 1
nMaxD <- 3
base_dir <- filename <- file.path(Sys.getenv("HOME"), "/mygithub/gnw/app24_chemotaxis/")
#archs <-  c("grnn_pred", "mlinGrnn_pred", "biRnn_pred_a", "kmlp_pred_a", "rnn_pred_a", "lasso_pred_a")
archs <-  c("grnn_pred",  "biRnn_pred_a", "mlp_pred_a", "rnn_pred_a")

top_perf_all <- data.frame()
for (arch in archs){
  top_perf <- get_top_performance(base_dir, arch, nMinD, nMaxD, nDataSize)
  top_perf_all <- rbind(top_perf_all, top_perf)
  print(sprintf("Summarized arch: %s (%d)", arch, nrow(top_perf)))
}

output_filepath <- sprintf("%s/top_aggr_n%d.csv", base_dir, nDataSize)
write.table(top_perf_all, file=output_filepath, sep=",")

