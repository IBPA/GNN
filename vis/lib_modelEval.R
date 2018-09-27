
getDfDirPrefixSize <- function(strDir, prefix, nSize){
  print(sprintf("%s_n%d_f[0-9]*.csv", prefix, nSize))
  
  filenames <- list.files(strDir, pattern = sprintf("^%s_n%d_f[0-9]*.csv", prefix, nSize),
                          full.names=TRUE)
  fileInfo <- file.info(filenames)
  if( nrow(fileInfo[fileInfo$size>0,]) < 5){
     return (data.frame())
  }

  datalist <- lapply(filenames, function(x){read.csv(file=x,header=TRUE, sep = "\t")})
  df_pred <- Reduce(function(x,y) {rbind(x,y)}, datalist)
  df_pred_flat <- melt(df_pred)
  
  if (nrow(df_pred_flat) ==0){
    print("LOOK!!!")
  }
  
  filenames <- list.files(strDir, pattern = sprintf("^actual_n%d_f[0-9]*.csv", nSize),
                          full.names=TRUE)
  datalist <- lapply(filenames, function(x){read.csv(file=x,header=TRUE, sep = "\t")})
  df_actual <- Reduce(function(x,y) {rbind(x,y)}, datalist)
  df_actual_flat <- melt(df_actual)
  print(nrow(df_pred_flat))
  print(nrow(df_actual_flat))

  df <- cbind.data.frame(df_pred_flat$variable, df_pred_flat$value, df_actual_flat$value)
  colnames(df) <- c("name", "pred", "actual")
  
  summary <- data.frame(prefix=prefix, 
                        nSize=nSize, 
                        MSE=mean((df$actual-df$pred)^2), 
                        MAE=mean(abs(df$actual-df$pred)), 
                        PCC=cor(df$actual, df$pred), 
                        stringsAsFactors=FALSE )
  return (summary)
}

getDfDirPrefix <- function(strDir, strPrefix){
  dfOne <- data.frame(prefix=character(), nSize=integer(), MSE=numeric(), MAE=numeric(), PCC=numeric(), stringsAsFactors=FALSE )
  for (nSize in seq(10, 100, 10)){
    dfOne[nrow(dfOne) + 1,] <- getDfDirPrefixSize(strDir, strPrefix, nSize)
  }
  
  return (dfOne)
}


getDfSummary <- function(df, groupBy){
  dfMean <- aggregate(df[,c("MSE","MAE","PCC" )], list(df[,c(groupBy)]), mean)
  colnames(dfMean) <- c(groupBy, "mean_MSE","mean_MAE","mean_PCC" )
  dfSd <- aggregate(df[,c("MSE","MAE","PCC" )], list(df[,c(groupBy)]), sd)
  colnames(dfSd) <- c(groupBy, "sd_MSE","sd_MAE","sd_PCC" )
  
  dfSummary <- cbind(dfMean[,c(groupBy)], dfMean[,c(2:4)], dfSd[,c(2:4)])
  colnames(dfSummary)[1] <- groupBy
  dfSummary$prefix <- df$prefix[1]
  return(dfSummary)
}

