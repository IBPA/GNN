# plot app24, (for different dataset sizes)

library(reshape)
library(ggplot2)

color_BiRNN <- "#ffb400"
color_GNN <- "#7fb800"
color_MLP <- "#0d2c54"
color_RNN <- "#f6511d"
color_Lasso <- "grey"
color_LinGNN <- "#00A6ED"

standard_error <- function(x) sd(x)/sqrt(length(x))

base_dir <- filename <- file.path(Sys.getenv("HOME"), "/mygithub/gnw/app24_chemotaxis/")

# 0) Load all
df_all <- data.frame()
files <- list.files(base_dir, pattern = "top_aggr_n.*", full.names = TRUE)
for ( file in files){
  df <- read.csv(file, sep=",", stringsAsFactors = FALSE)
  df_all <- rbind(df_all, df)
}

# 1) aggregate
#df_all$arch <- lapply(df_all$prefix, function(x) str_split_fixed(x, "_pred", 2)[1][1][1])
df_all$arch <- gsub("_pred.*", "", df_all$prefix)
df_all_mean <- aggregate(df_all[,c("MAE", "PCC")], by = list(df_all$arch, df_all$nDataSize), mean)
colnames(df_all_mean) <- c("arch", "nDataSize", "MAE_mean", "PCC_mean")
df_all_sd <- aggregate(df_all[,c("MAE", "PCC")], by = list(df_all$arch, df_all$nDataSize), standard_error)
colnames(df_all_sd) <- c("arch", "nDataSize", "MAE_sd", "PCC_sd")

df <- cbind(df_all_mean, df_all_sd[,c("MAE_sd", "PCC_sd")])
df$arch <- factor(df$arch, levels = c("grnn", "mlinGrnn", "lasso", "mlp","rnn", "biRnn"))


myTheme <- theme(panel.grid.major = element_blank(), 
                 panel.grid.minor = element_blank(),
                 panel.background = element_blank(),
                 legend.box.background = element_blank(),
                 legend.key = element_blank(),
                 aspect.ratio=1,
                 axis.line = element_line(colour = "black"),
                 panel.border = element_rect(fill=NA, size=2),
                 text = element_text(size=34, family="Helvetica"))

pd <- position_dodge(width = 1.2)
gPlot <- ggplot(df, aes(x=nDataSize, y=MAE_mean, ymin=MAE_mean-MAE_sd, ymax=MAE_mean+MAE_sd, colour=arch))+
  geom_errorbar(width=5, size=.7, alpha=0.7, position = pd) +
  geom_point(size=4, position = pd)+
  geom_line(size=1.4, position = pd)+
  scale_y_continuous("Mean Absolute Error\n", breaks = c(0.10, 0.15))+
  scale_x_continuous("\ndataset size", limits = c(8, 101), breaks = c(10, 40, 70, 100))+
  scale_colour_manual(name="Architecture",
                      values=c("grnn" = color_GNN, "mlp" = color_MLP, 
                               "rnn" = color_RNN, "biRnn" = color_BiRNN,
                               "lasso" = color_Lasso, "mlinGrnn" = color_LinGNN))+
  myTheme
print(gPlot)
strFigFilename <- file.path(sprintf("./figure/app24_mae_comparison_syn.pdf") )
ggsave(strFigFilename, gPlot, width=15, height = 10)


y_lim_min_pcc <- min(df$PCC_mean - df$PCC_sd) - 0.01
gPlot <- ggplot(df, aes(x=nDataSize, y=PCC_mean, ymin=PCC_mean-PCC_sd, ymax=PCC_mean+PCC_sd, colour=arch))+
  geom_errorbar(width=5, size=.7, alpha=0.7, position = pd) +
  geom_point(size=4, position = pd)+
  geom_line(size=1.4, position = pd)+
  scale_y_continuous("Pearson Correlation\n", limits=c(y_lim_min_pcc, 1.0), breaks = c(0.4, 0.6,  0.8, 1.0))+
  scale_x_continuous("\ndataset size", limits = c(8, 101), breaks = c(10, 40, 70, 100))+
  scale_colour_manual(name="Architecture",
                      values=c("grnn" = color_GNN, "mlp" = color_MLP, 
                               "rnn" = color_RNN, "biRnn" = color_BiRNN,
                               "lasso" = color_Lasso, "mlinGrnn" = color_LinGNN))+
  myTheme

print(gPlot)
strFigFilename <- file.path(sprintf("./figure/app24_pcc_comparison_syn_n10.pdf") )
ggsave(strFigFilename, gPlot, width=15, height = 10)

gPlot <- ggplot(df, aes(x=arch, y=MAE_mean, fill=arch))+
  scale_y_continuous("Mean Absolute Error\n", breaks = c(0.05, 0.10, 0.15))+
  geom_boxplot(aes(fill=arch), outlier.shape=21)+
  scale_fill_manual(name="Architecture",
                    values=c("grnn" = color_GNN, "mlp" = color_MLP, 
                             "rnn" = color_RNN, "biRnn" = color_BiRNN,
                             "lasso" = color_Lasso, "mlinGrnn" = color_LinGNN))+
  xlab("")+
  myTheme

print(gPlot)
strFigFilename <- file.path(sprintf("./figure/app24_mae_overall_box_syn_n10.pdf") )
ggsave(strFigFilename, gPlot, width=15, height = 10)


# Print Summary
dfMean <- aggregate(df[,c("MAE_mean", "PCC_mean")], list(df$arch), mean)
colnames(dfMean) <- c("arch", "MAE_mean", "PCC_mean")
dfSD <- aggregate(df[,c("MAE_mean", "PCC_mean")], list(df$arch), sd)
colnames(dfSD) <- c("arch", "MAE_sd", "PCC_sd")
dfSummary <- data.frame(dfMean$arch,dfMean$MAE_mean, dfSD$MAE_sd , dfMean$PCC_mean, dfSD$PCC_sd)
options(digits=2)
dfSummary

