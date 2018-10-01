library(ggplot2)
library(stringr)
library(splines)
library(scales)
library(dplyr)

color_BiRNN <- "#ffb400"
color_GNN <- "#7fb800"
color_MLP <- "#0d2c54"
color_RNN <- "#f6511d"
color_Lasso <- "grey"
color_LinGNN <- "#00A6ED"

filename <- file.path(Sys.getenv("HOME"), "/mygithub/GRNN_Clean/data/dream5_ecoli/expr1/modules_aggr/train_aggr_all.csv")
df <- read.csv(filename, sep = ",")
df$arch <- str_split_fixed(df$prefix, "_", 3)[,2]
df <- df[df$arch != "avg",]
df$arch <- factor(df$arch, levels = c("grnn", "mlinGrnn", "lasso", "kmlp","rnn", "biRnn"))


myTheme <- theme(panel.grid.major = element_blank(), 
                 panel.grid.minor = element_blank(),
                 panel.background = element_blank(),
                 legend.box.background = element_blank(),
                 #legend.position =  c(0.6,0.1),
                 legend.key = element_blank(),
                 aspect.ratio=1,
                 axis.line = element_line(colour = "black"),
                 panel.border = element_rect(fill=NA, size=2),
                 text = element_text(size=34, family="Helvetica"))

gPlot <- ggplot(df, aes(x=netsize, y=MAE, colour=arch))+
                  geom_point(size=4)+
                  geom_line(size=1.4)+
                  scale_y_continuous("Mean Absolute Error\n")+
                  scale_x_continuous("\nnetwork size", limits = c(10, 1000), breaks = c(10, 200, 400, 600, 800, 1000))+
                  scale_colour_manual(name="Architecture",
                                     values=c("grnn" = color_GNN, "kmlp" = color_MLP, 
                                             "rnn" = color_RNN, "biRnn" = color_BiRNN,
                                             "lasso" = color_Lasso, "mlinGrnn" = color_LinGNN))+
                  myTheme
                
print(gPlot)
strFigFilename <- file.path(sprintf("./figure/low_res_mae_comparison_dream5_ecoli_expr1_n10_train.pdf") )
ggsave(strFigFilename, gPlot, width=15, height = 10)

y_lim_min_pcc <- min(df$PCC) - 0.01
gPlot <- ggplot(df, aes(x=netsize, y=PCC, colour=arch))+
  geom_point(size=4)+
  geom_line(size=1.4)+
  scale_y_continuous("Pearson Correlation\n", limits=c(y_lim_min_pcc, .95), breaks = c(.6,  0.7, 0.8, 0.9))+
  scale_x_continuous("\nnetwork size", limits = c(10, 1000), breaks = c(10, 200, 400, 600, 800, 1000))+
  scale_colour_manual(name="Architecture",
                      values=c("grnn" = color_GNN, "kmlp" = color_MLP, 
                               "rnn" = color_RNN, "biRnn" = color_BiRNN,
                               "lasso" = color_Lasso, "mlinGrnn" = color_LinGNN))+
  myTheme

strFigFilename <- file.path(sprintf("./figure/low_res_pcc_comparison_dream5_ecoli_expr1_n10_train.pdf") )
ggsave(strFigFilename, gPlot, width=15, height = 10)
print(gPlot)

gPlot <- ggplot(df, aes(x=arch, y=MAE, fill=arch))+
  geom_boxplot(aes(fill=arch), outlier.shape=21)+
  scale_fill_manual(name="Architecture",
                    values=c("grnn" = color_GNN, "kmlp" = color_MLP, 
                             "rnn" = color_RNN, "biRnn" = color_BiRNN,
                             "lasso" = color_Lasso, "mlinGrnn" = color_LinGNN))+
  ylab("Mean Absolute Error\n")+
  xlab("")+
  myTheme

print(gPlot)
strFigFilename <- file.path(sprintf("./figure/low_res_mae_overall_box_dream5_ecoli_expr1_n10_train.pdf") )
ggsave(strFigFilename, gPlot, width=15, height = 10)


# Print Summary
dfMean <- aggregate(df[,c("MAE", "PCC")], list(df$arch), mean)
colnames(dfMean) <- c("arch", "MAE_mean", "PCC_mean")
dfSD <- aggregate(df[,c("MAE", "PCC")], list(df$arch), sd)
colnames(dfSD) <- c("arch", "MAE_sd", "PCC_sd")
dfSummary <- data.frame(dfMean$arch,dfMean$MAE_mean, dfSD$MAE_sd , dfMean$PCC_mean, dfSD$PCC_sd)
options(digits=2)
dfSummary

