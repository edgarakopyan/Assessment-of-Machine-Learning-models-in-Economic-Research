# PLot and save ATE RMSE 

# Plot RMSE
library(Metrics)
library(ggplot2)
library(gridExtra)


# RMSE for IHDP 

dir= commandArgs(trailingOnly=TRUE)

ihdp_bart <- read.csv(paste(dir, "/Results", "/ihdp_acic_BART.csv", sep = ""))
ihdp_models <- read.csv(paste(dir, "/Results", "/ihdp_acic_models.csv", sep = ""))

final_ihdp <- data.frame(c(ihdp_bart, ihdp_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_ihdp)){
  if (grepl("ate_", i)){
    square_diffs = (final_ihdp$ates - final_ihdp[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_ihdp")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot1 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for IHDP") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")


# RMSE for Knob 8 


acic_bart <- read.csv(paste(dir, "/Results", "/8_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/8_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot2 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 8") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")


# Put on the same graph 

file = grid.arrange(plot1, plot2, ncol=2)

ggsave(filename = "ATE_main.png", file, path = paste(dir, "/Graphs_and_Tables/", sep = ""))
















# RMSE for Knob 2 


acic_bart <- read.csv(paste(dir, "/Results", "/2_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/2_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot3 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 2") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")



# RMSE for Knob 3 


acic_bart <- read.csv(paste(dir, "/Results", "/3_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/3_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot4 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 3") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")


# RMSE for Knob 4 


acic_bart <- read.csv(paste(dir, "/Results", "/4_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/4_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot5 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 4") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")


# RMSE for Knob 7


acic_bart <- read.csv(paste(dir, "/Results", "/7_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/7_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot6 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 7") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")


# RMSE for Knob 11


acic_bart <- read.csv(paste(dir, "/Results", "/11_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/11_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot7 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 11") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")


# RMSE for Knob 12 


acic_bart <- read.csv(paste(dir, "/Results", "/12_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/12_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot8 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 12") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")




# Put on the same graph 

file = grid.arrange(plot3, plot4, plot5,plot6,plot7,plot8,ncol=2)

ggsave(filename = "ATE_Knobs2-12.png", file, path = paste(dir, "/Graphs_and_Tables/", sep = ""))

















# RMSE for Knob 16 


acic_bart <- read.csv(paste(dir, "/Results", "/16_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/16_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot1 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 16") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")



# RMSE for Knob 25 


acic_bart <- read.csv(paste(dir, "/Results", "/25_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/25_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot2 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 25") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")


# RMSE for Knob 26


acic_bart <- read.csv(paste(dir, "/Results", "/26_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/26_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot3 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 26") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")


# RMSE for Knob 32


acic_bart <- read.csv(paste(dir, "/Results", "/32_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/32_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot4 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 32") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")


# RMSE for Knob 56


acic_bart <- read.csv(paste(dir, "/Results", "/56_acic_BART.csv", sep = ""))
acic_models <- read.csv(paste(dir, "/Results", "/56_acic_models.csv", sep = ""))

final_acic <- data.frame(c(acic_bart, acic_models))


ATE <- data.frame(stringsAsFactors=FALSE)
for (i in colnames(final_acic)){
  if (grepl("ate_", i)){
    square_diffs = (final_acic$ates - final_acic[i])^2
    ATE <- rbind(ATE, c(toupper(qdapRegex::ex_between(i, "ate_", "_acic")[[1]]), sqrt(as.numeric(colMeans(square_diffs)))))
  }
}
colnames(ATE) = c("Models", "RMSE")
ATE$RMSE <- as.double(ATE$RMSE)


plot5 <- ggplot(ATE, aes(x=Models, y=RMSE)) +
  geom_segment( aes(x=Models, xend=Models, y=0, yend=RMSE), colour= "red") + 
  geom_point(colour = "firebrick4") + 
  theme_linedraw() +
  ggtitle("Average Treatment Effect", subtitle = "Root Mean Squared Error for Knob 56") + 
  theme(axis.text.x = element_text(angle = 35, size=12, vjust = 0.5)) + 
  theme(axis.text.y = element_text(size=12))+
  theme(plot.title = element_text(size = 12, face = "bold"))+
  theme(axis.title = element_text(size = 12, face = "bold"))+
  xlab("") + 
  ylab("")

file = grid.arrange(plot1, plot2, plot3,plot4,plot5,ncol=2)


ggsave(filename = "ATE_Knobs16-56.png", file, path = paste(dir, "/Graphs_and_Tables/", sep = ""))

