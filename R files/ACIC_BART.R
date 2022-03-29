library(bartCause)
library(matrixStats)
library(reticulate)

# BART in ACIC Datasets

args= commandArgs(trailingOnly=TRUE)

dir = args[1]
knob = args[2]


acic_x <- read.csv(paste(dir, "/ACIC/x.csv", sep = ""))

dir_data <- paste(dir, "/ACIC/", knob, sep = "")
files <- list.files(path=dir_data, pattern="*.csv", full.names=TRUE, recursive=FALSE)

pehe_out_bart_acic = c()
pehe_in_bart_acic = c() 
coverage_in_bart_acic = c()
missed_in_bart_acic = c()
coverage_out_bart_acic = c()
missed_out_bart_acic = c()
ate_bart_acic = c()

my_data = list()
j = 0
for (i in files) {
  j = j +1
  acic_df <- data.frame(read.csv(i))
  my_data[[j]] <- acic_df
}
p = 0
for (i in my_data) {
  p = p + 1
  print(cbind("Working on", p))
  #out sample pehe
  acic_df <- i
  acic_df_train <- acic_df[c(0:4000), ]
  acic_df_test <- acic_df[c(4001:4802),]
  cate <- acic_df_test['mu1'] - acic_df_test['mu0']
  y <-  acic_df_train[, 'y']
  w <- acic_df_train[, 'z']
  x <- acic_x[c(0:4000),]
  x_testing <- acic_x[c(4001:4802),]
  
  bart_estimate <-bartc(y, w, x, method.rsp = "bart", method.trt = "bart", keepTrees = TRUE, seed = 123)
  predictions <- data.frame(predict(bart_estimate, x_testing, type = "icate"))
  posterior_standard_deviations <- colSds(data.matrix(predictions))
  data <- data.frame(colSums(predictions))/5000
  pehe_out_bart_acic <- append(pehe_out_bart_acic, sqrt(sum((cate - data)^2)/802))
  
  ci_low = stack(lapply(predictions, quantile, prob = 0.025, names = FALSE))[,1]
  ci_high = stack(lapply(predictions, quantile, prob = 0.975, names = FALSE))[,1]
  
  # Coverage 
  coverage_out_bart_acic <- append(coverage_out_bart_acic, mean((ci_low < cate) & (cate < ci_high)))
  
  # Missed
  missed_out_bart_acic <- append(missed_out_bart_acic, mean((ci_low < 0) & (0 < ci_high)))
  
  # in sample pehe
  cate <- acic_df_train['mu1'] - acic_df_train['mu0']
  
  ate_bart_acic <- append(ate_bart_acic, as.numeric(summary(bart_estimate)$estimate[1]))
  
  predictions <- data.frame(predict(bart_estimate, x, type = "icate"))
  posterior_standard_deviations <- colSds(data.matrix(predictions))
  data <- data.frame(colSums(predictions))/5000
  pehe_in_bart_acic <- append(pehe_in_bart_acic, sqrt(sum((cate - data)^2)/4802))
  
  ci_low = stack(lapply(predictions, quantile, prob = 0.025, names = FALSE))[,1]
  ci_high = stack(lapply(predictions, quantile, prob = 0.975, names = FALSE))[,1]
  
  # Coverage 
  coverage_in_bart_acic <- append(coverage_in_bart_acic, mean((ci_low < cate) & (cate < ci_high)))
  
  # Missed
  missed_in_bart_acic <- append(missed_in_bart_acic, mean((ci_low < 0) & (0 < ci_high)))
}


final_data_acic <- data.frame( pehe_in_bart_acic, coverage_in_bart_acic, missed_in_bart_acic, pehe_out_bart_acic, coverage_out_bart_acic, missed_out_bart_acic, ate_bart_acic)

write.csv(final_data_acic, paste(dir, "/Results/", knob, "_acic_BART.csv", sep = ""), row.names = FALSE)


