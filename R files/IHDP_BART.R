library(bartCause)
library(matrixStats)
library(reticulate)
library(rstudioapi)

dir= commandArgs(trailingOnly=TRUE)

np <- import("numpy")
npz_train <- np$load(paste(dir, "/ihdp_npci_1-100.train.npz", sep = ""))

x_train <- npz_train$f[['x']]
y_train <- npz_train$f[['yf']]
w_train <- npz_train$f[['t']]
mu0_train <- npz_train$f[['mu0']]
mu1_train <- npz_train$f[['mu1']]

npz_test <- np$load(paste(dir, "/ihdp_npci_1-100.test.npz", sep = ""))

x_test <- npz_test$f[['x']]
y_test <- npz_test$f[['yf']]
w_test <- npz_test$f[['t']]
mu0_test <- npz_test$f[['mu0']]
mu1_test <- npz_test$f[['mu1']]


cate_train <- mu1_train - mu0_train
cate_test <- mu1_test - mu0_test

pehe_out_bart_ihdp = c()
pehe_in_bart_ihdp = c() 
coverage_in_bart_ihdp = c()
missed_in_bart_ihdp = c()
coverage_out_bart_ihdp = c()
missed_out_bart_ihdp = c()
ate_bart_ihdp = c()

for (i in 1:100) {
  
  #out sample pehe
  
  cate <- cate_test[, i]
  y <-  y_train[, i]
  w <- w_train[, i]
  x <- x_train[, , i]
  x_testing <- x_test[, , i]
  
  bart_estimate <-bartc(y, w, x, method.rsp = "bart", method.trt = "bart", keepTrees = TRUE, seed = 123)
  predictions <- data.frame(predict(bart_estimate, x_testing, type = "icate"))
  posterior_standard_deviations <- colSds(data.matrix(predictions))
  data <- data.frame(colSums(predictions))/5000
  pehe_out_bart_ihdp <- append(pehe_out_bart_ihdp, sqrt(sum((cate - data)^2)/75))
  
  ci_low = stack(lapply(predictions, quantile, prob = 0.025, names = FALSE))[,1]
  ci_high = stack(lapply(predictions, quantile, prob = 0.975, names = FALSE))[,1]
  
  # Coverage 
  coverage_out_bart_ihdp <- append(coverage_out_bart_ihdp, mean((ci_low < cate) & (cate < ci_high)))
  
  # Missed
  missed_out_bart_ihdp <- append(missed_out_bart_ihdp, mean((ci_low < 0) & (0 < ci_high)))
  
  # ate estimate
  ate_bart_ihdp <- append(ate_bart_ihdp, as.numeric(summary(bart_estimate)$estimate[1]))
  
  # in sample pehe
  cate <- cate_train[, i]
  predictions <- data.frame(predict(bart_estimate, x, type = "icate"))
  posterior_standard_deviations <- colSds(data.matrix(predictions))
  data <- data.frame(colSums(predictions))/5000
  pehe_in_bart_ihdp <- append(pehe_in_bart_ihdp, sqrt(sum((cate - data)^2)/747))
  
  ci_low = stack(lapply(predictions, quantile, prob = 0.025, names = FALSE))[,1]
  ci_high = stack(lapply(predictions, quantile, prob = 0.975, names = FALSE))[,1]
  
  # Coverage 
  coverage_in_bart_ihdp <- append(coverage_in_bart_ihdp, mean((ci_low < cate) & (cate < ci_high)))
  
  # Missed
  missed_in_bart_ihdp <- append(missed_in_bart_ihdp, mean((ci_low < 0) & (0 < ci_high)))
}


final_data_ihdp <- data.frame(pehe_in_bart_ihdp, coverage_in_bart_ihdp, missed_in_bart_ihdp,pehe_out_bart_ihdp, coverage_out_bart_ihdp, missed_out_bart_ihdp, ate_bart_ihdp)

write.csv(final_data_ihdp,paste(dir, "/Results/ihdp_acic_BART.csv", sep = ""), row.names = FALSE)
