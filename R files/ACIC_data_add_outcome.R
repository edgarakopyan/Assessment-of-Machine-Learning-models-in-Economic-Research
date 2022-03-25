library(aciccomp2016)
main_dir= commandArgs(trailingOnly=TRUE)
for (j in c(1:77)){
  dir <- paste(main_dir, "/data_cf_all/", j, sep="")
  files <- list.files(path=dir, pattern="*.csv", full.names=TRUE, recursive=FALSE)
  for (i in c(1:100)){
    ys <- data.frame(dgp_2016(input_2016, j, i))$y
    df <- read.csv(files[i]) 
    df$y <- ys
    write.csv(df, paste(paste(main_dir, "/ACIC/",j, "/", sep = ""),i, ".csv",  sep = ""), row.names = FALSE)
  }
}
