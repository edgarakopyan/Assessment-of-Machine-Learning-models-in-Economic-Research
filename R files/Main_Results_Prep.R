library(gt)
library(tidyverse)
library(comprehenr)

# Prepare everything for result prep
dir= commandArgs(trailingOnly=TRUE)

Knobs = c( "2", "3", "4", "7", "8", "11", "12", "16", "25", "26", "32", "56")

round_df <- function(df, digits) {
  nums <- vapply(df, is.numeric, FUN.VALUE = logical(1))
  
  df[,nums] <- round(df[,nums], digits = digits)
  
  (df)
}


# Now create an empty dataset to store the results
final_data <- data.frame(matrix(, nrow=0, ncol=7))

# First add IHDP data
bart <- read.csv(paste(dir, "/Results/ihdp_acic_BART.csv", sep = ""))
models <- read.csv(paste(dir, "/Results/ihdp_acic_models.csv", sep = ""))
models <- models %>% relocate("pehe_in_gan_ihdp", .after = last_col())
models <-models %>% relocate("pehe_out_gan_ihdp", .after = last_col())
models <-models %>% relocate("ate_gan_ihdp", .after = last_col())

intermediary_data <- data.frame(c(bart, models))
final <- intermediary_data
df = data.frame(matrix(, nrow=9, ncol=0))
df = cbind(df, colMeans(final[to_vec(for (i in colnames(final)) if(grepl("pehe_in_", i)) i)]))
df = cbind(df, colMeans(final[to_vec(for (i in colnames(final)) if(grepl("pehe_out_", i)) i)]))
df = cbind(df, append(colMeans(final[to_vec(for (i in colnames(final)) if(grepl("coverage_in_", i)) i)]), NA))
df = cbind(df, append(colMeans(final[to_vec(for (i in colnames(final)) if(grepl("coverage_out_", i)) i)]), NA))
df = cbind(df, append(colMeans(final[to_vec(for (i in colnames(final)) if(grepl("missed_in_", i)) i)]), NA))
df = cbind(df, append(colMeans(final[to_vec(for (i in colnames(final)) if(grepl("missed_out_", i)) i)]), NA))
colnames(df) = c("PEHE in-sample", "PEHE out-sample", "Coverage in-sample", "Coverage out-sample", "Missed in-sample", "Missed out-sample")
rownames(df) = c("BART", "DR Forest", "Sparse Linear DML", "Linear DML", "Causal Forest", "Causal Forest DML", "DR Linear", "DR Sparse Linear", "GANITE")

df = round_df(df, 2)
df$Knob <- paste( toupper("ihdp"))
df$Model <- rownames(df)
final_data <- rbind(final_data, df)

for (i in Knobs_additional){
  
  bart <- read.csv(paste(dir , "/Results/", i, "_acic_BART.csv", sep = ""))
  models <- read.csv(paste(dir, "/Results/", i, "_acic_models.csv", sep = ""))
  models <- models %>% relocate("pehe_in_gan_acic", .after = last_col())
  models <- models %>% relocate("pehe_out_gan_acic", .after = last_col())
  models <- models %>% relocate("ate_gan_acic", .after = last_col())
  
  intermediary_data <- data.frame(c(bart, models))
  final <- intermediary_data
  df = data.frame(matrix(, nrow=9, ncol=0))
  df = cbind(df, colMeans(final[to_vec(for (i in colnames(final)) if(grepl("pehe_in_", i)) i)]))
  df = cbind(df, colMeans(final[to_vec(for (i in colnames(final)) if(grepl("pehe_out_", i)) i)]))
  df = cbind(df, append(colMeans(final[to_vec(for (i in colnames(final)) if(grepl("coverage_in_", i)) i)]), NA))
  df = cbind(df, append(colMeans(final[to_vec(for (i in colnames(final)) if(grepl("coverage_out_", i)) i)]), NA))
  df = cbind(df, append(colMeans(final[to_vec(for (i in colnames(final)) if(grepl("missed_in_", i)) i)]), NA))
  df = cbind(df, append(colMeans(final[to_vec(for (i in colnames(final)) if(grepl("missed_out_", i)) i)]), NA))
  colnames(df) = c("PEHE in-sample", "PEHE out-sample", "Coverage in-sample", "Coverage out-sample", "Missed in-sample", "Missed out-sample")
  rownames(df) = c("BART", "DR Forest", "Sparse Linear DML", "Linear DML", "Causal Forest", "Causal Forest DML", "DR Linear", "DR Sparse Linear", "GANITE")
  
  df = round_df(df, 2)
  if (i == "ihdp"){
    df$Knob <- paste( toupper(i))
  } else {
    df$Knob <- paste( "Knob",i)
  }
  df$Model <- rownames(df)
  final_data <- rbind(final_data, df)
}

# Add table constants 

n = 0
# c_rn = 63
c_save = TRUE
c_format = "html"


gt_table <- final_data %>%
  # head(c_rn) %>%
  filter(Knob %in% unique(final_data$Knob)) %>%
  gt(
    groupname_col = "Knob",
    rowname_col = "Model"
  )


gt_table <- gt_table %>% 
  cols_align(
    align = "right",
    columns = c("Model",  "PEHE in-sample", "PEHE out-sample", "Coverage in-sample",  "Coverage out-sample", "Missed in-sample", "Missed out-sample")
  ) %>% 
  opt_row_striping()

gt_table <- gt_table %>%
  tab_header(
    title = "Causal Machine Learning ITE Estimation",
    subtitle = "Based on Precision in Estimating Heterogenous Effects, Coverage and Missed Rate"
  ) %>%
  tab_source_note(
    source_note = md("Source: data from the Infant Health and Development Program and Atlantic Causal Inference Conference 2016")
  )

# Build spanner column

gt_table <- gt_table %>%
  tab_spanner(
    label =  "In Sample",
    columns = c("PEHE in-sample", "Coverage in-sample","Missed in-sample")
  )

gt_table <- gt_table %>%
  tab_spanner(
    label =  "Out Sample",
    columns = c("PEHE out-sample", "Coverage out-sample","Missed out-sample")
  )



c_col = c("#1e3048", "#274060", "#2f5375", "#4073a0", "#5088b9")
c_col_light_blue = c("#edf2fb", "#e2eafc", "#d7e3fc", "#ccdbfd", "#c1d3fe")
c_container_width = px(800)
c_table_width = px(650)

gt_table <- gt_table %>%
  tab_options(
    table.font.name = "TimesNewRoman",
    table.font.color = c_col[2],
    table.border.top.style = "solid",
    table.border.top.color = c_col[2],
    table.border.top.width = px(3),
    table.border.bottom.style = "solid",
    table.border.bottom.color = c_col[2],
    table.border.bottom.width = px(3),
    column_labels.border.top.color = "white",
    column_labels.border.top.width = px(3),
    column_labels.border.bottom.color = c_col[2],
    column_labels.border.bottom.width = px(3),
    
    row_group.border.bottom.width = px(3),
    row_group.border.bottom.style = "solid", 
    row_group.border.bottom.color = c_col[2],
    row_group.border.top.width = px(3),
    row_group.border.top.color = c_col[2],
    row_group.border.top.style = "solid",
    
    heading.border.bottom.style = "solid",
    heading.border.bottom.color = c_col[2],
    heading.border.bottom.width = px(3),
    
  ) %>% 
  tab_style(
    style = list(
      cell_text(
        size = px(30),
        weight = "normal",
        align = "left",
        font = "Bloomsbury"
      )
    ),
    locations = list(
      cells_title(groups = "title")
    )
  ) %>% 
  tab_style(
    style = list(
      cell_text(
        size = px(18),
        align = "left"
      )
    ),
    locations = list(
      cells_title(groups = "subtitle")
    )
  ) %>% 
  tab_style(
    style = list(
      cell_text(
        size = px(18)
      ),
      cell_borders(
        sides = c("bottom", "top"),
        color = c_col[1],
        weight = px(1)
      )
    ),
    locations = list(
      cells_body(gt::everything())
    )
  ) %>% 
  tab_style(
    style = list( 
      cell_text(
        size = px(14),
        color = "#2f5375",
        font = "Bloomsbury"
      )
    ),
    locations = list(
      cells_column_labels(everything())
    )
  ) %>% 
  tab_style(
    style = list( 
      cell_text(
        size = px(16),
        color = "#2f5375",
        font = "Bloomsbury"
      ),
      cell_borders(
        sides = c("bottom", "right"),
        style = "solid",
        color = "white",
        weight = px(1)
      )
    ),
    locations = list(
      cells_stub(gt::everything()),
      cells_stubhead()
    )
  ) %>% 
  tab_style(
    style = list(
      cell_text(
        font = "Bloomsbury", size = px(16), 
        color = "#2f5375")
    ),
    location = list(
      cells_body(columns = c("Model"))
    )
  )



gt_table <- gt_table %>% 
  tab_footnote(
    footnote= "Only PEHE is calculated for GANITE",
    locations = cells_stub(rows = c(9, 18, 27, 36, 45, 54, 63,72, 81,90,99))
    
  )


gt_table <- gt_table %>% 
  cols_label(
    "PEHE in-sample" = "PEHE",
    "Coverage in-sample" = "Coverage",
    "Missed in-sample" = "Missed",
    "PEHE out-sample" = "PEHE",
    "Coverage out-sample" = "Coverage",
    "Missed out-sample" = "Missed"
  )

gtsave(gt_table, filename =  "final_table.tex", path = paste(dir, "/Graphs_and_Tables/"))
