library(tidyverse)
library(censReg)

raw_df <- read_csv("raw_df.csv", progress = F)

df = raw_df %>% 
  groupby("num_sets", "sketch_estimator", "scenario") %>% 
  summarize(re_mean = mean(relative_error_1), 
            re_std = std(relative_error_1)) %>%
  ungroup() %>% 
  mutate(re_std_log = log(re_std), 
         re_std_log_cens = pmin(re_std_log, 0), 
         universe_size = 1000000 * as.numeric(scenario), 
         log10_universe_size = log10(universe_size), 
         sketch_size = as.numeric(gsub("k_.*", "", sketch_estimator)), 
         log10_sketch_size = sketch_size, 
         flip_prob = as.numeric(gsub(".*_", "", sketch_estimator))) 

df = df.query('num_sets % 5 == 0')
df["num_sets"] = df["num_sets"].astype(int)