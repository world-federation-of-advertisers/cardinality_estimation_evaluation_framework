suppressPackageStartupMessages({
  library(tidyverse)
  library(censReg)
  library(GGally)
})

raw_df <- read_csv("raw_df.csv", progress = F)

df <- raw_df %>% 
  group_by(num_sets, sketch_estimator, scenario) %>% 
  summarize(re_mean = mean(relative_error_1), 
            re_std = sd(relative_error_1)) %>%
  ungroup() %>% 
  mutate(re_std_log = log(re_std), 
         re_std_log_cens = pmin(re_std_log, 0), 
         universe_size = 1e6 * as.numeric(scenario), 
         log10_universe_size = log10(universe_size), 
         sketch_size = 1e3 * as.numeric(gsub("k_.*", "", sketch_estimator)), 
         log10_sketch_size = log10(sketch_size), 
         flip_prob = as.numeric(gsub(".*_", "", sketch_estimator))) %>%
  select(-sketch_estimator, -scenario)

## remove censored data points
df1 <- df %>% 
  filter(num_sets %% 4 == 0, re_std_log_cens != 0) %>%
  select(re_std_log, flip_prob, log10_sketch_size, num_sets, log10_universe_size)

## use all 1st-order term
fit0 <- lm(re_std_log ~ flip_prob 
           + log10_sketch_size 
           + num_sets 
           + log10_universe_size, 
           data = df1)
summary(fit0)
plot(fit0) ## all diagnostic plots LGTM

# ## remove universe size
# fit1 <- lm(re_std_log ~ flip_prob + log10_sketch_size + num_sets, 
#            data = df1)
# summary(fit1)
# c(AIC(fit1), BIC(fit1))
# # plot(fit1)

# ## drop universe_size, add flip_prob:log10_sketch_size
# fit2 <- lm(re_std_log ~ flip_prob*log10_sketch_size*num_sets, data = df1)
# summary(fit2)
# c(AIC(fit2), BIC(fit2))

# ## try something complex
# fit3 <- lm(re_std_log ~ flip_prob + log10_sketch_size + num_sets 
#            + flip_prob:num_sets 
#            + flip_prob:log10_sketch_size:num_sets
#            , data = df1)
# summary(fit3)
# c(AIC(fit3), BIC(fit3))

## censored regression
fit4 <- censReg(re_std_log_cens ~ flip_prob 
                + log10_sketch_size 
                + num_sets
                + log10_universe_size
                # + flip_prob:num_sets 
                # + flip_prob:log10_sketch_size 
                # + log10_sketch_size:num_sets
                # + flip_prob:log10_sketch_size:num_sets
                , data = df, left = -Inf, right = 0)
summary(fit4, logSigma = F)
## censored reg model result is somewhat close to fit0