suppressPackageStartupMessages({
  library(tidyverse)
  library(AER)
  library(GGally)
  library(xtable)
})

raw_df <- read_csv("raw_df.csv", progress = F)

## censor threshold
censor = .5

df <- raw_df %>% 
  group_by(num_sets, sketch_estimator, scenario) %>% 
  summarize(re_mean = mean(relative_error_1), 
            re_std = sd(relative_error_1)) %>%
  ungroup() %>% 
  mutate(re_std_log = log(re_std), 
         re_std_log_cens = pmin(re_std_log, log(censor)), 
         universe_size = 1e6 * as.numeric(scenario), 
         log_universe_size = log(universe_size), 
         sketch_size = 1e3 * as.numeric(gsub("k_.*", "", sketch_estimator)), 
         log_sketch_size = log(sketch_size), 
         flip_prob = as.numeric(gsub(".*_", "", sketch_estimator))) %>%
  select(-sketch_estimator, -scenario)

## remove censored data points
df1 <- df %>% 
  filter(num_sets %% 4 == 0, re_std_log_cens != log(censor)) %>%
  select(re_std_log, flip_prob, log_sketch_size, num_sets, log_universe_size)

## use all 1st-order term ----
fit0 <- lm(re_std_log ~ flip_prob 
           + log_sketch_size 
           # + num_sets
           + log_universe_size
           # + flip_prob:log_sketch_size
           + flip_prob:num_sets ##
           # + flip_prob:log_universe_size
           # + log_sketch_size:num_sets
           # + log_sketch_size:log_universe_size
           # + num_sets:log_universe_size
           # + flip_prob:log_sketch_size:num_sets
           , data = df1)
summary(fit0)
# f <- step(fit0)
# summary(f)
c(AIC(fit0), BIC(fit0))
plot(fit0, which = c(1,2)) ## all diagnostic plots LGTM
plot(fit0, which = 2) ## all diagnostic plots LGTM

# # ## remove universe size
# fit1 <- lm(re_std_log ~ flip_prob + log_sketch_size + flip_prob:num_sets,
#            data = df1)
# # summary(fit1)
# # plot(fit1)
# data.frame(residual = residuals(fit1), 
#            universe_size = exp(df1$log_universe_size)) %>% 
#   ggplot(aes(factor(universe_size), residual)) +
#   geom_boxplot() +
#   labs(x = "universe size") + 
#   theme_bw()


## censored regression ---- 
fit4 <- tobit(re_std_log_cens ~ flip_prob 
              + log_sketch_size 
              # + num_sets
              + log_universe_size
              # + flip_prob:log_sketch_size
              + flip_prob:num_sets ##
              # + flip_prob:log_universe_size
              # + log_sketch_size:num_sets
              # + log_sketch_size:log_universe_size
              # + num_sets:log_universe_size
              # + flip_prob:log_sketch_size:num_sets
              , data = df, left = -Inf, right = log(censor))
summary(fit4)
c(AIC(fit4), BIC(fit4))
## censored reg model result is somewhat close to fit0


## R^2 (it does not make sense to calculate R^2 for censored data)
#' @param model a `tobit` object.
R.squared <- function(model, 
                      exclude_censored = TRUE, 
                      adjusted = FALSE) {
  y_hat <- fitted(model) ## fitted value
  y = model$y[,1] ## observed value
  p = model$df - 1
  n = nrow(model$y) - 1
  if (exclude_censored) {
    status = model$y[,2] ## 0 for censored, 1 for kept. 
    y_hat = y_hat[!!status]
    y = y[!!status]
    n = sum(status) - 1
  }
  SSTot <- sum((y - mean(y))^2)
  SSRes <- sum((y_hat - y)^2)
  R2 = 1 - SSRes / SSTot 
  if (adjusted) {
    return(1 - (1 - R2) * (n - 1) / (n - p - 1))
  } else 
    return(R2)
}
R.squared(fit4, exclude_censored = T, adjusted = F)
R.squared(fit4, exclude_censored = T, adjusted = T)


## Diagnostic plots ---- 
## (https://rpubs.com/therimalaya/43190)
## Residual plot
data.frame(cencored = df$re_std_log_cens == log(censor), 
           residuals = residuals(fit4), 
           fitted = fitted(fit4)) %>% 
  filter(!cencored) %>%
  ggplot(aes(fitted, residuals)) + 
  geom_point(color = "grey40", alpha = .5) +
  stat_smooth(method = "loess") +
  geom_hline(yintercept = 0, size = .7, col = "red", linetype = 2) +
  geom_vline(xintercept = log(0.35), size = .7, col = "grey50") +
  labs(x = "Fitted values", y = "Residuals", 
       title = "Residual vs Fitted Plot") + 
  theme_bw() + 
  ggsave("5_residuals.pdf", width = 7, height = 5)

## Q-Q plot
data.frame(cencored = df$re_std_log_cens == log(censor), 
           residuals = residuals(fit4)) %>%
  mutate(stdresid = residuals / sd(residuals)) %>% 
  filter(!cencored) %>% 
  ggplot(aes(sample = stdresid)) + 
  stat_qq(color = "grey40", alpha = .8, shape = 1) + 
  stat_qq_line(color = "red", linetype = 2) + 
  labs(x = "Theoretical Quantiles", y = "Standardized Residuals", 
       title = "Normal Q-Q") + 
  theme_bw() + 
  ggsave("5_qqplot.pdf", width = 6, height = 5)

## Scale-Location
data.frame(cencored = df$re_std_log_cens == log(censor), 
           residuals = residuals(fit4), 
           fitted = fitted(fit4)) %>%
  mutate(stdresid = residuals / sd(residuals)) %>% 
  filter(!cencored) %>% 
  ggplot(aes(fitted, sqrt(abs(stdresid)))) + 
  geom_point(na.rm = TRUE) + 
  stat_smooth(method="loess", na.rm = TRUE) + 
  labs(x = "Fitted Value", 
       y = expression(sqrt("|Standardized residuals|")),
       title = "Scale-Location") + 
  theme_bw() + 
  ggsave("5_scale_location.pdf", width = 6, height = 5)

## prediction ----
## predict std(relative error)
predict_re_std_log <- function(model, 
                               flip_prob = 0.15, 
                               num_sets = 10,
                               universe_size = 1e7L, 
                               sketch_size = 1e5L, 
                               ...) {
  log_universe_size = log(universe_size)
  log_sketch_size = log(sketch_size)
  df_new <- data.frame(
    num_sets = num_sets,
    log_universe_size = log_universe_size,
    log_sketch_size = log_sketch_size,
    flip_prob = flip_prob
  )
  return(predict(model, newdata = df_new, ...) )
}

## predict flip prob
predict_flip_prob <- function(model, 
                              re_std = 0.1,
                              num_sets = 20,
                              universe_size = 1e7L, 
                              sketch_size = 1e5L, 
                              tol = 1e-3) {
  re_std_log = log(re_std)
  lower = 0
  upper = 1 
  p_hat = (lower + upper) / 2
  while (lower + tol < upper) {
    re_std_log_hat <- predict_re_std_log(model, 
                                         num_sets = num_sets,
                                         universe_size = universe_size,
                                         sketch_size = sketch_size,
                                         flip_prob = p_hat, 
                                         inteval = "none")
    if (re_std_log_hat > re_std_log) {
      upper = p_hat
    } else {
      lower = p_hat
    }
    p_hat = (lower + upper) / 2
  }
  return(p_hat)
}


predict_re_std_log(fit0, 
                   flip_prob = .15,
                   num_sets = 15,
                   universe_size = 10000000, 
                   sketch_size = 100000, 
                   interval = "prediction") %>% exp()


predict_flip_prob(fit0, 
                  re_std = 0.1, 
                  num_sets = 10,
                  universe_size = 100000000, 
                  sketch_size = 10000)

predict_flip_prob(fit4, 
                  universe_size = 10000000, 
                  sketch_size = 1000000)
predict_flip_prob(fit0, 
                  re_std = 0.05, 
                  num_sets = 20,
                  universe_size = 200000000, 
                  sketch_size = 100000)

predict_flip_prob(fit4, 
                  universe_size = 10000000, 
                  sketch_size = 1000000)
predict_flip_prob(fit4, 
                  re_std = 0.05,
                  universe_size = 10000000, 
                  sketch_size = 1000000)
predict_flip_prob(fit4, 
                  universe_size = 10000000, 
                  sketch_size = 1000000)
predict_flip_prob(fit4, 
                  num_sets = 50,
                  universe_size = 10000000, 
                  sketch_size = 1000000)
predict_flip_prob(fit4, 
                  universe_size = 10000000, 
                  sketch_size = 500000)


## approximate form
fit2 <- fit0
fit2$coefficients <- c("(Intercept)" = log(1.2), 
                       "flip_prob" = log(100), 
                       "log_sketch_size" = -0.4, 
                       "log_universe_size" = 0, 
                       "flip_prob:num_sets" = log(2.5))
y_pred = predict(fit2, df)
## Residual plot
data.frame(cencored = df$re_std_log_cens == 0, 
           residuals = y_pred - fit4$y[,1], 
           fitted = y_pred) %>% 
  filter(!cencored) %>% 
  ggplot(aes(fitted, residuals)) + 
  geom_point(color = "grey40", alpha = .5) + 
  stat_smooth(method="loess") + 
  geom_hline(yintercept = 0, col = "red", linetype = "dashed") +
  labs(x = "Fitted values", y = "Residuals", 
       title = "Residual vs Fitted Plot") + 
  theme_bw() + 
  ggsave("5_residuals.pdf", width = 7, height = 5)

