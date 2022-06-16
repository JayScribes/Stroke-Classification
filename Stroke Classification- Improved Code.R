---
  title: "Stroke"
author: "Jeremy Larcher"
date: '2022-06-10'
output: html_document
---
  
  ```{r Library}
library(tidyr)
library(ggplot2)
library(corrr)
library(rsample)
library(recipes)
library(parsnip)
library(yardstick)
library(skimr)
library(psych)
library(ranger)
library(tidyverse)
library(tidymodels)
install.packages("GGally")
library(GGally)
install.packages("baguette")
options(scipen=999)
install.packages("doParallel")
library(doParallel)
install.packages("themis")
library(themis)
```

```{r EDA & Cleaning}
skim(Stroke)

pairs.panels(Stroke[,c('stroke', 'bmi', 'age','gender','hypertension','heart_disease','work_type')]) 
pairs.panels(Stroke[,c('stroke', 'avg_glucose_level', 'Residence_type','ever_married','gender')]) 

Stroke$stroke <- as.numeric(Stroke$stroke)

stroke.data <- Stroke %>% 
  filter(!is.na(bmi)) %>% 
  mutate(stroke = case_when(stroke > 0.5 ~ "Yes", TRUE ~ "No")) %>% 
  mutate(hypertension = case_when(hypertension > 0.5 ~ "Yes", TRUE ~ "No")) %>% 
  mutate(heart_disease = case_when(heart_disease > 0.5 ~ "Yes", TRUE ~ "No")) %>% 
  select(-id)

stroke.data$stroke <- as.factor(stroke.data$stroke)

skim(stroke.data)

## Numerical Plots
stroke.data %>% 
  select(stroke, age, avg_glucose_level, bmi) %>% 
  ggpairs(columns = 2:4, aes(color = stroke, alpha = 0.5))

## Categorical Plots

stroke.data %>% 
  select(stroke, gender, ever_married, work_type, Residence_type, hypertension, heart_disease, smoking_status) %>% 
  pivot_longer(gender:smoking_status) %>% 
  ggplot(aes(y = value, fill = stroke))+
  geom_bar(position = "fill")+
  facet_wrap(vars(name), scale = "free")+
  labs(x = NULL, y= NULL, fil = NULL)

```

```{r Splitting Data}
set.seed(123)
data_split <- initial_split(stroke.data)
data_train <- training(data_split)
data_test <- testing(data_split)
```

```{r Resampling Folds}
stroke_folds <- vfold_cv(data_train, v = 5, strata = stroke)
stroke_folds

stroke_metrics <- metric_set(accuracy, sensitivity, specificity)
```

```{r Recipe}
stroke_recipe <- recipe(stroke ~ ., data = data_train) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(), - all_outcomes()) %>% 
  step_zv(all_predictors())

stroke_recipe
```

```{r Bag Tree Model}
library(baguette)

bag_spec <-
  bag_tree(min_n = 10) %>% 
  set_engine("rpart", times = 25) %>% 
  set_mode("classification")
```

```{r Fitting Bag Tree onto data}
imb_wf <- workflow() %>% 
  add_recipe(stroke_recipe) %>% 
  add_model(bag_spec)

fit(imb_wf, data = data_train)
```

```{r Accounting for Class Imbalance}
doParallel::registerDoParallel()
set.seed(123)
imb_results <- fit_resamples(
  imb_wf,
  resamples = stroke_folds,
  metrics = stroke_metrics
)

collect_metrics(imb_results)

```

```{r UpSampling, Balanced}
bal_rec <- stroke_recipe %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_smote(stroke)


bal_wf <- workflow() %>% 
  add_recipe(bal_rec) %>% 
  add_model(bag_spec)

set.seed(123)
bal_results <- fit_resamples(
  bal_wf,
  resamples = stroke_folds,
  metrics = stroke_metrics,
  control = control_resamples(save_pred = TRUE))


collect_metrics(bal_results)

bal_results %>% 
  conf_mat_resampled()

```

```{r Fitting Onto Testing Data}
stroke_final <- bal_wf %>% 
  last_fit(data_split)

collect_metrics(stroke_final)

collect_predictions(stroke_final) %>% 
  conf_mat(stroke, .pred_class)


```

