


## Background

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

Data provided by Fedesoriano on Kaggle.
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset


__Variables in The Data__

1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
  12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient


## Loading Libraries
```{r}
library(corrr)
library(skimr)
library(psych)
library(ranger)
library(tidyverse)
library(tidymodels)
library(GGally)
options(scipen=999)
library(doParallel)
library(themis)
library(baguette)
```

## Loading Data

```{r warning = FALSE}
Stroke <- read_csv("C:/Users/Dell/Desktop/Data Projects/Portfolio/ML Projects/Stroke/healthcare-dataset-stroke-data.csv", 
                   col_types = cols(age = col_number(), 
                                    avg_glucose_level = col_number(), 
                                    bmi = col_number()))
```

## EDA & Cleaning

### Skimming Data
```{r}
skim(Stroke)
```


### Creating Data Set

```{r}
stroke.data <- Stroke %>% 
  filter(!is.na(bmi)) %>% 
  mutate(stroke = case_when(stroke > 0.5 ~ "Yes", TRUE ~ "No")) %>% 
  mutate(hypertension = case_when(hypertension > 0.5 ~ "Yes", TRUE ~ "No")) %>% 
  mutate(heart_disease = case_when(heart_disease > 0.5 ~ "Yes", TRUE ~ "No")) %>% 
  select(-id)

stroke.data$stroke <- as.factor(stroke.data$stroke)
```

### Pairs Plots - Numerical Variables

```{r}
stroke.data %>% 
  select(stroke, age, avg_glucose_level, bmi) %>% 
  ggpairs(columns = 2:4, aes(color = stroke, alpha = 0.5))
```

### Pairs Plots - Categorical Varibles

```{r}
stroke.data %>% 
  select(stroke, gender, ever_married, work_type, Residence_type, hypertension, heart_disease, smoking_status) %>% 
  pivot_longer(gender:smoking_status) %>% 
  ggplot(aes(y = value, fill = stroke))+
  geom_bar(position = "fill")+
  facet_wrap(vars(name), scale = "free")+
  labs(x = NULL, y= NULL, fil = NULL)

```

## Splitting Data
```{r}
set.seed(123)
data_split <- initial_split(stroke.data)
data_train <- training(data_split)
data_test <- testing(data_split)
```

## Resampling Folds
```{r}
stroke_folds <- vfold_cv(data_train, v = 5, strata = stroke)
stroke_folds

stroke_metrics <- metric_set(accuracy, sensitivity, specificity, recall)
```

## Recipe

```{r}
stroke_recipe <- recipe(stroke ~ ., data = data_train) %>% 
  step_normalize(all_numeric()) %>% 
  step_dummy(all_nominal(), - all_outcomes()) %>% 
  step_zv(all_predictors())

stroke_recipe
```

## Generating Model
```{r}
bag_spec <-
  bag_tree(min_n = 10) %>% 
  set_engine("rpart", times = 25) %>% 
  set_mode("classification")
```

## Fitting Model onto Data

```{r}
imb_wf <- workflow() %>% 
  add_recipe(stroke_recipe) %>% 
  add_model(bag_spec)

var.imp.t <- fit(imb_wf, data = data_train)
```

## Accounting For Imbalance

```{r}
doParallel::registerDoParallel()
set.seed(123)
imb_results <- fit_resamples(
  imb_wf,
  resamples = stroke_folds,
  metrics = stroke_metrics
)

collect_metrics(imb_results)
```

## Upsampling, Balancing

```{r}
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

## Fitting on Testing Data

```{r}
stroke_final <- bal_wf %>% 
  last_fit(data_split)

collect_metrics(stroke_final)

collect_predictions(stroke_final) %>% 
  conf_mat(stroke, .pred_class)
```

## Variable Importance Table - Test Fitting
```{r}
var.imp.t
```

## ROC Curve - Test Fitting

```{r}
stroke_final %>% 
  collect_predictions() %>% 
  group_by(id) %>% 
  roc_curve(stroke, .pred_No) %>% 
  ggplot(aes(1 - specificity, sensitivity, color = id))+
  geom_abline(lty = 2, color = "gray90", size = 1.5)+
  geom_path(show.legend = FALSE, alpha = 0.6, size =1.2)+
  coord_equal()+theme_classic()
```


## Confusion Matrix Graph - Test Fitting

```{r}
collect_predictions(stroke_final) %>% 
  conf_mat(stroke, .pred_class) %>% 
  autoplot(cm, type = "heatmap")
```

