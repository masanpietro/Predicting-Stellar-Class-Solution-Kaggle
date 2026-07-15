# Kaggle Playground Series Predicting Stellar Class S6E6: CatBoost & Optuna Optimization

This repository contains my solution for the Kaggle Playground Series Predicting Stellar Class Competition (Season 6, Episode 6). It uses a CatBoost classifier combined with Optuna for hyperparameter tuning. The main focus of this approach is a custom evaluation metric designed to strictly penalize overfitting while maximizing validation accuracy.

Using this pipeline, the test accuracy achieved was 0.93697.

## The Custom Evaluation Metric

I wanted to prevent the model from simply memorizing the training data, which often happens when relying solely on standard loss minimization like Logloss. To fix this, the hyperparameter search is guided by a custom function. It takes the validation accuracy and divides it by the absolute gap between the training and validation performance.

I also added a small constant (0.01) to the denominator to prevent division by zero in case the accuracies match perfectly.

$$\text{Custom Score} = \frac{\text{Accuracy}_{val}}{|\text{Accuracy}_{train} - \text{Accuracy}_{val}| + 0.01}$$

By maximizing this equation, the pipeline naturally selects parameters that generalize much better to unseen data.

## Model Architecture and Search Space

The model is built on CatBoost. Categorical variables are one-hot encoded prior to training. To avoid building overly simplistic or completely brittle trees, I limited the Optuna search space, specifically enforcing a minimum tree depth of 5. 

The full search space explores:
* **Iterations:** 100 to 1000
* **Learning Rate:** 0.001 to 0.1 (Logarithmic scale)
* **Tree Depth:** 5 to 10 
* **L2 Leaf Regularization:** 1.0 to 10.0

## Installation and Requirements

Ensure you have the following dependencies installed in your Python environment:

```bash
pip install numpy pandas catboost scikit-learn optuna kagglehub
