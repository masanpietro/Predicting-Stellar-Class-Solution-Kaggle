import kagglehub
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

kagglehub.login()
path = kagglehub.competition_download('playground-series-s6e6')
train = pd.read_csv(path + "/train.csv")
test = pd.read_csv(path + "/test.csv")

def preprocess_data(df):
    df.fillna("NA", inplace=True)
    df = pd.get_dummies(df, columns=['galaxy_population', 'spectral_type'], drop_first=True)
    return df

processedtrain = preprocess_data(train)
processedtest = preprocess_data(test)

X = processedtrain.drop(columns=['id', 'class'])
y = processedtrain['class']

# Ensure test columns perfectly match train columns after get_dummies
processedtest = processedtest.reindex(columns=X.columns, fill_value=0)

# Split data into train and validation sets for evaluating the metric
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    # Define the search space with a minimum depth of 5 to prevent underfitting
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'depth': trial.suggest_int('depth', 5, 10), # MINIMUM DEPTH SET TO 5
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_state': 42,
        'verbose': False
    }
    
    # Train the model on the current trial's parameters
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    
    # Get predictions (flattened to 1D)
    train_preds = model.predict(X_train).flatten()
    val_preds = model.predict(X_val).flatten()
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    # Calculate accuracy/(|prediction-value|+0.01), 
    # essentially maximizing accuracy divided by a penalty for overfitting 
    # (the absolute difference between train and val accuracy plus a small constant to prevent division by zero)
    overfit_penalty = abs(train_acc - val_acc) + 0.01 
    custom_score = val_acc / overfit_penalty
    
    return custom_score

# Run the Optuna study to maximize custom metric
study = optuna.create_study(direction="maximize")
#20 trials is a reasonable number for a quick optimization, but this can be increased for better results at the cost of a longer runtime
study.optimize(objective, n_trials=20) 

# Train the final model using the absolute best parameters on ALL the data
best_model = CatBoostClassifier(**study.best_params, random_state=42, verbose=False)
best_model.fit(X, y)

# Predict and output submission
test_predictions = best_model.predict(processedtest).flatten()
submission = pd.DataFrame({'id': test['id'], 'class': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Done")