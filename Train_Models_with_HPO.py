""" Imports """
import numpy as np
import pandas as pd

import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import optuna

from hyperparameters import *

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
""" ---------- """


""" Parameters """
hpo_cv_n_trials = 100
hpo_n_jobs = -1
hpo_cv_n_splits = 5
hpo_final_n_trials = 200
optuna_verbosity = optuna.logging.WARNING
outer_cv_n_splits = 5
scaler_type = StandardScaler # MinMaxScaler 
scaler_params = {} # {"feature_range" : (-1, 1)} 
data_path = "..."
results_dir_path = "..."
""" ---------- """


df = pd.read_pickle(data_path)

print("\nStart! Daten:\n")
print(df)
print("\n")

features = ["...", "...", "..."]

targets = ["...", "...", "..."]


# Dictionary with model name, model class and function to get hyperparameters
models_and_get_params_funcs = [
    ("SVR", SVR, get_params_svr),
    ("Random Forest", RandomForestRegressor, get_params_rf),
    ("Decision Tree", DecisionTreeRegressor, get_params_dt),
    ("K-Nearest Neighbor", KNeighborsRegressor, get_params_knn),
    ("XGBoost", XGBRegressor, get_params_xgb),
    ("LightGBM", LGBMRegressor, get_params_lgbm)
    #("CatBoost", CatBoostRegressor, get_params_cat)
]


def objective_general(trial, reg_class, get_params_func, X_train, y_train):
    """General objective function for all models. Can be specialized (e.g. using a lambda expr.)
       to get an objective function for a particular model for Optuna. Returns negative RMSE."""

    params = get_params_func(trial)
    reg = reg_class(**params)
      
    neg_rmse = np.mean(cross_val_score(reg, X_train, y_train, cv=hpo_cv_n_splits, scoring="neg_root_mean_squared_error"))

    return neg_rmse


def main():
    # Go through all targets
    for target in targets:
        print(f"\nZielgröße: {target}\n")

        # Get data for the current target
        X = df[features]
        y = df[target].dropna()
        X = X.loc[y.index]

        # Outer cross-validation
        kfold = KFold(n_splits=outer_cv_n_splits, shuffle=True, random_state=42)

        # List for the results of the cross-validation
        results = []

        # Go through all folds
        for i, (train_index, test_index) in enumerate(kfold.split(X)):
            print(f"\nFold {i + 1}\n")

            # Preprocess and split
            X_train, X_test, y_train, y_test = prepare_data(X, y, train_index, test_index)

            # Go through all models
            for model_name, model_class, get_params_func in models_and_get_params_funcs:
                # Hyperparameter optimization
                params = optimize_model(model_name, model_class, get_params_func, X_train, y_train, hpo_cv_n_trials)
                # Train and evaluate model with best hyperparameters
                print(f"\nTrainiere Modell mit besten Hyperparametern auf allen Trainingsdaten\n")
                reg = model_class(**params)
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                # Collect results
                results.append({
                    "Name" : model_name, 
                    "R2"   : r2_score(y_test, y_pred),
                    "RMSE" : mean_squared_error(y_test, y_pred, squared=False),
                    "MAPE" : mean_absolute_percentage_error(y_test, y_pred),
                    **params
                })

        # Convert results dict to pandas DataFrame
        results_df = pd.DataFrame(results, index=list(range(len(results))))

        # Save results to Excel file
        print(f"\nSpeichere Ergebnisse der Cross-Validation\n")
        results_df.to_excel(os.path.join(results_dir_path, f"Ergebnisse_Alle_Modelle_CV_{target.replace(' ', '_')}.xlsx"))

        # Get the average results for each model
        results_avg_df = results_df[["Name", "R2", "RMSE", "MAPE"]].groupby("Name").mean()

        # Save average results to Excel file
        print(f"\nSpeichere durchschnittliche Ergebnisse der Cross-Validation\n")
        results_avg_df.to_excel(os.path.join(results_dir_path, f"Ergebnisse_Alle_Modelle_CV_Durchschnitt_{target.replace(' ', '_')}.xlsx"))

        # Get the model with the best average RMSE
        best_model = results_avg_df["RMSE"].idxmin()
        best_model_and_get_params_func = list(filter(lambda t: t[0] == best_model, models_and_get_params_funcs))[0]
        model_class = best_model_and_get_params_func[1]
        get_params_func = best_model_and_get_params_func[2]

        # Train the best model on all data

        print(f"\nOptimiere Parameter für bestes Modell auf allen Daten\n")

        X = df[features]
        y = df[target].dropna()
        X = X.loc[y.index]

        params = optimize_model(best_model, model_class, get_params_func, X, y, hpo_final_n_trials)

        print(f"\nFertig!\nHyperparameter: {params}\n")
        
        # Save the hyperparameters for the best model

        print(f"\nSpeichere Parameter für bestes Modell\n")
        save_results(params, best_model, results_avg_df, target)

        print("\nFertig!\n")


def prepare_data(X, y, train_index=None, test_index=None):
    """Preprocesses data and splits it into train and test set.
       Returns X_train, X_test, y_train, y_test."""

    if train_index is not None and test_index is not None:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    print(f"\nDatensätze mit Zielgröße != NaN: {len(X)}\nTrainingsset: {len(X_train)}\nTestset: {len(X_test)}\n")

    scaler = scaler_type(**scaler_params)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def optimize_model(model_name, model_class, get_params_func, X_train, y_train, n_trials):
    """Optimizes the hyperparameters of a model using Optuna.
       Returns the best hyperparameters."""

    print(f"\nModell: {model_name}, Starte Hyperparameter-Optimierung\n")

    objective = lambda trial: objective_general(trial, model_class, get_params_func, X_train, y_train)

    optuna.logging.set_verbosity(optuna_verbosity)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=hpo_n_jobs)

    print(f"\nModell: {model_name}, Fertig\nBester Score: {-1 * study.best_value:.2f}\nHyperparameter:\n{study.best_params}\n")

    return study.best_params


def save_results(params, best_model, results_avg_df, target):
    """Saves the results of the hyperparameter optimization as a pickled pandas DataFrame"""

    params_df = pd.DataFrame(index=[best_model])

    params_df.loc[best_model, "Average R2"] = results_avg_df.loc[best_model, "R2"]
    params_df.loc[best_model, "Average RMSE"] = results_avg_df.loc[best_model, "RMSE"]
    params_df.loc[best_model, "Average MAPE"] = results_avg_df.loc[best_model, "MAPE"]
    params_df.loc[best_model, "Target"] = target

    params_df = pd.concat([params_df, pd.DataFrame(params, index=[best_model])], axis="columns")

    params_df.to_pickle(os.path.join(results_dir_path, f"Bestes_Modell_{target.replace(' ', '_')}.pkl"))


if __name__ == "__main__":
    main()