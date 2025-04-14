import pandas as pd

import os

from autogluon.tabular import TabularPredictor
from .AutoGluonMultilabelPredictor import MultilabelPredictor
from . import HyperparameterOptimization as hpo

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

def train_and_eval_predictor_multiclass(label, preset, predictor_path,
    X=None, y=None, X_train=None, y_train=None, X_test=None, y_test=None,
    eval_metric="f1", problem_type=None, test_size=0.2, scale_data=True, stratify=True, predictor_kwargs={}, fit_kwargs={}):
    """
    Trains and evaluates a multi-class predictor using AutoGluon, either by splitting the provided data or using pre-split data.

    Args:
        label (str): The name of the target column in the dataset.
        preset (str): Preset configurations for AutoGluon's TabularPredictor.
        predictor_path (str): Path to save the trained predictor.
        X (pd.DataFrame, optional): Features dataset to be split into training and testing sets.
        y (pd.Series, optional): Target labels corresponding to the X dataset.
        X_train (pd.DataFrame, optional): Pre-split training features dataset.
        y_train (pd.Series, optional): Pre-split training labels.
        X_test (pd.DataFrame, optional): Pre-split test features dataset.
        y_test (pd.Series, optional): Pre-split test labels.
        eval_metric (str, optional): Evaluation metric to use (e.g., "f1"). Default is "f1".
        problem_type (str, optional): The type of prediction problem ("binary", "multiclass", etc.). Default is None, meaning it will be inferred from the data.
        test_size (float, optional): Fraction of data to use for testing if X and y are provided. Default is 0.2.
        scale_data (bool, optional): Whether to scale the data using StandardScaler. Default is True.
        stratify (bool, optional): Whether to stratify the data when splitting. Default is True.
        predictor_kwargs (dict, optional): Additional keyword arguments for the TabularPredictor initialization.
        fit_kwargs (dict, optional): Additional keyword arguments for the TabularPredictor's fit method.

    Raises:
        ValueError: If neither the full dataset (X, y) nor pre-split data (X_train, y_train, X_test, y_test) are provided.

    Returns:
        dict: A dictionary containing:
            - "performance" (dict): The performance metrics of the model on the test data.
            - "leaderboard" (pd.DataFrame): Leaderboard of model performances.
            - "feature_importances" (pd.DataFrame): Feature importance values for the test data.

    Example:
        result = train_and_eval_predictor_multiclass(
            label="target",
            preset="best_quality",
            predictor_path="./predictor",
            X=df_features,
            y=df_target
        )
    """

    # If neither X and y nor X_train, y_train, X_test, and y_test are provided, raise an error
    if (X is None or y is None) and (X_train is None or y_train is None or X_test is None or y_test is None):
        raise ValueError("Either X and y or X_train, y_train, X_test, and y_test must be provided.")

    '''
    If X and y are provided, use them to split data into training and test set.
    If X_train, y_train, X_test, and y_test are provided, use them directly.
    '''
    if X is not None and y is not None:
        # Split data into training and test set
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    '''
    Scale data. Not necessary for AutoGluon, but good practice and important for models to be usable
    in two-step approach together with OneClassSVM (which uses scaled data).
    '''
    if scale_data:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Add labels back to data for AutoGluon
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Train and save AutoGluon predictor
    predictor = TabularPredictor(label=label, problem_type=problem_type, eval_metric=eval_metric, path=predictor_path, **predictor_kwargs)
    predictor.fit(train_data, presets=preset, **fit_kwargs)

    # Evaluate predictor
    performance = predictor.evaluate(test_data)
    leaderboard = predictor.leaderboard(test_data)
    feature_importances = predictor.feature_importance(test_data)

    return {
        "performance": performance,
        "leaderboard": leaderboard,
        "feature_importances": feature_importances,
        "predictor": predictor
    }

def run_cross_validation_multiclass(X, y, label, preset,
    n_splits=5, eval_metric="f1", problem_type=None, scale_data=True, predictor_kwargs={}, fit_kwargs={}):
    """
    Runs cross-validation for a multi-class predictor using AutoGluon using a StratifiedKFold split.

    Args:
        label (str): The name of the target column in the dataset.
        preset (str): Preset configurations for AutoGluon's TabularPredictor.
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target labels corresponding to the X dataset.
        eval_metric (str, optional): Evaluation metric to use (e.g., "f1"). Default is "f1".
        problem_type (str, optional): The type of prediction problem ("binary", "multiclass", etc.). Default is None, meaning it will be inferred from the data.
        scale_data (bool, optional): Whether to scale the data using StandardScaler. Default is True.
        predictor_kwargs (dict, optional): Additional keyword arguments for the TabularPredictor initialization.
        fit_kwargs (dict, optional): Additional keyword arguments for the TabularPredictor's fit method.

    Returns:
        pd.DataFrame: A DataFrame containing the cross-validation results with mean and standard deviation.
    """

    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store results
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []

    # Cross-validation
    for i, (train_index, test_index) in enumerate(skfold.split(X, y)):
        # Split data into training and test set based on StratifiedKFold split indices
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        '''
        Scale data. Not necessary for AutoGluon, but good practice and important for models to be usable
        in two-step approach together with OneClassSVM (which uses scaled data).
        '''
        if scale_data:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        # Add labels back to data for AutoGluon
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # Train AutoGluon predictor
        predictor = TabularPredictor(label=label, problem_type=problem_type, eval_metric=eval_metric, **predictor_kwargs)
        predictor.fit(train_data, presets=preset, **fit_kwargs)

        # Evaluate predictor
        leaderboard = predictor.leaderboard(test_data, silent=True)
        best_model = leaderboard.loc[0, "model"]
        results = predictor.evaluate(test_data, model=best_model, silent=True)

        # Store results
        accuracies.append(results["accuracy"])
        f1_scores.append(results["f1"])
        precisions.append(results["precision"])
        recalls.append(results["recall"])

    # Present results as a table, with mean and standard deviation
    results = pd.DataFrame({"Accuracy": accuracies, "F1": f1_scores, "Precision": precisions, "Recall": recalls}).T
    results.columns = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
    results["Mean"] = results.mean(axis=1)
    results["Std"] = results.std(axis=1)

    return results

def train_and_eval_predictor_multilabel(
    labels, preset, predictor_path,
    X=None, y=None, X_train=None, y_train=None, X_test=None, y_test=None,
    eval_metrics=None, problem_types=None, test_size=0.2, scale_data=True, 
    optimize_hyperparameters=False, hpo_num_trials=50, 
    feature_prune_kwargs=None, # TODO
    predictor_kwargs={}, fit_kwargs={}):
    """
    Trains and evaluates a multilabel predictor using AutoGluon.  
    Includes the option to optimize hyperparameters.
    Args:
        label (str): The label for the predictor.
        preset (str): The preset configuration for AutoGluon.
        predictor_path (str): The path to save the trained predictor.
        X (pd.DataFrame, optional): The input features for training and testing.
        y (pd.DataFrame, optional): The target labels for training and testing.
        X_train (pd.DataFrame, optional): The input features for training.
        y_train (pd.DataFrame, optional): The target labels for training.
        X_test (pd.DataFrame, optional): The input features for testing.
        y_test (pd.DataFrame, optional): The target labels for testing.
        eval_metrics (list, optional): The evaluation metrics to use. Defaults to None.
        problem_types (list, optional): The problem types for each label. Defaults to None.
        test_size (float, optional): The proportion of the dataset to use for testing. Defaults to 0.2.
        scale_data (bool, optional): Whether to scale the data. Defaults to True.
        optimize_hyperparameters (bool, optional): Whether to optimize hyperparameters. Defaults to False.
        hpo_num_trials (int, optional): The number of hyperparameter optimization trials. Defaults to 50.
        predictor_kwargs (dict, optional): Additional keyword arguments for the predictor. Defaults to {}.
        fit_kwargs (dict, optional): Additional keyword arguments for the fit method. Defaults to {}.
    Returns:
        dict: A dictionary containing the average performance and per-label performance of the predictor and the predictor itself.  
              Keys: "performance_avg", "performance_per_label", "predictor"
    """
    

    # If neither X and y nor X_train, y_train, X_test, and y_test are provided, raise an error
    if (X is None or y is None) and (X_train is None or y_train is None or X_test is None or y_test is None):
        raise ValueError("Either X and y or X_train, y_train, X_test, and y_test must be provided.")

    if eval_metrics is None:
        print("No evaluation metrics provided. Using 'f1' for all labels.")
        eval_metrics = ["f1"] * len(y.columns) if y is not None else ["f1"] * len(y_train.columns)

    '''
    If X and y are provided, use them to split data into training and test set.
    If X_train, y_train, X_test, and y_test are provided, use them directly.
    '''
    if X is not None and y is not None:
        # Split data into training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    '''
    Scale data. Not necessary for AutoGluon, but good practice and important for models to be usable
    in two-step approach together with OneClassSVM (which uses scaled data).
    '''
    if scale_data:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Add labels back to data for AutoGluon
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    if not optimize_hyperparameters:
        # Train and save AutoGluon predictor
        predictor = MultilabelPredictor(labels=labels, problem_types=problem_types, eval_metrics=eval_metrics, path=predictor_path, **predictor_kwargs)
        predictor.fit(train_data, presets=preset, **fit_kwargs)
    else:
        # Hyperparameter-optimize and save AutoGluon predictor
        search_strategy = 'auto'
        hyperparameter_tune_kwargs = {
            'num_trials': hpo_num_trials,
            'scheduler' : 'local',
            'searcher': search_strategy,
        }
        predictor = MultilabelPredictor(labels=labels, problem_types=problem_types, eval_metrics=eval_metrics,
                                        path=predictor_path, **predictor_kwargs)
        predictor.fit(train_data, presets=preset, 
                      hyperparameters=hpo.hyperparameters, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                      **fit_kwargs)

    # Evaluate predictor
    avg_performance = predictor.evaluate(test_data, per_predictor=False)
    per_label_performance = predictor.evaluate(test_data, per_predictor=True)

    # Leaderboard and feature importances cannot be generated easily for MultilabelPredictor
    # since they are different for every sub-predictor
    # Possible TODO: Implement leaderboard and feature importances in Multilabel Predictor for every sub-predictor
    # leaderboard = predictor.leaderboard(test_data)
    # feature_importances = predictor.feature_importance(test_data)

    return {
        "performance_avg": pd.DataFrame(avg_performance, index=["Value"]),
        "performance_per_label": pd.DataFrame(per_label_performance),
        "predictor": predictor,
        # "leaderboard": leaderboard,
        # "feature_importances": feature_importances
    }