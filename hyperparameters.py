
def get_params_xgb(trial):
    params = {
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]), # "gblinear" doesn't work on the dataset
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0)
    }

    if params["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        params["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)

        # minimum child weight, larger the term more conservative the tree.
        params["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)

        # defines how selective algorithm is.
        params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if params["booster"] == "dart":
        params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
 
    return params


def get_params_lgbm(trial):
    params = {
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
    }

    return params


def get_params_cat(trial):
    params = {
        "logging_level" : "Silent",
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        )
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    return params


def get_params_svr(trial):
    params = {
        "kernel" : trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "tol" : trial.suggest_float("tol", 1e-3, 1.0, log=True),
        "C": trial.suggest_float("C", 1e-3, 5, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 3, log=True)
    }
    
    if params["kernel"] in ["rbf", "poly", "sigmoid"]:
        params["gamma"] = "auto"
    
    if params["kernel"] in ["poly", "sigmoid"]:
        params["coef0"] = trial.suggest_float("coef0", 1e-3, 5.0, log=True)
    
    if params["kernel"] == "poly":
        params["degree"] = trial.suggest_int("degree", 1, 5)

    return params


def get_params_rf(trial):
    params = {
        "n_estimators" : trial.suggest_int("n_estimators", 5, 500),
        "max_depth" : trial.suggest_int('max_depth', 2, 24),
        "min_samples_split" : trial.suggest_float("min_samples_split", 1e-4, 1.0, log=True),
        "min_samples_leaf" : trial.suggest_float("min_samples_leaf", 1e-4, 1.0, log=True),
        "ccp_alpha" : trial.suggest_float("ccp_alpha", 1e-4, 1.0, log=True)
    }

    return params


def get_params_dt(trial):
    params = {
        "max_depth" : trial.suggest_int('max_depth', 2, 24),
        "min_samples_split" : trial.suggest_float("min_samples_split", 1e-4, 1.0, log=True),
        "min_samples_leaf" : trial.suggest_float("min_samples_leaf", 1e-4, 1.0, log=True),
        "ccp_alpha" : trial.suggest_float("ccp_alpha", 1e-4, 1.0, log=True)
    }

    return params


def get_params_knn(trial):
    params = {
        'n_neighbors': trial.suggest_int("n_neighbors", 4, 16),
        'weights': trial.suggest_categorical("weights", ['uniform', 'distance']),
        'algorithm': trial.suggest_categorical("algorithm", ['auto', 'ball_tree', 'kd_tree']),
        'leaf_size': trial.suggest_int("leaf_size", 7, 9),
        'p': trial.suggest_int("p", 1, 4)
    }

    return params