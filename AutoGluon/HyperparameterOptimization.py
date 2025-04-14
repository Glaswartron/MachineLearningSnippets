from autogluon.common import space

# Specifies non-default hyperparameter values for neural network models
_nn_options = {
    'num_epochs': space.Int(5, 30, default=10),
    'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),
    'activation': space.Categorical('relu', 'softrelu', 'tanh'),
    'dropout_prob': space.Real(0.0, 0.5, default=0.1),
}

# XGBoost hyperparameters
_xgb_hyperparameters = {
    "learning_rate" : space.Real(0.01, 0.99, default=0.3),
    "max_depth" : space.Int(1, 40, default=6),
    "gamma" : space.Real(0, 15, default=0),
    "min_child_weight" : space.Real(0.2, 10, default=1),
    "reg_lambda" : space.Real(0.2, 500, default=1),
    "subsample" : space.Real(0.05, 1, default=1),
    "colsample_bytree" : space.Real(0.05, 1, default=1),
    "n_estimators" : space.Int(500, 5000, default=2000)
}

# LightGBM hyperparameters
_lgbm_hyperparameters = {
    "max_bin" : space.Int(10, 1024, default=256),
    "learning_rate" : space.Real(0.01, 0.99, default=0.1),
    "num_leaves" : space.Int(6, 3000, default=31),
    "lambda_l1" : space.Real(0, 500, default=0),
    "lambda_l2" : space.Real(0, 500, default=0),
    "min_data_in_leaf" : space.Int(1, 100, default=20)
}

# Catboost hyperparameters (causes errors)
# catboost_hyperparameters = {
#     "learning_rate" : space.Real(0.01, 0.99, default=0.03),
#     "max_depth" : space.Int(1, 16, default=6),
#     "l2_leaf_reg" : space.Real(0, 500, default=3.0),
#     "n_estimators" : space.Int(500, 5000, default=1000)
# }

hyperparameters = {
    'NN_TORCH': _nn_options,
    "XGB" : _xgb_hyperparameters,
    # "CAT": catboost_hyperparameters,
    'GBM': _lgbm_hyperparameters
} 