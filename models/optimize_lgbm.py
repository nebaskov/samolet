import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import optuna

train = pd.read_csv("../data/train_refactored_tables.csv")
train.drop(["Лом_3А_руб_т_без_НДС_y",
            "ЖРС_среднее_руб_т_без_НДС_y",
            "Чугун_Россия, FCA руб./т, без НДС_y"
            ], axis=1, inplace=True)

test = pd.read_csv("../data/test_refactored_tables.csv")
test.drop(["Лом_3А_руб_т_без_НДС_y",
           "ЖРС_среднее_руб_т_без_НДС_y",
           "Чугун_Россия, FCA руб./т, без НДС_y"
           ], axis=1, inplace=True)

X_train = train.drop(["dt", "Цена на арматуру"], axis=1)
y_train = train["Цена на арматуру"]
X_test = test.drop(["dt", "Цена на арматуру"], axis=1)
y_test = test["Цена на арматуру"]

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)


def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 125,
        "verbosity": -1,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2 ** 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(x_train, y_train, verbose=False)
    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)
