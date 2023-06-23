import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor


def run_catboost(x: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=234)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(X_test)
    model = CatBoostRegressor()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)