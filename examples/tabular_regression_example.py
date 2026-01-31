"""Complete tabular regression example with improvements and comparisons.

Loads datasets, applies various models and techniques, and compares MSE.
"""
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression

# Load datasets
data_small = load_diabetes()
df_small = pd.DataFrame(data_small.data, columns=data_small.feature_names)
df_small['target'] = data_small.target

data_large = fetch_california_housing()
df_large = pd.DataFrame(data_large.data, columns=data_large.feature_names)
df_large['target'] = data_large.target

datasets = {'Diabetes (small)': (df_small, 'target'), 'Boston Housing (large)': (df_large, 'target')}

results = {}

for name, (df, target_col) in datasets.items():
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Tuned RandomForest
    model_rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    results[f'{name} - Tuned RF'] = mse_rf
    
    # 2. Cross-validation (mean MSE)
    cv_scores = cross_val_score(model_rf, X, y, cv=5, scoring='neg_mean_squared_error')
    mse_cv = -cv_scores.mean()
    results[f'{name} - RF CV'] = mse_cv
    
    # 3. Ridge regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model_ridge = Ridge(alpha=0.1)
    model_ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = model_ridge.predict(X_test_scaled)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    results[f'{name} - Ridge'] = mse_ridge
    
    # 4. SVR
    model_svr = SVR(kernel='rbf', C=1.0)
    model_svr.fit(X_train_scaled, y_train)
    y_pred_svr = model_svr.predict(X_test_scaled)
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    results[f'{name} - SVR'] = mse_svr
    
    # 5. Feature engineering (polynomial + selection)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    selector = SelectKBest(f_regression, k=10)
    X_selected = selector.fit_transform(X_poly, y)
    X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model_rf.fit(X_train_sel, y_train_sel)
    y_pred_eng = model_rf.predict(X_test_sel)
    mse_eng = mean_squared_error(y_test_sel, y_pred_eng)
    results[f'{name} - RF with Eng'] = mse_eng

# Print comparisons
print("MSE Comparisons:")
for key, mse in results.items():
    print(f"{key}: {mse:.2f}")