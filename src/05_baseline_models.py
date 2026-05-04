import pandas as pd
import numpy as np
import itertools
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

# Suppress warnings from statsmodels for a clean terminal output
warnings.filterwarnings("ignore")

def run_baselines():
    print("Loading preprocessed dataset...")
    df = pd.read_csv("../data/final/tft_master_dataset.csv")
    df = df.sort_values(by=["prov_name", "year"]).reset_index(drop=True)

    # Define test set (2019-2022)
    test_set = df[(df["year"] >= 2019) & (df["year"] <= 2022)].copy()

    # --- 1. Naive Persistence ---
    print("\n[1/3] Running Naive Persistence Baseline...")
    last_known = df[df["year"] == 2018][["prov_name", "dynasty_share"]].copy()
    last_known.rename(columns={"dynasty_share": "naive_forecast"}, inplace=True)

    naive_results = test_set.merge(last_known, on="prov_name", how="left")
    naive_mae = np.mean(np.abs(naive_results["dynasty_share"] - naive_results["naive_forecast"]))

    # --- 2. SARIMA (Univariate) ---
    print("[2/3] Running SARIMA Baseline (Grid Search per province)...")
    p_vals, d_vals, q_vals = [0, 1, 2], [0, 1], [0, 1, 2]
    pdq = list(itertools.product(p_vals, d_vals, q_vals))
    sarima_forecasts = []
    
    for province in df["prov_name"].unique():
        prov_data = df[df["prov_name"] == province]
        train_series = prov_data[prov_data["year"] <= 2018]["dynasty_share"].values
        
        best_aic, best_model = float("inf"), None
        for order in pdq:
            try:
                model = SARIMAX(train_series, order=order, enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic, best_model = results.aic, results
            except: continue
                
        if best_model is not None:
            forecast = best_model.forecast(steps=4)
            for i, year in enumerate([2019, 2020, 2021, 2022]):
                sarima_forecasts.append({"prov_name": province, "year": year, "sarima_forecast": forecast[i]})

    sarima_df = pd.DataFrame(sarima_forecasts)
    sarima_merged = test_set.merge(sarima_df, on=["prov_name", "year"], how="left")
    sarima_mae = np.mean(np.abs(sarima_merged["dynasty_share"] - sarima_merged["sarima_forecast"]))

    # --- 3. Ridge Regression (Multivariate) ---
    print("[3/3] Running Ridge Regression Baseline...")
    features = ["time_idx", "poverty_incidence", "ira_funding_million_php"]
    target = "dynasty_share"

    train_df = df[df["year"] <= 2018].dropna(subset=features + [target])
    test_df_ridge = test_set.dropna(subset=features + [target])

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(train_df[features], train_df[target])
    ridge_preds = ridge.predict(test_df_ridge[features])
    ridge_mae = mean_absolute_error(test_df_ridge[target], ridge_preds)

    print("\n=== BASELINE RESULTS (Mean Absolute Error) ===")
    print(f"Naive Persistence: {naive_mae:.2f}")
    print(f"SARIMA:            {sarima_mae:.2f}")
    print(f"Ridge Regression:  {ridge_mae:.2f} (Penalty Alpha: {ridge.alpha_})")
    print("==============================================")

if __name__ == "__main__":
    run_baselines()