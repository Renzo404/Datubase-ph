import pandas as pd
import torch
import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

def prepare_datasets(df):
    df["prov_name"] = df["prov_name"].astype(str)
    df["gov_is_dynasty"] = df["gov_is_dynasty"].astype(str)
    df["vice_gov_is_dynasty"] = df["vice_gov_is_dynasty"].astype(str)

    historical_df = df[df["year"] <= 2022].copy()
    train_mask = historical_df["year"] <= 2015

    scaler = StandardScaler()
    historical_df.loc[train_mask, "ira_funding_million_php"] = scaler.fit_transform(
        historical_df.loc[train_mask, ["ira_funding_million_php"]]
    )
    historical_df.loc[~train_mask, "ira_funding_million_php"] = scaler.transform(
        historical_df.loc[~train_mask, ["ira_funding_million_php"]]
    )

    training_dataset = TimeSeriesDataSet(
        historical_df[historical_df["year"] <= 2015],
        time_idx="time_idx",
        target="dynasty_share",
        group_ids=["prov_name"],
        min_encoder_length=9, max_encoder_length=9,
        min_prediction_length=1, max_prediction_length=6,
        static_categoricals=["prov_name"],
        time_varying_known_reals=["time_idx", "year"], 
        time_varying_unknown_reals=["poverty_incidence", "ira_funding_million_php", "dynasty_share"],
        time_varying_unknown_categoricals=["gov_is_dynasty", "vice_gov_is_dynasty"],
        add_relative_time_idx=True, 
        add_target_scales=True, 
        add_encoder_length=True,
        allow_missing_timesteps=True,  # <--- ADD THIS LINE HERE
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, historical_df[historical_df["year"] <= 2018], predict=True, stop_randomization=True
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, historical_df[historical_df["year"] <= 2022], predict=True, stop_randomization=True
    )

    return training_dataset, validation_dataset, test_dataset

def main():
    pl.seed_everything(42)
    print("Loading data and configuring PyTorch datasets...")
    df = pd.read_csv("../data/final/tft_master_dataset.csv")
    train_ds, val_ds, test_ds = prepare_datasets(df)
    
    train_dataloader = train_ds.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_dataloader = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4])
        lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2])
        dropout = trial.suggest_float("dropout", 0.05, 0.30)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        
        early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, mode="min")
        trainer = pl.Trainer(max_epochs=30, gradient_clip_val=0.01, callbacks=[early_stop], accelerator="auto", enable_progress_bar=False, logger=False)
        
        tft = TemporalFusionTransformer.from_dataset(
            train_ds, learning_rate=learning_rate, hidden_size=hidden_size,
            attention_head_size=attention_head_size, dropout=dropout, lstm_layers=lstm_layers,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]), optimizer="adam"
        )
        
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        return trainer.callback_metrics["val_loss"].item()

    print("\nStarting Optuna Hyperparameter Tuning (50 Trials)...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50)
    
    print("\n=== OPTUNA RESULTS ===")
    print(f"Best validation loss: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\nEvaluating Best Model on Test Set (2019-2022)...")
    best_tft = TemporalFusionTransformer.from_dataset(train_ds, **study.best_params)
    
    # Rapid retraining of the best model architecture
    trainer = pl.Trainer(max_epochs=20, gradient_clip_val=0.01, accelerator="auto", enable_progress_bar=False)
    trainer.fit(best_tft, train_dataloaders=train_dataloader)

    test_dataloader = test_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    preds = best_tft.predict(test_dataloader, return_y=True)
    mae = torch.nn.functional.l1_loss(preds.output[:, :, 1], preds.y[0]) 
    print(f"\nFINAL TFT TEST MAE: {mae.item():.2f}")

if __name__ == "__main__":
    main()