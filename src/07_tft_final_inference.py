import pandas as pd
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings("ignore")

def main():
    pl.seed_everything(42)
    print("Loading data for Final Inference...")
    df = pd.read_csv("../data/final/tft_master_dataset.csv")

    df["prov_name"] = df["prov_name"].astype(str)
    df["gov_is_dynasty"] = df["gov_is_dynasty"].astype(str)
    df["vice_gov_is_dynasty"] = df["vice_gov_is_dynasty"].astype(str)

    scaler = StandardScaler()
    df["ira_funding_million_php"] = scaler.fit_transform(df[["ira_funding_million_php"]])

    full_training_dataset = TimeSeriesDataSet(
            df[df["year"] <= 2022],
            time_idx="time_idx",
            target="dynasty_share",
            group_ids=["prov_name"],
            min_encoder_length=9, max_encoder_length=9,
            min_prediction_length=1, max_prediction_length=6,
            static_categoricals=["prov_name"],
            time_varying_known_reals=["time_idx", "year"], 
            time_varying_unknown_reals=["poverty_incidence", "ira_funding_million_php", "dynasty_share"],
            time_varying_unknown_categoricals=["gov_is_dynasty", "vice_gov_is_dynasty"],
            add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
            allow_missing_timesteps=True # <--- ADDED THIS LINE HERE
        )
    
    full_dataloader = full_training_dataset.to_dataloader(train=True, batch_size=32, num_workers=0)

    # ---> IMPORTANT: Update these with the exact output from Script 06 <---
    # best_params = {
    #     'hidden_size': 64, 
    #     'attention_head_size': 4,
    #     'lstm_layers': 2,
    #     'dropout': 0.124,
    #     'learning_rate': 0.0035
    # }
    
    #Updated
    best_params = {
        "hidden_size": 128,
        "attention_head_size": 2,
        "lstm_layers": 1,
        "dropout": 0.21694811239602307,
        "learning_rate": 0.0012622311438484168
    }

    print("\nTraining Final TFT Model on complete 2000-2022 panel...")
    trainer = pl.Trainer(max_epochs=50, gradient_clip_val=0.01, accelerator="auto", enable_progress_bar=True)
    final_tft = TemporalFusionTransformer.from_dataset(full_training_dataset, **best_params)
    trainer.fit(final_tft, train_dataloaders=full_dataloader)

    print("\nExtracting Interpretability Metrics...")
    os.makedirs("../data/outputs", exist_ok=True)

    # Generate predictions to extract the interpretation dictionaries
    predictions = final_tft.predict(full_dataloader, return_x=True, mode="raw") # <--- CHANGED TO "raw"
    interpretation = final_tft.interpret_output(predictions.output, reduction="sum")

    # 1. Export VSN Weights
    vsn_weights = pd.DataFrame({
        "Variable": final_tft.encoder_variables,
        "Importance": interpretation["encoder_variables"].numpy()
    })
    vsn_weights.to_csv("../data/outputs/vsn_importance.csv", index=False)
    print(" [✓] Saved VSN weights to: data/outputs/vsn_importance.csv")

    # 2. Export Attention Matrix
    attention_weights = pd.DataFrame(interpretation["attention"].numpy())
    attention_weights.to_csv("../data/outputs/attention_weights.csv", index=False)
    print(" [✓] Saved Attention Matrix to: data/outputs/attention_weights.csv")
    
    print("\n=== PIPELINE COMPLETE ===")
    print("Your data is ready to be loaded into Tableau.")

if __name__ == "__main__":
    main()