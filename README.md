# Datubase-ph

### Deciphering the Digital Kadatuan: A Predictive Analysis of Philippine Political Dynasties

`Datubase-ph` is a data science and forecasting repository for analyzing political dynasty saturation in the Philippine elective landscape. The project transforms socioeconomic, fiscal, and electoral data into a harmonized provincial panel and applies Temporal Fusion Transformer (TFT) modeling to forecast future dynastic saturation.

The study focuses on provincial-level political dynasty concentration, using historical data from 2000 to 2022 and generating multi-horizon forecasts for the 2028 and 2031 electoral horizons.

---

## Project Purpose

This repository supports a research project on the structural persistence and future risk of political dynasties in the Philippines. It combines:

- electoral dynasty indicators,
- poverty incidence,
- fiscal allocation data,
- provincial harmonization rules,
- baseline forecasting models, and
- interpretable deep learning through Temporal Fusion Transformers.

The main goal is not only to predict dynastic saturation, but also to make the forecasting process explainable through attention weights, variable importance, benchmark comparisons, and visualization-ready outputs.

---

## The Wordplay: Data vs. Datu

The name `Datubase-ph` plays on the overlap between a standard “database” and the “Datu,” a pre-colonial political leader. The wordplay reflects the project’s focus on how modern electoral datasets can reveal persistent patterns of inherited political power.

---

## Scope of the Dataset

The unit of analysis is the Philippine province. The modeling panel uses a standardized 81-province structure from 2000 to 2022.

The repository excludes areas that do not fit the provincial governance structure used by the model. These include:

- National Capital Region districts,
- independent component cities, and
- non-provincial electoral units.

The project focuses on provincial-level dynastic saturation and provincial executive indicators. It does not model national offices, party-list seats, barangay-level positions, or individual candidate-level campaign dynamics.

---

## Data Sources and Provenance

The repository uses three major categories of data:

1. **Political dynasty indicators**
   - Derived from the Ateneo Policy Center political dynasties dataset.
   - The APC source file must be manually downloaded and placed in `data/raw/` before running the preprocessing notebooks.

2. **Socioeconomic indicators**
   - Poverty incidence data compiled from Philippine Statistical Yearbook sources.
   - Cleaned and transformed versions are stored under `data/modified/poverty_incidence/`.

3. **Fiscal indicators**
   - Internal Revenue Allotment / National Tax Allotment-related fiscal data compiled from Philippine Statistical Yearbook sources.
   - Cleaned and transformed versions are stored under `data/modified/ira/`.

The final modeling dataset is stored as:

```text
data/final/tft_master_dataset.csv
```

---

## Geographic Harmonization Rules

Temporal forecasting requires a continuous and consistent provincial panel. To avoid broken time indices, the project applies geographic harmonization rules before modeling.

### Excluded units

The following are removed because they do not fit the provincial executive structure used in the model:

- NCR districts,
- Cotabato City,
- Isabela City,
- other non-provincial units.

### Reunified or standardized units

Certain historical boundary changes are harmonized to maintain continuity:

- Maguindanao del Norte and Maguindanao del Sur are treated under Maguindanao for historical continuity.
- Shariff Kabunsuan is folded back into Maguindanao.
- Western Samar is standardized as Samar.
- Cotabato is standardized as North Cotabato.
- Compostela Valley is standardized as Davao de Oro.
- Mt. Province is standardized as Mountain Province.
- Saranggani is standardized as Sarangani.

---

## Research Gaps Addressed

This repository addresses two major analytical gaps.

### 1. Temporal gap

Philippine election data is naturally tied to election cycles, while socioeconomic and fiscal indicators may be annual or irregular. The project harmonizes these sources into a continuous 2000–2022 provincial panel.

### 2. Predictive and interpretability gap

The project uses the Temporal Fusion Transformer to forecast future dynasty saturation while also generating interpretable outputs such as:

- variable importance,
- temporal attention weights,
- quantile-based uncertainty estimates,
- benchmark comparisons,
- ground-truth alignment figures, and
- Tableau-ready dashboard files.

---

## Repository Structure

```text
Datubase-ph/
│
├── data/
│   ├── final/
│   │   └── tft_master_dataset.csv
│   │
│   └── modified/
│       ├── ira/
│       └── poverty_incidence/
│
├── notebooks/
│   ├── 01_prep_poverty_incidence.ipynb
│   ├── 02_prep_ira_funding.ipynb
│   ├── 03_prep_apc_poldyn.ipynb
│   ├── 04_tft_master_merge.ipynb
│   ├── 05_tft_modelling.ipynb
│   └── 05_tft_modelling_finetuned_v3.ipynb
│
├── src/
│   ├── 05_baseline_models.py
│   ├── 06_tft_tuning_and_eval.py
│   └── 07_tft_final_inference.py
│
├── outputs/
│   ├── figures/
│   ├── logs/
│   └── results/
│
├── tableauref/
│   ├── benchmark_results.csv
│   ├── dashboard_forecast_final.csv
│   ├── optuna_trials.csv
│   ├── section8_forecast_uncertainty_distribution.csv
│   ├── section8_quantile_metrics.csv
│   ├── tft_attention_weights.csv
│   ├── tft_forecast_output.csv
│   ├── tft_master_dataset.csv
│   ├── tft_variable_importance.csv
│   └── Tableau workbook files
│
├── requirements.txt
└── README.md
```

---

## Required Raw Dataset

Before running the preprocessing notebooks, download the Ateneo Policy Center (APC) Political Dynasties Dataset Excel file from the Inclusive Democracy data page:

https://www.inclusivedemocracy.ph/data-and-infographics

After downloading the file, place it inside the following folder:

    data/raw/

This raw APC Excel file is required for the political dynasty preprocessing notebook. It provides the local-level political dynasty indicators used to construct the provincial dynasty variables included in the final TFT-ready dataset.

---

## Pipeline Overview

### Phase 1: Data Preparation

Run the preprocessing notebooks in order:

```text
01_prep_poverty_incidence.ipynb
02_prep_ira_funding.ipynb
03_prep_apc_poldyn.ipynb
04_tft_master_merge.ipynb
```

These notebooks clean the poverty, fiscal, and political dynasty datasets, then merge them into the final TFT-ready provincial panel.

Output:

```text
data/final/tft_master_dataset.csv
```

### Phase 2: Modeling and Evaluation

The main modeling workflow is handled through the TFT modeling notebooks and supporting scripts.

Important modeling files:

```text
notebooks/05_tft_modelling.ipynb
notebooks/05_tft_modelling_finetuned_v3.ipynb
src/05_baseline_models.py
src/06_tft_tuning_and_eval.py
src/07_tft_final_inference.py
```

The modeling phase includes:

- baseline model comparison,
- Optuna hyperparameter tuning,
- TFT model training,
- held-out 2019–2022 evaluation,
- 2028 and 2031 forecast generation,
- quantile uncertainty analysis,
- attention heatmap export,
- variable importance export,
- ground-truth alignment visualization, and
- Tableau dashboard preparation.

---

## Current Forecasting Targets

The current repository outputs focus on the following horizons:

```text
2028 forecast horizon
2031 forecast horizon
```

These horizons are used for future dynastic saturation forecasting and dashboard visualization.

---

## Main Outputs

### Results

Stored in:

```text
outputs/results/
tableauref/
```

Important CSV outputs include:

```text
benchmark_results.csv
dashboard_forecast_final.csv
optuna_trials.csv
section8_forecast_uncertainty_distribution.csv
section8_quantile_metrics.csv
tft_attention_weights.csv
tft_forecast_output.csv
tft_variable_importance.csv
```

### Figures

Stored in:

```text
outputs/figures/
```

Important generated figures include:

```text
fig_attention_heatmap.png
fig_benchmark.png
fig_correlation_heatmap.png
fig_correlation_scatter.png
fig_forecast_uncertainty_distribution.png
fig_section8_quantile_metrics.png
fig_top5_trajectories.png
fig_top5_trajectories_2028.png
fig_top5_trajectories_2031.png
fig_variable_importance.png
```

Recent updates also include expanded 2022 ground-truth alignment figures for high, middle, and low dynastic saturation tiers.

### Tableau Reference Files

Stored in:

```text
tableauref/
```

This folder contains dashboard-ready CSVs and Tableau workbook files for visualizing:

- provincial forecast outputs,
- 2028 and 2031 projected dynasty saturation,
- benchmark metrics,
- uncertainty distributions,
- variable importance,
- attention weights, and
- final dashboard datasets.

---

## Model Benchmarks

The repository compares the TFT model against traditional baseline methods:

- Naive Persistence,
- SARIMA,
- Ridge Regression,
- Temporal Fusion Transformer.

The latest committed benchmark file includes a held-out 2019–2022 evaluation set. These results should be interpreted as model evaluation outputs rather than final political claims.

---

## Interpretability Outputs

The TFT pipeline produces interpretability files to support model explanation.

### Variable importance

```text
tft_variable_importance.csv
fig_variable_importance.png
```

These outputs summarize which model inputs contributed most strongly to the TFT forecasts.

### Attention weights

```text
tft_attention_weights.csv
fig_attention_heatmap.png
```

These outputs support temporal interpretation by showing which historical time steps the model emphasized during forecasting.

### Forecast uncertainty

```text
section8_forecast_uncertainty_distribution.csv
section8_quantile_metrics.csv
fig_forecast_uncertainty_distribution.png
fig_section8_quantile_metrics.png
```

These outputs summarize uncertainty through quantile-based predictions and pinball loss metrics.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Renzo404/Datubase-ph.git
cd Datubase-ph
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it.

For macOS/Linux:

```bash
source venv/bin/activate
```

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository requirements indicate that the environment was tested with Python 3.11.

### 4. Install notebook cleanup tool

```bash
python -m pip install nbstripout
nbstripout --install
```

This keeps notebook outputs and execution metadata from cluttering Git commits.

---

## Running the Pipeline

### Data preparation

Open and run the notebooks in this order:

```text
notebooks/01_prep_poverty_incidence.ipynb
notebooks/02_prep_ira_funding.ipynb
notebooks/03_prep_apc_poldyn.ipynb
notebooks/04_tft_master_merge.ipynb
```

### Baseline models

The baseline script expects paths relative to the `src/` directory.

```bash
cd src
python 05_baseline_models.py
```

### TFT tuning and evaluation

```bash
python 06_tft_tuning_and_eval.py
```

### Final TFT inference

```bash
python 07_tft_final_inference.py
```

For the most updated modeling workflow, use:

```text
notebooks/05_tft_modelling_finetuned_v3.ipynb
```

---

## Reproducibility Notes

- Use the same Python environment across preprocessing, modeling, and visualization steps.
- Keep `data/final/tft_master_dataset.csv` synchronized with the latest preprocessing notebooks.
- Avoid committing large temporary notebook outputs unless needed for documentation.
- Use `nbstripout` before committing notebooks.
- Tableau files in `tableauref/` may be large because they include packaged workbook data.

---

## Limitations

This repository supports forecasting and exploratory interpretation. The model outputs should not be treated as deterministic predictions of political outcomes.

Known limitations include:

- the analysis is limited to the provincial level;
- non-provincial electoral units are excluded;
- the model does not capture campaign spending, party-switching, voter behavior, or informal patronage networks directly;
- future predictions depend on the quality and continuity of historical data;
- geographic harmonization may simplify complex boundary changes;
- forecast uncertainty should be considered when interpreting province-level results.

---

## Suggested Citation / Acknowledgment

This repository was developed for an academic data science project on Philippine political dynasty forecasting using Temporal Fusion Transformers.

Data sources include publicly available or manually compiled Philippine electoral, fiscal, and socioeconomic datasets. Users should properly cite the original data providers, including the Ateneo Policy Center and Philippine Statistical Yearbook sources, when using this repository for academic work.
