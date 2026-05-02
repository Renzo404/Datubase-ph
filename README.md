# Datubase-ph 🏛️📊

### *Deciphering the Digital Kadatuan: A Predictive Analysis of Philippine Dynasties*

**`Datubase-ph`** is a data science repository dedicated to breaking the "analytical ceilings" of Philippine political research. By treating the 2022 Ateneo Policy Center (APC) dataset as a modern "Datu-base," we apply Deep Learning to quantify and forecast how political bloodlines achieve provincial saturation.

---

## 🏛️ The Wordplay: Data vs. Datu
The name **`Datubase-ph`** plays on the phonetic overlap between a standard **Database** and the **Datu** (the pre-colonial sovereign). It highlights the democratic irony where modern political "databases" are still populated by "Datu" lineages—revealing a system where power is often a birthright rather than a choice.

## ⚠️ Data Acquisition & Provenance
* **Target Data (APC):** To run these notebooks, you must manually download the **APC 2022 Political Dynasties Dataset** from the Ateneo Policy Center. Place the `.xlsx` file into `data/raw/` before running the preprocessing notebooks.
* **Feature Data (PSY):** The repository includes manually compiled and structurally modified CSVs derived from historical Philippine Statistical Yearbook (PSY) reports. These public data transformations are available in the `data/modified/` directory.

## 🗺️ Geographic Harmonization (ETL Rules)
To ensure structural integrity across 30 years of data for our Temporal Fusion Transformer (TFT), the following strict spatial rules are enforced to match the **May 2022** political landscape:

* **Boundary Limits:** The Maguindanao separation (post-May 2022) is out of scope. The Negros Island Region is excluded as it was reverted in the 2024 PSY records and was not a legal entity during the 2022 elections.
* **Nomenclature Standardization:** Renamed historical records to match the 2022 map: 
    * *Western Samar* → *Samar*
    * *North Cotabato* → *Cotabato*
    * *Compostela Valley* → *Davao de Oro*
* **Unbalanced Panel Handing (The "Missing Years"):** To prevent the model from learning synthetic historical patterns, splinter provinces and newly defined entities are left explicitly blank (`NaN`) for the years prior to their creation:
    * **Davao Occidental:** Blank prior to 2015.
    * **Dinagat Islands:** Blank prior to its establishment.
    * **Zamboanga Sibugay:** Blank in the 2000 records and prior.
    * **NCR Districts:** District-level data was not recorded in the 1991–2000 PSY tables and is intentionally left blank to train the TFT on an unbalanced timeline.

## 🚀 Research Gaps & Workflow
This project implements a technical pipeline to address two specific "analytical ceilings":

1.  **Temporal Gap:** Integrating longitudinal socioeconomic features (1991–2023) with the 2019/2022 election cycles.
2.  **Predictive Gap:** Using the **Temporal Fusion Transformer (TFT)** for multi-horizon forecasting of dynastic saturation toward 2028-2031.

---

## 🛠️ Repository Flow
Please execute the notebooks in `notebooks/` in the following strict order to reproduce the data pipeline:

### Phase 1: ETL & Feature Engineering
* `01_prep_poverty_incidence.ipynb`: Cleans PSY legacy data, handling missing variables and province splits.
* `02_prep_ira_funding.ipynb`: Standardizes Internal Revenue Allotment allocations.
* `03_prep_apc_poldyn.ipynb`: Cleans the base APC dataset and isolates target variables.
* `04_tft_master_merge.ipynb`: Final `pd.merge()` of all datasets, generating the `time_idx` column required for TFT sequence length detection.

### Phase 2: Modeling & Analysis
* `05_tft_forecasting.ipynb`: Model training, hyperparameter tuning, and future saturation predictions using TFT.

---

## 🔧 Setup & "Clean Diff" Protocol
To prevent Jupyter metadata (execution counts, cell outputs) from cluttering our Git history, we use `nbstripout`.

### Installation
```bash
# 0. From the parent Datubase folder, enter the project
cd Datubase-ph

# 1. Create and activate a repo-local virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install core dependencies inside the virtual environment
python -m pip install -r requirements.txt

# 3. Setup the automated notebook cleaner
python -m pip install nbstripout
nbstripout --install
