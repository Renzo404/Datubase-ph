# Datubase-ph 🏛️📊

### *Deciphering the Digital Kadatuan: A Predictive Analysis of Philippine Dynasties*

**`Datubase-ph`** is a data science repository dedicated to breaking the "analytical ceilings" of Philippine political research. By treating the 2022 Ateneo Policy Center (APC) dataset as a modern "Datu-base," we apply Deep Learning to quantify and forecast how political bloodlines achieve provincial saturation.

---

## 🏛️ The Wordplay: Data vs. Datu
The name **`Datubase-ph`** plays on the phonetic overlap between a standard **Database** and the **Datu** (the pre-colonial sovereign). It highlights the democratic irony where modern political "databases" are still populated by "Datu" lineages—revealing a system where power is often a birthright rather than a choice.

## ⚠️ Data Acquisition & Provenance
* **Target Data (APC):** To run these notebooks, you must manually download the **APC 2022 Political Dynasties Dataset** from the Ateneo Policy Center. Place the `.xlsx` file into `data/raw/` before running the preprocessing notebooks.
* **Feature Data (PSY):** The repository includes manually compiled and structurally modified CSVs derived entirely from historical **Philippine Statistical Yearbook (PSY)** reports. This includes both the socio-economic indicators (Poverty Incidence) and the financial records (Internal Revenue Allotment). These public data transformations are available in the `data/modified/` directory.

## 🗺️ Geographic Harmonization (ETL Rules)
Temporal Fusion Transformers strictly require unbroken, continuous timelines. To ensure structural integrity across our temporal grid, the following spatial rules are enforced to match a standardized **81-Province** landscape:

* **The "Non-Province" Exclusion:** The National Capital Region (NCR) districts (`1ST DISTRICT`, etc.) and Independent Component Cities (`COTABATO CITY`, `ISABELA CITY`) are explicitly dropped. They do not elect Provincial Governors, making them mathematically incompatible with our primary dynastic feature flags.
* **Re-unification of Split Boundaries:** To prevent broken time indices, territories that split or merged are mathematically re-unified (averaged) to maintain their continuous historical landmass:
    * *Maguindanao del Norte / Maguindanao del Sur* (Split post-2022) -> *Maguindanao*
    * *Shariff Kabunsuan* (Briefly existed 2006-2008) -> *Maguindanao*
* **Nomenclature Standardization:** Historical text variations are mapped to Tableau-compliant standard names: 
    * *Western Samar* -> *Samar*
    * *Cotabato* -> *North Cotabato*
    * *Compostela Valley* -> *Davao de Oro*
    * *Mt. Province* -> *Mountain Province*
    * *Saranggani* -> *Sarangani*

## 🚀 Research Gaps & Workflow
This project implements a technical pipeline to address two specific "analytical ceilings":

1.  **Temporal Gap:** Harmonizing triennial election cycles with annual socio-economic features to create a continuous, imputed 22-year panel (2000–2022), explicitly truncated to exclude the 1990s due to missing provincial financial allocations in the early PSY records.
2.  **Predictive Gap:** Using the **Temporal Fusion Transformer (TFT)** for multi-horizon forecasting of dynastic saturation toward 2025–2028.

---

## 🛠️ Repository Flow
Please execute the notebooks in `notebooks/` in the following strict order to reproduce the data pipeline:

### Phase 1: ETL & Feature Engineering
* `01_prep_poverty_incidence.ipynb`: Cleans legacy PSY data and standardizes regional headers.
* `02_prep_ira_funding.ipynb`: Standardizes Internal Revenue Allotment allocations from historical PSY tables.
* `03_prep_apc_poldyn.ipynb`: Cleans the base APC dataset and isolates targeted provincial executive features.
* `04_tft_master_merge.ipynb`: Builds the continuous temporal grid. It executes the critical `pd.merge()`, truncates missing pre-2000 financial data, interpolates poverty variables, forward-fills political terms, and generates the `time_idx` column required by PyTorch Forecasting.

### Phase 2: Modeling & Analysis
* `05_tft_forecasting.ipynb`: Model training, hyperparameter tuning, and future saturation predictions using the TFT architecture.

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