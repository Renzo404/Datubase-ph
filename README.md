# Datubase-ph 🏛️📊

### *Deciphering the Digital Kadatuan: A Predictive & Network Analysis of Philippine Dynasties*

**`Datubase-ph`** is a data science repository dedicated to breaking the "analytical ceilings" of Philippine political research. By treating the 2022 Ateneo Policy Center (APC) dataset as a modern "Datu-base," we apply Graph Theory and Deep Learning to quantify how bloodlines transition from municipal "feeder systems" to provincial saturation.

---

## 🏛️ The Wordplay: Data vs. Datu
The name **`Datubase-ph`** plays on the phonetic overlap between a standard **Database** and the **Datu** (the pre-colonial sovereign). It highlights the democratic irony where modern political "databases" are still populated by "Datu" lineages—revealing a system where power is often a birthright rather than a choice.

## ⚠️ Data Acquisition Note
To run these notebooks, you must manually download the **APC 2022 Political Dynasties Dataset** from the Ateneo Policy Center. 
Place the `.xlsx` file into `data/raw/` before running the preprocessing notebook.

## 🚀 Research Gaps & Workflow
This project implements a technical pipeline to address four specific "analytical ceilings":

1.  **Temporal Gap**: Integrating 2019/2022 election cycles.
2.  **Predictive Gap**: Using the **Temporal Fusion Transformer (TFT)** for 2028-2031 forecasting.
3.  **Feeder System Gap**: Granular Path Analysis of the municipal-to-provincial pipeline.
4.  **Network Influence Gap**: Applying **Graph Theory** to quantify real political "hubs."

---

## 🛠️ Repository Flow
Please execute the notebooks in the following order:

* `01_preprocessing.ipynb`: Data cleaning and entity resolution for surnames.
* `02_network_analysis.ipynb`: Calculating Centrality Metrics (Degree, Betweenness).
* `03_path_analysis.ipynb`: Modeling the municipal "Launchpad" effect.
* `04_tft_forecasting.ipynb`: Multi-horizon saturation predictions using TFT.

---

## 🔧 Setup & "Clean Diff" Protocol
To prevent Jupyter metadata from cluttering Git history, we use `nbstripout`.

### Installation
```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Setup the automated notebook cleaner
pip install nbstripout
nbstripout --install