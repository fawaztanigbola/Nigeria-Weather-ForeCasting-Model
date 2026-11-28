# Nigeria Weather Forecasting Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Domain](https://img.shields.io/badge/Domain-Climate%20%26%20Weather%20Forecasting-green.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An end-to-end **spatio-temporal data processing and deep learning project** developed to investigate and forecast daily minimum and maximum temperatures across Nigeria, with a focused regional implementation covering Southwest Nigeria.

The project integrates historical climate reanalysis data from **ERA5**, topographical elevation information derived from Digital Elevation Models (DEM), and Recurrent Neural Networks (LSTM) in PyTorch to evaluate multi-location temporal and geospatial forecasting.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Contributions](#-key-contributions)
- [Tech Stack](#-tech-stack)
- [Repository Workflow & Pipeline Architecture](#-repository-workflow--pipeline-architecture)
- [Dataset & Study Area](#-dataset--study-area)
- [Model Architecture & Experiments](#-model-architecture--experiments)
- [Model Performance & Benchmarks](#-model-performance--benchmarks)
- [Getting Started & Installation](#-getting-started--installation)
- [Usage & Quick Start](#-usage--quick-start)
- [Key Findings](#-key-findings)
- [Limitations & Future Work](#-limitations--future-work)
- [License](#-license)

---

## 🔍 Project Overview

Temperature forecasting plays a critical role in climate analysis, agricultural planning, environmental monitoring, and regional resource management. This project addresses the challenge of predicting daily minimum ($T_{\text{min}}$) and maximum ($T_{\text{max}}$) temperatures across spatial grid points using deep learning.

Beginning with a broader nationwide objective, the active empirical focus was refined to **Southwest Nigeria** (Lagos, Ogun, Oyo, and Osun states) to manage Google Colab computational constraints (RAM and training wall-clock time) while retaining a multi-location grid-based spatial problem.

### Key Pipeline Stages
1. **Climate & Elevation Acquisition**: Downloading netCDF historical ERA5 data via ECMWF API (`cdsapi`) and DEM elevation datasets.
2. **Multi-Year Integration**: Concatenating multi-year climate datasets into a continuous 10-year time series (2015–2024).
3. **Data Cleaning & Preprocessing**: Handling missing values, extracting daily temperature extremes, and normalizing features.
4. **Sequence Formatting**: Structuring temporal sliding-window sequences for recurrent neural network inputs.
5. **Model Experimentation**: Training baseline temperature-only LSTMs vs. geospatial/seasonal LSTMs incorporating location (Lat, Lon, Elevation) and cyclical day-of-year embeddings.
6. **Benchmarking**: Evaluating metrics (MAE, RMSE) against baseline persistence and day-of-year climatology reference forecasts.

---

## ✨ Key Contributions

- **10-Year Climate Data Engineering**: Ingested and structured 10 years (2015–2024) of daily ERA5 climate reanalysis metrics across 456 grid locations.
- **Geospatial Feature Fusion**: Combined temporal climate signals with static geographical attributes (latitude, longitude, DEM elevation).
- **Cyclical Encoding**: Modeled seasonal variations using day-of-year sine and cosine embeddings.
- **Deep Learning Architectures**: Built and evaluated baseline temporal LSTMs and spatial-concatenation LSTMs in PyTorch.
- **Controlled Benchmarking**: Computed relative forecast skill scores against both Persistence ($T_{t+1} = T_t$) and Climatology reference models.
- **Ablation Studies**: Tested historical window lengths (7, 14, 30 days) and static feature inclusion.

---

## 💻 Tech Stack

| Domain | Tools & Libraries |
| :--- | :--- |
| **Programming Language** | Python 3.8+ |
| **Deep Learning Framework** | PyTorch |
| **Climate Data Source** | Copernicus ERA5 Reanalysis, DEM Elevation |
| **Data Ingestion & Formatting** | `cdsapi`, `xarray`, `netCDF4`, `h5netcdf` |
| **Data Processing & Analytics** | `pandas`, `numpy`, `scipy`, `scikit-learn` |
| **Spatial Data Analysis** | `geopandas`, `rasterio`, `shapely` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Execution Environment** | Jupyter Notebooks, Google Colab (Tesla T4 GPU) |

---

## 🏗 Repository Workflow & Pipeline Architecture

### Notebook Map

```text
├── ERA5_And_DEM_Dataset_Download.ipynb      # Step 1: CDS API & DEM data retrieval
├── Combine_Yearly_Files.ipynb               # Step 2: Merge multi-year netCDF dataset files
├── Daily_Temperature_Preprocessing.ipynb    # Step 3: Clean, impute missing values, compute daily Tmin/Tmax
├── Models_Data_Preparation.ipynb            # Step 4: Sliding window generation & feature scaling
├── Chronologically_Splitting_Data.ipynb     # Step 5: Partition dataset chronologically (Train/Val/Test)
├── Baseline_LSTM.ipynb                      # Step 6: Train & evaluate temperature-only LSTM baseline
├── Geospatial_Model.ipynb                   # Step 7: Train & evaluate Geospatial + Cyclical LSTM
└── Southwest_Nigeria_Temperature_Project_Documentation.ipynb # Step 8: Consolidated analysis & report
```

### Pipeline Flowchart

```text
┌──────────────────────────────────────────────┐
│ ERA5 & DEM Data Acquisition                  │
│ Historical ERA5 NetCDF & Elevation DEM      │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Multi-Year Data Integration                  │
│ Concatenate yearly files (2015-2024)         │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Daily Temperature Preprocessing              │
│ Missing value handling, Tmin/Tmax extraction │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Sequence & Feature Engineering               │
│ Sliding windows, cyclical day-of-year, scaling│
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Chronological Partitioning                   │
│ Train / Validation / Test temporal splits    │
└──────────────────────┬───────────────────────┘
                       ▼
             ┌─────────┴─────────┐
             ▼                   ▼
┌────────────────────┐  ┌──────────────────────┐
│ Baseline LSTM      │  │ Geospatial LSTM      │
│ Temperature-only   │  │ Temporal sequence +  │
│ (7-day historical) │  │ Lat/Lon/Elev/Cyclic  │
└─────────┬──────────┘  └──────────┬───────────┘
          └────────────┬───────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Model Evaluation & Benchmarking              │
│ Compare vs. Persistence & Climatology        │
└──────────────────────────────────────────────┘
```

---

## 🌍 Dataset & Study Area

The primary region evaluated covers Southwest Nigeria (Lagos, Ogun, Oyo, Osun).

| Property | Value / Description |
| :--- | :--- |
| **Data Source** | ERA5 Climate Reanalysis |
| **Time Period** | 2015–2024 (10 years) |
| **Temporal Frequency** | Daily |
| **Total Timesteps** | 3,653 daily records |
| **Active Grid Cells** | 456 usable geographical locations |
| **Target Variables** | Daily $T_{\text{min}}$ and $T_{\text{max}}$ (°C) |
| **Geospatial Variables** | Latitude, Longitude, Topographical Elevation (DEM) |

---

## 🧠 Model Architecture & Experiments

Two neural network architectures were constructed and trained:

### 1. Temperature-Only Baseline LSTM
Designed to establish a benchmark performance metric using solely historical daily temperatures.
* **Inputs**: Previous 7 days of $T_{\text{min}}$ and $T_{\text{max}}$.
* **Architecture**: Single-layer LSTM (Hidden size = 64) $\rightarrow$ Linear dense layer $\rightarrow$ Output ($T_{\text{min}}$, $T_{\text{max}}$).
* **Optimization**: Adam optimizer, Mean Squared Error (MSE) loss, Early Stopping.

### 2. Geospatial LSTM
Designed to fuse temporal sequences with static spatial and dynamic seasonal context.
* **Temporal Inputs**: $T_{\text{min}}$, $T_{\text{max}}$, $\sin(\text{DOY})$, $\cos(\text{DOY})$.
* **Static Inputs**: Latitude, Longitude, Elevation.
* **Architecture**: Sequence processed by LSTM $\rightarrow$ Concatenation of final hidden state with scaled static spatial vector $\rightarrow$ Fully connected prediction layer.

### Ablation Experiments Conducted
* **Lookback Windows**: Tested 7, 14, and 30-day historical input sequences. (7-day window yielded optimal validation loss).
* **Network Depth**: Tested single-layer vs. multi-layer LSTMs with dropout. (Single-layer without dropout yielded optimal performance).
* **Spatial Feature Effect**: Compared model convergence with and without static spatial attributes.

---

## 📊 Model Performance & Benchmarks

Metrics represent Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) in **degrees Celsius (°C)** on unseen test data.

### 1. Model Errors

| Target | Baseline LSTM (MAE) | Baseline LSTM (RMSE) | Geospatial LSTM (MAE) | Geospatial LSTM (RMSE) |
| :--- | ---: | ---: | ---: | ---: |
| **Min Temperature ($T_{\text{min}}$)** | **0.502°C** | **0.690°C** | 0.535°C | 0.707°C |
| **Max Temperature ($T_{\text{max}}$)** | **0.988°C** | **1.324°C** | 1.021°C | 1.346°C |

### 2. Skill Scores (% Error Reduction over Reference Models)

Skill score is defined as:
$$\text{Skill Score} = \left( 1 - \frac{\text{Error}_{\text{Model}}}{\text{Error}_{\text{Reference}}} \right) \times 100$$

#### Forecast Skill vs. Persistence Baseline ($T_{t+1} = T_t$)

| Target | Baseline LSTM (MAE Skill) | Baseline LSTM (RMSE Skill) | Geospatial LSTM (MAE Skill) | Geospatial LSTM (RMSE Skill) |
| :--- | ---: | ---: | ---: | ---: |
| **$T_{\text{min}}$** | +12.85% | +11.31% | +8.23% | +9.24% |
| **$T_{\text{max}}$** | +11.55% | +16.94% | +10.75% | +17.73% |

#### Forecast Skill vs. Day-of-Year Climatology

| Target | Geospatial LSTM (MAE Skill) | Geospatial LSTM (RMSE Skill) |
| :--- | ---: | ---: |
| **$T_{\text{min}}$** | +41.34% | +41.43% |
| **$T_{\text{max}}$** | +25.91% | +22.24% |

---

## 🛠 Getting Started & Installation

### Prerequisites

* Python 3.8+
* Jupyter Notebook or JupyterLab / Google Colab environment
* CDS API key (if acquiring raw ERA5 datasets directly from Copernicus)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/fawaztanigbola/Nigeria-Weather-ForeCasting-Model.git
   cd Nigeria-Weather-ForeCasting-Model
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install torch numpy pandas xarray netcdf4 cdsapi scikit-learn geopandas rasterio matplotlib seaborn jupyter
   ```

---

## 🚀 Usage & Quick Start

To reproduce the workflow from raw data ingestion to model training and evaluation, run the Jupyter notebooks in the following order:

1. **Data Acquisition**:
   Open `ERA5_And_DEM_Dataset_Download.ipynb` to download ERA5 netCDF files and DEM rasters. (Requires configuring `~/.cdsapirc` with your Copernicus API credentials).
   
2. **File Integration**:
   Run `Combine_Yearly_Files.ipynb` to merge yearly climate datasets into a unified netCDF time series.

3. **Data Cleaning & Normalization**:
   Run `Daily_Temperature_Preprocessing.ipynb` to process missing values and extract daily extreme temperatures ($T_{\text{min}}$, $T_{\text{max}}$).

4. **Sequence Construction**:
   Run `Models_Data_Preparation.ipynb` and `Chronologically_Splitting_Data.ipynb` to build sliding-window sequences and create train/validation/test temporal splits.

5. **Model Training & Evaluation**:
   - For temperature-only LSTM baseline: Run `Baseline_LSTM.ipynb`.
   - For geospatial/cyclical LSTM: Run `Geospatial_Model.ipynb`.
   - For consolidated project documentation and detailed results: Open `Southwest_Nigeria_Temperature_Project_Documentation.ipynb`.

---

## 📌 Key Findings

1. **LSTMs Outperform Naïve References**: Both the Baseline LSTM and Geospatial LSTM demonstrated statistically significant error reductions over persistence ($T_{t+1} = T_t$) and seasonal climatology.
2. **Optimal Lookback Window**: A 7-day lookback window provided sufficient temporal history without overfitting, outperforming 14-day and 30-day windows.
3. **Static Feature Ablation**: Incorporating static spatial attributes (latitude, longitude, elevation) directly via concatenation did not yield performance gains over the purely temporal baseline in this spatial configuration.
4. **Predictability Gap**: Minimum daily temperature ($T_{\text{min}}$) was predicted with higher accuracy (MAE ~0.50°C) than maximum daily temperature ($T_{\text{max}}$) (MAE ~0.99°C), reflecting the higher variance of daytime thermal dynamics.

---

## ⚠️ Limitations & Future Work

- **Compute Boundaries**: The scope was limited to Southwest Nigeria due to memory bounds when staging large spatiotemporal matrices in cloud environments.
- **Identical Partitioning Protocol**: Future iterations should evaluate baseline and spatial architectures on unified cross-validation splits to enable direct comparative hypothesis testing.
- **Spatial Deep Learning**: Incorporating Convolutional Neural Networks (ConvLSTM / Graph Neural Networks) could better model regional spatial dependencies across adjacent grid cells than static point concatenation.
- **Additional Meteorological Variables**: Including humidity, solar radiation, cloud cover, and precipitation as dynamic input channels could further improve $T_{\text{max}}$ forecast skill.

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE).