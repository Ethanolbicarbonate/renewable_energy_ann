# Artificial Neural Network for Solar Energy Forecasting
Repo link: https://github.com/Ethanolbicarbonate/renewable_energy_ann

A Python-based machine learning pipeline that predicts hourly photovoltaic (PV) energy generation for a 1 kWp system in Iloilo City, Philippines, utilizing a multilayer feedforward ANN and localized meteorological data.

## Project Deliverables
- `docs/CCS229_final_report.pdf` — Comprehensive academic project report detailing methodology, model architecture, and evaluation.
- `docs/IS_PRESENTATION_compressed.pdf` — Slide deck summarizing the project context, results, and conclusions.
- `figures/` — Directory containing all generated data correlation heatmaps, loss curves, and comparative evaluation plots.
- `data/` — Directory containing the raw downloaded CSVs and the chronologically split, scaled NumPy binaries.

## Notebook Pipeline
- `notebooks/01_preprocessing.ipynb` — Handles automated API data extraction (PVGIS, NSRDB), UTC to PST timezone alignment, continuous/categorical feature engineering, hybrid scaling, and chronological data splitting (70/15/15).
- `notebooks/02_training.ipynb` — Defines, compiles, and trains the multilayer feedforward ANN. Implements LeakyReLU activations, Huber Loss ($\delta=1.0$), Dropout regularization, and dynamic callbacks (EarlyStopping, ReduceLROnPlateau).
- `notebooks/03_evaluation.ipynb` — Evaluates the ANN against a Linear Regression baseline. Applies inverse transformations to calculate physical error metrics (RMSE, MAE in kWh) and generates all final validation visualizations.

## Models & Artifacts
- `models/best_model_iloilo.keras` — The serialized, trained ANN model containing the optimal weights restored from the best training epoch.
- `models/scaler_X.save` — The scikit-learn `MinMaxScaler` fitted exclusively on the training set's continuous input features.
- `models/scaler_y.save` — The scikit-learn `MinMaxScaler` fitted to normalize the target energy generation variable.

## Data & Features
The system models the complex, non-linear relationship between atmospheric conditions and energy conversion.

**Target Variable:**
- `energy_kwh` — Hourly solar energy output in kilowatt-hours (converted from raw Watts), sourced from the PVGIS-ERA5 database.

**Input Features (NSRDB Himawari):**
- `GHI` — Global Horizontal Irradiance (W/m²).
- `DNI` — Direct Normal Irradiance (W/m²).
- `DHI` — Diffuse Horizontal Irradiance (W/m²).
- `Temperature` — Air Temperature (°C).
- `Wind Speed` — Wind Speed (m/s).
- `Relative Humidity` — Relative Humidity (%).

**Engineered Features:**
- `hour` — Extracted from the datetime index to model the diurnal cycle.
- `month` — Extracted from the datetime index to model annual variations.
- `is_wet_season` — Binary flag (1 or 0) encoding the Philippine monsoon/typhoon season (June through November).

## How to Run the Project

**1. Install Dependencies**
Ensure you have Python 3.8+ installed. Install the required libraries using the provided requirements file:
```bash
pip install -r requirements.txt
```
*(Core dependencies: `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `python-dotenv`, `requests`, `joblib`)*

**2. Configuration (.env setup)**
To download the raw weather data in `01_preprocessing.ipynb`, you need an API key from the National Renewable Energy Laboratory (NREL). Create a file named `.env` in the root directory and add your credentials:
```env
NREL_API_KEY=your_nrel_api_key_here
NREL_EMAIL=your_registered_email_here
```

**3. Execution Order**
The notebooks must be executed in strict sequential order to properly pass data through the pipeline:
1. Open and run all cells in `notebooks/01_preprocessing.ipynb`. This will download the raw CSVs into `data/raw/`, process the features, and save the scaled `.npy` arrays into `data/processed/`.
2. Open and run all cells in `notebooks/02_training.ipynb`. This will load the arrays, train the neural network, plot the training loss curves, and save the best model to the `models/` directory.
3. Open and run all cells in `notebooks/03_evaluation.ipynb`. This will load the saved model and scalers, run predictions on the test set, calculate metrics (RMSE, MAE, R²), and save the comparative graphs (scatter plots, 7-day overlays, residual plots) into `figures/`.

---

Aquino, Dallas A. <br>
Buñag, Frederick Jibril L. <br>
Carbonell, Ethan Jed V. <br>
Corpes, Vincent L. Jr. <br>

BSCS 3A AI
