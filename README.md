# Energy Forecasting with Transformer and LightGBM

This project addresses the forecasting of urban energy consumption using historical temperature and usage data from Chicago. Two model architectures are implemented and compared: a Transformer-based neural network and a LightGBM ensemble model.

## Project Overview

- **Goal**: Predict hourly energy consumption using time and temperature data.
- **Data Source**: Publicly available electricity and weather datasets from Chicago.
- **Methods**: Data preprocessing, feature engineering, model training and evaluation, interactive prediction via Streamlit dashboard.
- **Models Used**: Transformer (PyTorch) and LightGBM (scikit-learn interface).
- **Reproducibility**: Modular scripts and notebooks for each pipeline stage.

## Results

| Model       | RMSE     | MAE      | MAPE     |
|-------------|----------|----------|----------|
| Transformer | XX.XX    | XX.XX    | XX.XX %  |
| LightGBM    | XX.XX    | XX.XX    | XX.XX %  |

- The Transformer model performed better on long-term patterns and sequences.
- LightGBM was more robust to noise and required less computational effort.
- All results are documented in `lightgbm_model/results/` and `transformer_model/results/`.

## Live Demo

You can try the model predictions interactively in the Streamlit dashboard:

**🔗 [Launch Streamlit App](https://your-streamlit-url.streamlit.app)**

(Optional) Preview:

![Streamlit Dashboard Preview](assets/dashboard_preview.gif)

## Model Development

### Transformer

- Implemented in PyTorch with positional encoding and multi-head attention.
- Variants tested: embedding types, weight decay, dropout levels.
- Training performance and evaluation metrics are saved in the results folder.

### LightGBM

- Extensive hyperparameter tuning (e.g., `num_leaves`, `learning_rate`, `min_child_samples`).
- Feature importance used for selection and reduction.
- Evaluation includes learning curves and error plots.

## Data

- Raw consumption data: `data/raw/COMED_hourly.csv`
- Raw temperature data: `data/external/Temperature_chicago.csv`
- Merged and cleaned data: `data/processed/energy_consumption_aggregated_cleaned.csv`

To use your own dataset, ensure it contains the following columns:
timestamp, consumption, temperature

## Repository Structure

energy-forecasting-transformer-lightgbm/
│
├── data/
│ ├── raw/
│ ├── external/
│ └── processed/
│
├── notebooks/
│ ├── eda/
│ ├── lightgbm/
│ └── transformer/
│
├── scripts/
│ └── data_preprocessing/
│
├── lightgbm_model/
│ ├── model/
│ ├── results/
│ └── scripts/
│ ├── train/
│ └── eval/
│
├── transformer_model/
│ ├── model/
│ ├── results/
│ └── scripts/
│ ├── training/
│ ├── evaluation/
│ └── utils/
│
├── streamlit_simulation/
│ └── app.py
│
├── requirements.txt
├── requirements_lgbm.txt
├── setup.py
└── README.md


## Installation and Execution

### Prerequisites

- Python ≥ 3.9
- Recommended: virtual environment (`venv` or `conda`)

### Setup

git clone https://github.com/dlajic/energy-forecasting-transformer-lightgbm.git
cd energy-forecasting-transformer-lightgbm
pip install -r requirements.txt

### Preprocess the Data

python scripts/data_preprocessing/preprocess_data.py
## Train Models

### LightGBM
python lightgbm_model/scripts/train/train_lightgbm.py

### Transformer
python transformer_model/scripts/training/train.py

## Evaluate Models
python lightgbm_model/scripts/eval/eval_lightgbm.py
python transformer_model/scripts/evaluation/evaluate.py

## Run Streamlit Dashboard
streamlit run streamlit_simulation/app.py

## Reproducibility
All code is modular and executable with any dataset in the required format.

The pipeline can be restarted from preprocessing with new input data.

## Author
Dean Lajic
GitHub: https://github.com/dlajic
