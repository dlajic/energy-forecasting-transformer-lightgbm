
# ⚡️ Energy Forecasting with Transformer & LightGBM

Ein Machine-Learning-Projekt zur Vorhersage des städtischen Energieverbrauchs auf Basis von Wetter- und Verbrauchsdaten aus Chicago. Zwei Modellansätze wurden implementiert und verglichen: ein Transformer-Modell (Deep Learning) und ein LightGBM-Modell (Gradient Boosting).

---

## 📸 Live-Demo

> ✨ **Live-Dashboard**:  
> 👉 [Streamlit App starten](https://your-deployment-link.com)  
>  
> ![Demo](assets/demo.gif)

---

## 📂 Projektüberblick

- **Ziel**: Zeitreihen-Vorhersage des Energieverbrauchs auf Basis von Temperaturdaten.
- **Daten**: Öffentliche Verbrauchs- und Temperaturdaten aus Chicago.
- **Modelle**: Transformer (PyTorch) & LightGBM (Scikit-Learn).
- **Workflow**: Datenaufbereitung → Feature Engineering → Modelltraining → Evaluation → Deployment mit Streamlit.

---

## 📈 Ergebnisse

| Modell      | RMSE     | MAE      | MAPE    |
|-------------|----------|----------|---------|
| Transformer | XX.XX    | XX.XX    | XX.X %  |
| LightGBM    | XX.XX    | XX.XX    | XX.X %  |

- Transformer zeigt Vorteile bei langfristigen Trends.
- LightGBM ist robuster bei begrenzten Daten und schneller im Training.

> 📊 Siehe Vergleichsplots in `lightgbm_model/results/` und `transformer_model/results/`

---

## 🧠 Modellarchitektur & Parameterwahl

### Transformer
- Multi-Head Attention, Positional Encoding, Dropout
- Getestete Varianten: Embedding-Typen, Weight Decay, Dropout-Raten
- Trainings-Monitoring: `training_plot.png`, Metriken als JSON

### LightGBM
- Extensive Gridsearch: `num_leaves`, `min_child_samples`, `learning_rate`, u.v.m.
- Feature Importance zur Reduktion irrelevanter Features
- Learning Curves, Fehlerverläufe dokumentiert

---

## 🧾 Datengrundlage

- Verbrauchsdaten: `data/raw/COMED_hourly.csv`
- Temperaturdaten: `data/external/Temperature_chicago.csv`
- Vorverarbeitet in: `data/processed/energy_consumption_aggregated_cleaned.csv`

> CSV-Format für eigene Daten:  
> `timestamp, consumption, temperature`

---

## 🗂️ Ordnerstruktur

```bash
energy-forecasting-transformer-lightgbm/
│
├── data/                     # Roh-, externe und bereinigte Daten
│   ├── raw/
│   ├── external/
│   └── processed/
│
├── notebooks/               # EDA & Modellprototyping (LightGBM + Transformer)
│   ├── eda/
│   ├── lightgbm/
│   └── transformer/
│
├── scripts/                 # Preprocessing & Konfiguration
│   ├── data_preprocessing/
│   └── config_main.py
│
├── lightgbm_model/          # LightGBM Modell, Skripte, Ergebnisse
│   ├── model/
│   ├── results/
│   └── scripts/
│       ├── train/
│       └── eval/
│
├── transformer_model/       # Transformer-Modell inkl. Training/Eval/Utils
│   ├── model/
│   ├── results/
│   └── scripts/
│       ├── training/
│       ├── evaluation/
│       └── utils/
│
├── streamlit_simulation/    # Streamlit App zur Modellvorhersage
│   └── app.py
│
├── requirements.txt
├── requirements_lgbm.txt
├── setup.py
└── README.md
```

---

## 🚀 Installation & Ausführung

### Voraussetzungen

- Python ≥ 3.9
- Virtuelle Umgebung empfohlen (z. B. `venv` oder `conda`)

### Setup

```bash
git clone https://github.com/dlajic/energy-forecasting-transformer-lightgbm.git
cd energy-forecasting-transformer-lightgbm
pip install -r requirements.txt
```

### Datenvorverarbeitung

```bash
python scripts/data_preprocessing/preprocess_data.py
```

### Modelltraining

```bash
# LightGBM
python lightgbm_model/scripts/train/train_lightgbm.py

# Transformer
python transformer_model/scripts/training/train.py
```

### Evaluation

```bash
python lightgbm_model/scripts/eval/eval_lightgbm.py
python transformer_model/scripts/evaluation/evaluate.py
```

### Streamlit App starten

```bash
streamlit run streamlit_simulation/app.py
```

---

## 🧪 Reproduzierbarkeit

- Alle Schritte sind modular und automatisiert aufrufbar.
- Eigene Daten im CSV-Format (`timestamp, consumption, temperature`) können direkt über das Preprocessing-Skript genutzt werden.

---

## 🔭 Ausblick

- [ ] Weitere Städte / Klimazonen integrieren
- [ ] Modell-Deployment als REST API (z. B. mit FastAPI)
- [ ] Automatisierte Hyperparameteroptimierung (Optuna)

---

## 👤 Autor

**Damir Lajic**  
[GitHub](https://github.com/dlajic) · [LinkedIn](https://www.linkedin.com/in/dein-name)
