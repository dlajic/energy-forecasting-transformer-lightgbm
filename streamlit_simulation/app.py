import sys
import os
import streamlit as st
import pickle
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import torch

from config_streamlit import (MODEL_PATH_LIGHTGBM, DATA_PATH, TRAIN_RATIO,
                             TEXT_COLOR, HEADER_COLOR, ACCENT_COLOR,
                             BUTTON_BG, BUTTON_HOVER_BG, BG_COLOR,
                             INPUT_BG, PROGRESS_COLOR, PLOT_COLOR
                             )
from lightgbm_model.scripts.config_lightgbm import FEATURES
from transformer_model.scripts.utils.informer_dataset_class import InformerDataset
from transformer_model.scripts.training.load_basis_model import load_moment_model
from transformer_model.scripts.config_transformer import CHECKPOINT_DIR, FORECAST_HORIZON, SEQ_LEN
from sklearn.preprocessing import StandardScaler


# ============================== Layout ==============================

# Streamlit & warnings config
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="Electricity Consumption Forecast", layout="wide")

#CSS part
st.markdown(f"""
    <style>
        body, .block-container {{
            background-color: {BG_COLOR} !important;
        }}

        html, body, [class*="css"] {{
            color: {TEXT_COLOR} !important;
            font-family: 'sans-serif';
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {HEADER_COLOR} !important;
        }}

        .stButton > button {{
            background-color: {BUTTON_BG};
            color: {TEXT_COLOR};
            border: 1px solid {ACCENT_COLOR};
        }}

        .stButton > button:hover {{
            background-color: {BUTTON_HOVER_BG};
        }}

        .stSelectbox div[data-baseweb="select"],
        .stDateInput input {{
            background-color: {INPUT_BG} !important;
            color: {TEXT_COLOR} !important;
        }}

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {{
            color: {TEXT_COLOR} !important;
        }}

        .stMarkdown p {{
            color: {TEXT_COLOR} !important;
        }}

        .stDataFrame tbody tr td {{
            color: {TEXT_COLOR} !important;
        }}

        .stProgress > div > div {{
            background-color: {PROGRESS_COLOR} !important;
        }}

        /* Alle Label-Texte für Inputs/Sliders */
        label {{
            color: {TEXT_COLOR} !important;
        }}

        /* Text in selectbox-Optionsfeldern */
        .stSelectbox label, .stSelectbox div {{
            color: {TEXT_COLOR} !important;
        }}

        /* DateInput angleichen an Selectbox */
        .stDateInput input {{
            background-color: #f2f6fa !important;
            color: {TEXT_COLOR} !important;
            border: none !important;
            border-radius: 5px !important;
        }}
        
    </style>
""", unsafe_allow_html=True)

st.title("Electricity Consumption Forecast: Hourly Simulation")
st.write("Welcome to the simulation interface!")

# ============================== Session State Init ==============================
def init_session_state():
    defaults = {
        "is_running": False,
        "start_index": 0,
        "true_vals": [],
        "pred_vals": [],
        "true_timestamps": [],
        "pred_timestamps": [],
        "last_fig": None,
        "valid_pos": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================== Loaders ==============================

@st.cache_data
def load_lightgbm_model():
    with open(MODEL_PATH_LIGHTGBM, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_transformer_model_and_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_moment_model()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Datasets
    train_dataset = InformerDataset(data_split="train", forecast_horizon=FORECAST_HORIZON, random_seed=13)
    test_dataset = InformerDataset(data_split="test", forecast_horizon=FORECAST_HORIZON, random_seed=13)
    test_dataset.scaler = train_dataset.scaler

    return model, test_dataset, device

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


# ============================== Utility Functions ==============================

def predict_transformer_step(model, dataset, idx, device):
    """Performs a single prediction step with the transformer model."""
    timeseries, _, input_mask = dataset[idx]
    timeseries = torch.tensor(timeseries, dtype=torch.float32).unsqueeze(0).to(device)
    input_mask = torch.tensor(input_mask, dtype=torch.bool).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x_enc=timeseries, input_mask=input_mask)

    pred = output.forecast[:, 0, :].cpu().numpy().flatten()

    # Rückskalieren
    dummy = np.zeros((len(pred), dataset.n_channels))
    dummy[:, 0] = pred
    pred_original = dataset.scaler.inverse_transform(dummy)[:, 0]

    return float(pred_original[0])


def init_simulation_layout():
    """Creates layout containers for plot and info sections."""
    col1, spacer, col2 = st.columns([3, 0.2, 1])
    plot_title = col1.empty()
    plot_container = col1.empty()
    x_axis_label = col1.empty()
    info_container = col2.empty()
    return plot_title, plot_container, x_axis_label, info_container


def create_prediction_plot(pred_timestamps, pred_vals, true_timestamps, true_vals, window_hours, y_min=None, y_max=None):
    """Generates the matplotlib figure for plotting prediction vs. actual."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True, facecolor=PLOT_COLOR)
    ax.set_facecolor(PLOT_COLOR)

    ax.plot(pred_timestamps[-window_hours:], pred_vals[-window_hours:], label="Prediction", color="#EF233C", linestyle="--")
    if true_vals:
        ax.plot(true_timestamps[-window_hours:], true_vals[-window_hours:], label="Actual", color="#0077B6")

    ax.set_ylabel("Consumption (MW)", fontsize=8, color=TEXT_COLOR)
    ax.legend(
    fontsize=8,
    loc="upper left",
    bbox_to_anchor=(0, 0.95),
    facecolor= INPUT_BG,    # INPUT_BG
    edgecolor= ACCENT_COLOR,    # ACCENT_COLOR
    labelcolor= TEXT_COLOR    # TEXT_COLOR
    )
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=5, colors=TEXT_COLOR)
    ax.tick_params(axis="y", labelsize=5, colors=TEXT_COLOR)
    #fig.patch.set_facecolor('#e6ecf0')  # outer area

    for spine in ax.spines.values():
        spine.set_visible(False)

    st.session_state.last_fig = fig
    return fig


def render_simulation_view(timestamp, prediction, actual, progress, fig, paused=False):
    """Displays the simulation plot and metrics in the UI."""
    title = "Actual vs. Prediction (Paused)" if paused else "Actual vs. Prediction"
    plot_title.markdown(
        f"<div style='text-align: center; font-size: 20pt; font-weight: bold; color: {TEXT_COLOR}; margin-bottom: -0.7rem; margin-top: 0rem;'>"
        f"{title}</div>",
        unsafe_allow_html=True
    )
    plot_container.pyplot(fig)

    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
    x_axis_label.markdown(
        f"<div style='text-align: center; font-size: 14pt; color: {TEXT_COLOR}; margin-top: -0.5rem;'>"
        f"Time</div>",
        unsafe_allow_html=True
    )

    with info_container.container():
        st.markdown("<div style='margin-top: 5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<span style='font-size: 24px; font-weight: 600; color: {HEADER_COLOR} !important;'>Time: {timestamp}</span>",
            unsafe_allow_html=True
        )

        st.metric("Prediction", f"{prediction:,.0f} MW" if prediction is not None else "–")
        st.metric("Actual", f"{actual:,.0f} MW" if actual is not None else "–")
        st.caption("Simulation Progress")
        st.progress(progress)

        if len(st.session_state.true_vals) > 1:
            true_arr = np.array(st.session_state.true_vals)
            pred_arr = np.array(st.session_state.pred_vals[:-1])

            min_len = min(len(true_arr), len(pred_arr)) #just start if there are 2 actual values
            if min_len >= 1:
                errors = np.abs(true_arr[:min_len] - pred_arr[:min_len])
                mape = np.mean(errors / np.where(true_arr[:min_len] == 0, 1e-10, true_arr[:min_len])) * 100
                mae = np.mean(errors)
                max_error = np.max(errors)

                st.divider()
                st.markdown(
                    f"<span style='font-size: 24px; font-weight: 600; color: {HEADER_COLOR} !important;'>Interim Metrics</span>",
                    unsafe_allow_html=True
                )
                st.metric("MAPE (so far)", f"{mape:.2f} %")
                st.metric("MAE (so far)", f"{mae:,.0f} MW")
                st.metric("Max Error", f"{max_error:,.0f} MW")



# ============================== Data Preparation ==============================

df_full = load_data()

# Split Train/Test
train_size = int(len(df_full) * TRAIN_RATIO)
test_df_raw = df_full.iloc[train_size:].reset_index(drop=True)

# Start at first full hour (00:00)
first_full_day_index = test_df_raw[test_df_raw["date"].dt.time == pd.Timestamp("00:00:00").time()].index[0]
test_df_full = test_df_raw.iloc[first_full_day_index:].reset_index(drop=True)

# Select simulation window via date picker
min_date = test_df_full["date"].min().date()
max_date = test_df_full["date"].max().date()

# ============================== UI Controls ==============================

st.markdown("### Simulation Settings")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**General Settings**")
    model_choice = st.selectbox("Choose prediction model", ["LightGBM", "Transformer Model (moments)"])
    if model_choice == "Transformer Model(moments)":
        st.caption("⚠️ Note: Transformer model runs slower without GPU. (Use Speed = 10)")
    window_days = st.selectbox("Display window (days)", options=[3, 5, 7], index=0)
    window_hours = window_days * 24
    speed = st.slider("Speed", 1, 10, 5)

with col2:
    st.markdown(f"**Date Range** (from {min_date} to {max_date})")
    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    

# ============================== Data Preparation (filtered) ==============================

# final filtered date window
test_df_filtered = test_df_full[
    (test_df_full["date"].dt.date >= start_date) &
    (test_df_full["date"].dt.date <= end_date)
].reset_index(drop=True)

# For progression bar
total_steps_ui = len(test_df_filtered)

# ============================== Buttons ==============================

st.markdown("### Start Simulation")
col1, col2, col3 = st.columns([1, 1, 14])
with col1:
    play_pause_text = "▶️ Start" if not st.session_state.is_running else "⏸️ Pause"
    if st.button(play_pause_text):
        st.session_state.is_running = not st.session_state.is_running
        st.rerun()
with col2:
    reset_button = st.button("🔄 Reset")

# Reset logic
if reset_button:
    st.session_state.start_index = 0
    st.session_state.pred_vals = []
    st.session_state.true_vals = []
    st.session_state.pred_timestamps = []
    st.session_state.true_timestamps = []
    st.session_state.last_fig = None
    st.session_state.is_running = False
    st.session_state.valid_pos = 0
    st.rerun()

# Auto-reset on critical parameter change while running
if st.session_state.is_running and (
    start_date != st.session_state.get("last_start_date") or
    end_date != st.session_state.get("last_end_date") or
    model_choice != st.session_state.get("last_model_choice")
):
    st.session_state.start_index = 0
    st.session_state.pred_vals = []
    st.session_state.true_vals = []
    st.session_state.pred_timestamps = []
    st.session_state.true_timestamps = []
    st.session_state.last_fig = None
    st.session_state.valid_pos = 0
    st.rerun()

# Track current selections for change detection
st.session_state.last_start_date = start_date
st.session_state.last_end_date = end_date
st.session_state.last_model_choice = model_choice


# ============================== Paused Mode ==============================

if not st.session_state.is_running and st.session_state.last_fig is not None:
    st.write("Simulation paused...")
    plot_title, plot_container, x_axis_label, info_container = init_simulation_layout()

    timestamp = st.session_state.pred_timestamps[-1] if st.session_state.pred_timestamps else "–"
    prediction = st.session_state.pred_vals[-1] if st.session_state.pred_vals else None
    actual = st.session_state.true_vals[-1] if st.session_state.true_vals else None
    progress = st.session_state.start_index / total_steps_ui

    render_simulation_view(timestamp, prediction, actual, progress, st.session_state.last_fig, paused=True)


# ============================== initialize values ==============================

#if lightGbm use testdata from above
if model_choice == "LightGBM":
        test_df = test_df_filtered.copy()

#Shared state references for storing predictions and ground truths

true_vals = st.session_state.true_vals
pred_vals = st.session_state.pred_vals
true_timestamps = st.session_state.true_timestamps
pred_timestamps = st.session_state.pred_timestamps

# ============================== LightGBM Simulation ==============================

if model_choice == "LightGBM" and st.session_state.is_running:
    model = load_lightgbm_model()
    st.write("Simulation started...")
    st.markdown('<div id="simulation"></div>', unsafe_allow_html=True)

    plot_title, plot_container, x_axis_label, info_container = init_simulation_layout()

    for i in range(st.session_state.start_index, len(test_df)):
        if not st.session_state.is_running:
            break

        current = test_df.iloc[i]
        timestamp = current["date"]
        features = current[FEATURES].values.reshape(1, -1)
        prediction = model.predict(features)[0]

        pred_vals.append(prediction)
        pred_timestamps.append(timestamp)

        if i >= 1:
            prev_actual = test_df.iloc[i - 1]["consumption_MW"]
            prev_time = test_df.iloc[i - 1]["date"]
            true_vals.append(prev_actual)
            true_timestamps.append(prev_time)

        fig = create_prediction_plot(
            pred_timestamps, pred_vals,
            true_timestamps, true_vals,
            window_hours,
            y_min= test_df_filtered["consumption_MW"].min() - 2000,
            y_max= test_df_filtered["consumption_MW"].max() + 2000
        )

        render_simulation_view(timestamp, prediction, prev_actual if i >= 1 else None, i / len(test_df), fig)

        plt.close(fig)  # Speicher freigeben

        st.session_state.start_index = i + 1
        time.sleep(1 / (speed + 1e-9))

    st.success("Simulation completed!")



# ============================== Transformer Simulation ==============================

if model_choice == "Transformer Model(moments)":
    if st.session_state.is_running:
        st.write("Simulation started (Transformer)...")
        st.markdown('<div id="simulation"></div>', unsafe_allow_html=True)

        plot_title, plot_container, x_axis_label, info_container = init_simulation_layout()

        # Zugriff auf Modell, Dataset, Device
        model, test_dataset, device = load_transformer_model_and_dataset()
        data = test_dataset.data  # bereits skaliert
        scaler = test_dataset.scaler
        n_channels = test_dataset.n_channels

        test_start_idx = len(InformerDataset(data_split="train", forecast_horizon=FORECAST_HORIZON)) + SEQ_LEN
        base_timestamp = pd.read_csv(DATA_PATH, parse_dates=["date"])["date"].iloc[test_start_idx] #get original timestamp for later, cause not in dataset anymore

        # Schritt 1: Finde Index, ab dem Stunde = 00:00 ist
        offset = 0
        while (base_timestamp + pd.Timedelta(hours=offset)).time() != pd.Timestamp("00:00:00").time():
            offset += 1
        
        # Neuer Startindex in der Simulation
        start_index = offset

        # Session-State bei Bedarf initial setzen
        if "start_index" not in st.session_state or st.session_state.start_index == 0:
            st.session_state.start_index = start_index


        # Vorbereiten: Liste der gültigen i-Werte im gewünschten Zeitraum
        valid_indices = []
        for i in range(start_index, len(test_dataset)):
            timestamp = base_timestamp + pd.Timedelta(hours=i)
            if start_date <= timestamp.date() <= end_date:
                valid_indices.append(i)

        # Fortschrittsanzeige
        total_steps = len(valid_indices)

        # Aktueller Fortschritt in der Liste (nicht: globaler Dataset-Index!)
        if "valid_pos" not in st.session_state:
            st.session_state.valid_pos = 0

        # Hauptschleife: Nur noch über gültige Indizes iterieren
        for relative_idx, i in enumerate(valid_indices[st.session_state.valid_pos:]):

        #for i in range(st.session_state.start_index, len(test_dataset)):
            if not st.session_state.is_running:
                break

            current_pred = predict_transformer_step(model, test_dataset, i, device)
            current_time = base_timestamp + pd.Timedelta(hours=i)

            pred_vals.append(current_pred)
            pred_timestamps.append(current_time)

            if i >= 1:
                prev_actual = test_dataset[i - 1][1][0, 0]  # erster Forecast-Wert der letzten Zeile
                # Rückskalieren
                dummy_actual = np.zeros((1, n_channels))
                dummy_actual[:, 0] = prev_actual
                actual_val = scaler.inverse_transform(dummy_actual)[0, 0]

                true_time = current_time - pd.Timedelta(hours=1)

                if true_time >= pd.to_datetime(start_date):
                    true_vals.append(actual_val)
                    true_timestamps.append(true_time)

            # Plot erzeugen
            fig = create_prediction_plot(
                pred_timestamps, pred_vals,
                true_timestamps, true_vals,
                window_hours,
                y_min= test_df_filtered["consumption_MW"].min() - 2000,
                y_max= test_df_filtered["consumption_MW"].max() + 2000
            )
            if len(pred_vals) >= 2 and len(true_vals) >= 1:
                render_simulation_view(current_time, current_pred, actual_val if i >= 1 else None, st.session_state.valid_pos / total_steps, fig)

            plt.close(fig)  # Speicher freigeben

            st.session_state.valid_pos += 1
            time.sleep(1 / (speed + 1e-9))

        st.success("Simulation completed!")


# ============================== Scroll Sync ==============================

st.markdown("""
    <script>
    window.addEventListener("message", (event) => {
        if (event.data.type === "save_scroll") {
            const pyScroll = event.data.scrollY;
            window.parent.postMessage({type: "streamlit:setComponentValue", value: pyScroll}, "*");
        }
    });
    </script>
""", unsafe_allow_html=True)

