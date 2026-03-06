import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("RUL Simulation Dashboard")

# ==========================================================
# 1️⃣ LOAD DATA
# ==========================================================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

# ==========================================================
# 2️⃣ PREPROCESS
# ==========================================================
required_cols = ["timestamp", "PREDICTED_THICKNESS", "PREDICTED_CR"]

for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ----- Robust SHUTDOWN handling -----
if "SHUTDOWN" in df.columns:
    df["SHUTDOWN"] = df["SHUTDOWN"].astype(str).str.strip().str.upper()
    df["SHUTDOWN"] = df["SHUTDOWN"].isin(["TRUE", "1", "YES"])
else:
    df["SHUTDOWN"] = False

# ==========================================================
# 3️⃣ PREDICTION VARIABLES PLOT
# ==========================================================
st.subheader("Prediction Variables Trend")

prediction_variables = [
    col for col in [
        "PREDICTED_THICKNESS",
        "PREDICTED_CR",
        "PREDICTED_CR_DERIVATIVE",
    ] if col in df.columns
]

selected_pred = st.multiselect(
    "Select variables:",
    prediction_variables,
    default=["PREDICTED_THICKNESS"],
    key="pred_select"
)

col1, col2 = st.columns(2)
normalize_flag = col1.toggle("Normalize (0-1)", value=False, key="normalize")
hide_shutdown = col2.toggle("Hide Shutdown Period", value=False, key="hide_shutdown")

plot_df = df.copy()
if hide_shutdown:
    plot_df = plot_df[~plot_df["SHUTDOWN"]]

def normalize(series):
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return series * 0
    return (series - min_val) / (max_val - min_val)

fig = go.Figure()

for var in selected_pred:
    if var not in plot_df.columns:
        continue

    y = plot_df[var]
    if normalize_flag:
        y = normalize(y)

    axis = "y1" if "THICKNESS" in var else "y2"

    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"],
        y=y,
        mode="lines",
        name=var,
        yaxis=axis
    ))

if not hide_shutdown:
    shutdown_df = df[df["SHUTDOWN"]]

    if not shutdown_df.empty:
        y_shutdown = shutdown_df["PREDICTED_THICKNESS"]
        if normalize_flag:
            y_shutdown = normalize(y_shutdown)

        fig.add_trace(go.Scatter(
            x=shutdown_df["timestamp"],
            y=y_shutdown,
            mode="markers",
            marker=dict(symbol="x", size=8),
            name="Shutdown",
            yaxis="y1"
        ))

fig.update_layout(
    xaxis=dict(title="Date"),
    yaxis=dict(title="Thickness"),
    yaxis2=dict(title="Corrosion Metrics", overlaying="y", side="right"),
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 🔹 PROCESS VARIABLES PLOT
# ==========================================================
st.subheader("Process Variables Trend")

exclude_cols = [
    "timestamp",
    "PREDICTED_THICKNESS",
    "PREDICTED_CR",
    "PREDICTED_CR_DERIVATIVE",
    "SHUTDOWN"
]
process_variables = [
    col for col in df.columns
    if col not in exclude_cols
]

if len(process_variables) == 0:
    st.info("No process variable columns found in dataset.")
else:

    selected_proc = st.multiselect(
        "Select process variables:",
        process_variables,
        default=process_variables[:2],
        key="proc_select"
    )

    col1, col2 = st.columns(2)
    normalize_proc = col1.toggle("Normalize (0-1)", value=False, key="normalize_proc")
    hide_shutdown_proc = col2.toggle("Hide Shutdown Period", value=False, key="hide_shutdown_proc")

    proc_df = df.copy()
    if hide_shutdown_proc:
        proc_df = proc_df[~proc_df["SHUTDOWN"]]

    fig_proc = go.Figure()

    for var in selected_proc:
        y = proc_df[var]

        if normalize_proc:
            y = normalize(y)

        if "EE-" in var:
            axis = "y2"
        else:
            axis = "y1"

        fig_proc.add_trace(go.Scatter(
            x=proc_df["timestamp"],
            y=y,
            mode="lines",
            name=var,
            yaxis=axis
        ))

    fig_proc.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Process Variables"),
        yaxis2=dict(title="Power Signals", overlaying="y", side="right"),
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig_proc, use_container_width=True)

# ==========================================================
# 4️⃣ CR STATISTICS
# ==========================================================
st.subheader("Corrosion Rate Statistics")

window = st.slider("Select CR Window Size", 10, 200, 12, key="cr_window")

recent_cr = df["PREDICTED_CR"].dropna().tail(window)

if recent_cr.empty:
    st.warning("Not enough CR data.")
    st.stop()

mean_cr = -abs(recent_cr.mean())
std_cr = abs(recent_cr.std())

cr_worst = mean_cr - 1.96 * std_cr
cr_best = mean_cr + 1.96 * std_cr

c1, c2 = st.columns(2)
c1.metric("Mean CR", f"{mean_cr:.6f}")
c2.metric("Std Dev CR", f"{std_cr:.6f}")

# ==========================================================
# 5️⃣ RUL SIMULATION
# ==========================================================
st.subheader("Remaining Useful Life Simulation")

min_thickness = st.number_input("Minimum Thickness Allowance", value=0.0)

last_thickness = df["PREDICTED_THICKNESS"].dropna().iloc[-1]
last_timestamp = df["timestamp"].iloc[-1]

DAYS_PER_YEAR = 365.25


# -------------------------------
# RUL Function
# -------------------------------
def calculate_rul_days(cr):
    cr = abs(cr)
    remaining = last_thickness - min_thickness

    if cr <= 0:
        return np.inf
    if remaining <= 0:
        return 0

    loss_per_day = (cr / 10) * 60 * 24
    return remaining / loss_per_day if loss_per_day > 0 else np.inf


if st.button("Run Forecast Simulation"):

    rul_mean  = calculate_rul_days(mean_cr)
    rul_worst = calculate_rul_days(cr_worst)
    rul_best  = calculate_rul_days(cr_best)

    # -------------------------------
    # Forecast Horizon (covers best case)
    # -------------------------------
    max_days = (
        int(max(rul_best, rul_worst) * 1.1)
        if np.isfinite(max(rul_best, rul_worst))
        else 365
    )
    max_days = max(max_days, 30)

    future_days = np.linspace(0, max_days, 300)
    future_dates = last_timestamp + pd.to_timedelta(future_days, unit="D")

    def forecast(cr):
        loss = (abs(cr) / 10) * 60 * 24
        return last_thickness - loss * future_days

    thk_mean  = forecast(mean_cr)
    thk_worst = forecast(cr_worst)
    thk_best  = forecast(cr_best)

    upper = np.maximum(thk_best, thk_worst)
    lower = np.minimum(thk_best, thk_worst)

    # -------------------------------
    # Plot
    # -------------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["PREDICTED_THICKNESS"],
        mode="lines",
        name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=thk_mean,
        mode="lines",
        line=dict(dash="dash"),
        name="Mean Forecast"
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        fillcolor="rgba(200,0,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Band",
        hoverinfo="skip"
    ))

    fig.add_hline(
        y=min_thickness,
        line_dash="dot",
        annotation_text="Minimum Thickness"
    )

    fig.update_layout(
        xaxis=dict(
            title="Date",
            type="date",
            range=[df["timestamp"].min(), future_dates[-1]]
        ),
        yaxis=dict(title="Thickness"),
        hovermode="x unified",
        height=500
    )

    st.subheader("Forecast Thickness Projection")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # RUL Display (Days + Years)
    # -------------------------------
    st.subheader("RUL Estimation")

    def fmt(days):
        if np.isfinite(days) and days > 0:
            return f"{days:.1f} days", f"{days/DAYS_PER_YEAR:.2f} years"
        if days == 0:
            return "0 days", "Failure reached"
        return "∞", "No failure"

    col1, col2, col3 = st.columns(3)

    col1.metric("Worst Case", *fmt(rul_worst))
    col2.metric("Mean Case",  *fmt(rul_mean))
    col3.metric("Best Case",  *fmt(rul_best))

    # -------------------------------
    # Failure Dates
    # -------------------------------
    if np.isfinite(rul_mean) and rul_mean > 0:
        st.subheader("Failure Dates Estimation")

        d1, d2, d3 = st.columns(3)
        d1.metric("Worst Case Date",
                  (last_timestamp + pd.Timedelta(days=rul_worst)).strftime("%Y-%m-%d"))
        d2.metric("Mean Case Date",
                  (last_timestamp + pd.Timedelta(days=rul_mean)).strftime("%Y-%m-%d"))
        d3.metric("Best Case Date",
                  (last_timestamp + pd.Timedelta(days=rul_best)).strftime("%Y-%m-%d"))

    elif rul_mean == 0:
        st.error("Component already below minimum thickness!")
    else:
        st.success("No predicted failure (CR ≈ 0)")