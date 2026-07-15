from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =========================================================
# Streamlit page configuration
# =========================================================
st.set_page_config(
    page_title="Taxi Demand Dashboard",
    page_icon="🚖",
    layout="wide",
)

st.title("🚖 Taxi Demand Operational Performance Portal")
st.caption(
    "Live streaming metrics and continuous financial validation "
    "served through FastAPI and PostgreSQL."
)


# =========================================================
# Application configuration
# =========================================================
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://fastapi:8000",
)

DEFAULT_PICKUP_LOCATION_ID = int(
    os.getenv(
        "PICKUP_LOCATION_ID",
        "237",
    )
)


# =========================================================
# API helpers
# =========================================================
def fetch_json(url: str) -> dict | None:
    """
    Fetch JSON data from the FastAPI service.

    A cache-busting query parameter is added so Streamlit receives
    the latest API response during each rerun.
    """
    try:
        cache_buster = int(pd.Timestamp.now().timestamp())

        separator = "&" if "?" in url else "?"
        bust_url = f"{url}{separator}_cb={cache_buster}"

        response = requests.get(
            bust_url,
            timeout=5,
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None

    except ValueError as exc:
        st.error(f"API returned invalid JSON: {exc}")
        return None


# =========================================================
# Display helpers
# =========================================================
def class_label(value) -> str:
    """
    Convert numeric demand classes into readable labels.
    """
    if value is None or pd.isna(value):
        return "Pending Ground Truth"

    try:
        return {
            0: "Low Demand",
            1: "Normal Demand",
            2: "High Demand",
        }.get(
            int(value),
            str(value),
        )

    except (ValueError, TypeError):
        return str(value)


def format_penalty(value) -> str:
    """
    Format a penalty value for the metric cards.
    """
    if value is None or pd.isna(value):
        return "Awaiting Settlement"

    try:
        return f"{float(value):,.0f} Units"

    except (ValueError, TypeError):
        return "Awaiting Settlement"


# =========================================================
# Hourly operational chart
# =========================================================
def make_washboard(df: pd.DataFrame) -> go.Figure:
    """
    Build the hourly operational performance chart.

    Positive bars represent underprediction cost.
    Negative bars represent overprediction cost.
    """
    visible = df.copy()

    visible["prediction_hour_ts"] = pd.to_datetime(
        visible["prediction_hour_ts"],
        errors="coerce",
    )

    visible["actual_class_numeric"] = pd.to_numeric(
        visible["actual_class"],
        errors="coerce",
    )

    visible["predicted_class_numeric"] = pd.to_numeric(
        visible["predicted_class"],
        errors="coerce",
    )

    visible["model_penalty_numeric"] = pd.to_numeric(
        visible["model_penalty"],
        errors="coerce",
    )

    visible["underprediction_penalty"] = 0.0
    visible["overprediction_penalty"] = 0.0

    has_actual = (
        visible["actual_class_numeric"].notna()
        & visible["predicted_class_numeric"].notna()
        & visible["model_penalty_numeric"].notna()
    )

    underprediction = has_actual & (
        visible["actual_class_numeric"] > visible["predicted_class_numeric"]
    )

    overprediction = has_actual & (
        visible["predicted_class_numeric"] > visible["actual_class_numeric"]
    )

    visible.loc[
        underprediction,
        "underprediction_penalty",
    ] = visible.loc[
        underprediction,
        "model_penalty_numeric",
    ]

    visible.loc[
        overprediction,
        "overprediction_penalty",
    ] = -visible.loc[
        overprediction,
        "model_penalty_numeric",
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=visible["prediction_hour_ts"],
            y=visible["underprediction_penalty"],
            name="Underprediction Cost (Missed Revenue)",
            marker_color="#d62728",
            hovertemplate=(
                "<b>%{x|%b %d, %Y %I:%M %p}</b>"
                "<br>Missed Revenue Penalty: %{y:,.0f} units"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            x=visible["prediction_hour_ts"],
            y=visible["overprediction_penalty"],
            name="Overprediction Cost (Wasted Resources)",
            marker_color="#ff7f0e",
            hovertemplate=(
                "<b>%{x|%b %d, %Y %I:%M %p}</b>"
                "<br>Resource Waste Penalty: %{y:,.0f} units"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=visible["prediction_hour_ts"],
            y=visible["actual_class_numeric"],
            name="Actual Demand (Truth)",
            mode="lines+markers",
            yaxis="y2",
            line=dict(
                color="#1f77b4",
                width=3,
            ),
            line_shape="hv",
            hovertemplate=(
                "<b>%{x|%b %d, %Y %I:%M %p}</b>"
                "<br>Actual Demand Tier: %{y:.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=visible["prediction_hour_ts"],
            y=visible["predicted_class_numeric"],
            name="Model Prediction",
            mode="lines+markers",
            yaxis="y2",
            line=dict(
                color="#9467bd",
                width=2,
                dash="dot",
            ),
            line_shape="hv",
            hovertemplate=(
                "<b>%{x|%b %d, %Y %I:%M %p}</b>"
                "<br>Predicted Demand Tier: %{y:.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="Hourly Operational Fleet Friction Log",
        barmode="relative",
        xaxis_title="Prediction Hour",
        yaxis=dict(
            title="Penalty Units",
            zeroline=True,
        ),
        yaxis2=dict(
            title="Demand Tier (0=Low, 1=Normal, 2=High)",
            overlaying="y",
            side="right",
            range=[-0.3, 2.3],
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=["Low", "Normal", "High"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(
            t=100,
            r=80,
        ),
        hovermode="x unified",
        height=500,
    )

    return fig


# =========================================================
# Cumulative operational cost chart
# =========================================================
def make_cumulative_cost(df: pd.DataFrame) -> go.Figure:
    """
    Compare cumulative ML-model penalties against the naive baseline.
    """
    visible = df.copy()

    visible["prediction_hour_ts"] = pd.to_datetime(
        visible["prediction_hour_ts"],
        errors="coerce",
    )

    visible["model_penalty_numeric"] = pd.to_numeric(
        visible["model_penalty"],
        errors="coerce",
    )

    visible["baseline_penalty_numeric"] = pd.to_numeric(
        visible["baseline_penalty"],
        errors="coerce",
    )

    settled_mask = (
        visible["model_penalty_numeric"].notna()
        & visible["baseline_penalty_numeric"].notna()
    )

    settled = visible.loc[settled_mask].copy()

    if settled.empty:
        plot_data = visible.copy()

        plot_data["model_penalty_numeric"] = 0.0
        plot_data["baseline_penalty_numeric"] = 0.0

        subtitle_text = (
            "⏳ Performance comparison is awaiting settled ground-truth records."
        )
        subtitle_color = "grey"

    else:
        plot_data = settled

        plot_data["model_penalty_numeric"] = plot_data["model_penalty_numeric"].fillna(
            0.0
        )

        plot_data["baseline_penalty_numeric"] = plot_data[
            "baseline_penalty_numeric"
        ].fillna(0.0)

        final_model_cost = float(plot_data["model_penalty_numeric"].sum())

        final_baseline_cost = float(plot_data["baseline_penalty_numeric"].sum())

        net_savings = final_baseline_cost - final_model_cost

        if final_baseline_cost > 0:
            cost_change_pct = (net_savings / final_baseline_cost) * 100

            if cost_change_pct > 0:
                subtitle_text = (
                    "🔥 The ML model reduced operational cost by "
                    f"<b>{cost_change_pct:.1f}%</b> compared with the "
                    "naive baseline over the selected evaluation period."
                )
                subtitle_color = "#2ca02c"

            elif cost_change_pct < 0:
                subtitle_text = (
                    "⚠️ The ML model increased operational cost by "
                    f"<b>{abs(cost_change_pct):.1f}%</b> compared with "
                    "the naive baseline over the selected evaluation period."
                )
                subtitle_color = "#d62728"

            else:
                subtitle_text = (
                    "⚖️ The ML model matched the naive baseline over "
                    "the selected evaluation period."
                )
                subtitle_color = "grey"

        elif final_model_cost == 0:
            subtitle_text = (
                "⚖️ Neither system incurred an operational penalty "
                "over the selected evaluation period."
            )
            subtitle_color = "grey"

        else:
            subtitle_text = (
                "⚠️ The ML model incurred operational penalties while "
                "the naive baseline incurred none."
            )
            subtitle_color = "#d62728"

    plot_data["model_cumulative"] = plot_data["model_penalty_numeric"].cumsum()

    plot_data["baseline_cumulative"] = plot_data["baseline_penalty_numeric"].cumsum()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_data["prediction_hour_ts"],
            y=plot_data["baseline_cumulative"],
            name="Naive Baseline",
            mode="lines",
            line=dict(
                color="grey",
                width=2,
                dash="dash",
            ),
            hovertemplate=(
                "<b>%{x|%b %d, %Y %I:%M %p}</b>"
                "<br>Baseline Cost: %{y:,.0f} units"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_data["prediction_hour_ts"],
            y=plot_data["model_cumulative"],
            name="ML Model",
            mode="lines",
            line=dict(
                color="#d62728",
                width=3,
            ),
            hovertemplate=(
                "<b>%{x|%b %d, %Y %I:%M %p}</b>"
                "<br>ML Model Cost: %{y:,.0f} units"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=(
                "Cumulative Operational Cost: ML Model vs. Naive Baseline"
                f"<br><span style='font-size:13px; "
                f"color:{subtitle_color};'>"
                f"{subtitle_text}"
                "</span>"
            ),
            x=0,
            xanchor="left",
            y=0.96,
        ),
        xaxis_title="Prediction Hour",
        yaxis_title="Total Penalty Units",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(
            t=110,
        ),
        hovermode="x unified",
        height=440,
    )

    return fig


# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("🎛️ Evaluation Controls")

pickup_location_id = st.sidebar.number_input(
    "Pickup Location ID",
    min_value=0,
    value=DEFAULT_PICKUP_LOCATION_ID,
    step=1,
)

if pickup_location_id == 0:
    st.info("Enter the pickup location ID used by the trained model.")
    st.stop()

lookback_option = st.sidebar.selectbox(
    "Evaluation Lookback Horizon",
    options=[
        "All History",
        "7 Days",
        "10 Days",
        "14 Days",
    ],
    index=0,
    key="lookback_horizon_select",
)


# =========================================================
# Load prediction data
# =========================================================
payload = fetch_json(f"{API_BASE_URL}/predictions/{pickup_location_id}")

if not payload or payload.get("count", 0) == 0:
    st.warning(
        "No predictions were returned. Check the Airflow scoring "
        "pipeline and FastAPI service."
    )
    st.stop()

predictions = payload.get("predictions", [])

# OPTIMIZATION FIX: Parse JSON records safely to prevent Arrow thread stalls
df = pd.DataFrame.from_records(predictions)

if df.empty:
    st.warning("The API returned an empty prediction timeline.")
    st.stop()


# =========================================================
# Validate required columns
# =========================================================
required_columns = {
    "prediction_hour_ts",
    "predicted_class",
    "actual_class",
    "model_penalty",
    "baseline_penalty",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    st.error(
        "The prediction API response is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()


# =========================================================
# Prepare and sort prediction data
# =========================================================
df["prediction_hour_ts"] = pd.to_datetime(
    df["prediction_hour_ts"],
    errors="coerce",
)

df = (
    df.dropna(subset=["prediction_hour_ts"])
    .sort_values("prediction_hour_ts")
    .reset_index(drop=True)
)

if df.empty:
    st.warning("No records contain a valid prediction timestamp.")
    st.stop()


# =========================================================
# Apply the selected lookback window
# =========================================================
if lookback_option != "All History":
    lookback_days = int(lookback_option.split(" ")[0])

    latest_timestamp = df["prediction_hour_ts"].max()

    start_horizon_ts = latest_timestamp - pd.Timedelta(days=lookback_days)

    df = df.loc[df["prediction_hour_ts"] >= start_horizon_ts].reset_index(drop=True)

if df.empty:
    st.warning("No prediction records are available for the selected lookback window.")
    st.stop()


# =========================================================
# Use the latest record in the selected window
# =========================================================
row = df.iloc[-1]

st.subheader(f"Latest Prediction: {row['prediction_hour_ts']:%B %d, %Y at %I:%M %p}")

st.caption(
    f"Showing {len(df):,} prediction records for pickup location "
    f"{pickup_location_id} within: {lookback_option}."
)


# =========================================================
# Current prediction metrics
# =========================================================
predicted_class_value = pd.to_numeric(
    pd.Series([row.get("predicted_class")]),
    errors="coerce",
).iloc[0]

if pd.notna(predicted_class_value):
    predicted_class_int = int(predicted_class_value)
    probability_column = f"prob_class_{predicted_class_int}"
    confidence_value = row.get(probability_column)
else:
    confidence_value = None

try:
    confidence_numeric = float(confidence_value)
    confidence_string = f"{confidence_numeric:.2%}"
except (TypeError, ValueError):
    confidence_string = None

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Predicted State",
    class_label(row.get("predicted_class")),
    (f"Confidence: {confidence_string}" if confidence_string else None),
)

col2.metric(
    "Actual Observed State",
    class_label(row.get("actual_class")),
)

col3.metric(
    "ML Model Loss",
    format_penalty(row.get("model_penalty")),
)

col4.metric(
    "Naive Baseline Loss",
    format_penalty(row.get("baseline_penalty")),
)


# =========================================================
# Charts
# =========================================================
st.markdown("---")

chart_key_suffix = lookback_option.lower().replace(" ", "_")

st.plotly_chart(
    make_washboard(df),
    width="stretch",
    key=f"washboard_{chart_key_suffix}",
)

st.plotly_chart(
    make_cumulative_cost(df),
    width="stretch",
    key=f"cumulative_cost_{chart_key_suffix}",
)


# =========================================================
# Raw data
# =========================================================
with st.expander("🔍 Latest Raw Data Audit Record"):
    st.json(row.to_dict())
