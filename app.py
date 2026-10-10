
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def datetime_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Predictive AI | Industrial Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PROFESSIONAL THEME ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F4F6FB;
    color: #263453;
    font-family: Inter, sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#172554,#263B70);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
.block-container {
    padding-top: 2rem;
    max-width: 1500px;
}
h1, h2, h3 {
    color: #243B70 !important;
    font-weight: 750;
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE3F0;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 14px rgba(30,50,90,.05);
}
[data-testid="stMetricLabel"] {
    color: #687792 !important;
}
[data-testid="stMetricValue"] {
    color: #243B70 !important;
    font-weight: 800 !important;
}
.stButton > button {
    background: #334EAC;
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}
.stButton > button:hover {
    background: #243B70;
    color: white;
}
.stDownloadButton > button {
    background: #E8EDF8;
    color: #243B70;
    border-radius: 9px;
}
.hero {
    background: linear-gradient(115deg,#172554,#405CC5);
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 25px;
}
.hero h1 {
    color: white !important;
    font-size: 32px;
}
.hero p {
    color: #DCE6FF !important;
}
.eyebrow {
    color: #C1D0FF;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}
.ok {
    background: #E5F5EE;
    border-left: 5px solid #29966D;
    padding: 16px;
    border-radius: 8px;
    color: #176746;
    font-weight: 600;
}
.risk {
    background: #FFF0EB;
    border-left: 5px solid #D65D48;
    padding: 16px;
    border-radius: 8px;
    color: #963B2B;
    font-weight: 600;
}
.footer {
    text-align: center;
    color: #8792A7;
    font-size: 12px;
    padding: 25px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATASET ----------------
DATA_FILE = Path(__file__).parent / "ai4i2020.csv"

@st.cache_data
def load_data():
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)

    # Fallback demo data if CSV is unavailable
    rng = np.random.default_rng(42)
    n = 1000

    air = rng.normal(300, 2, n)
    process = air + rng.normal(10, 1.5, n)
    rpm = rng.integers(1200, 2800, n)
    torque = rng.uniform(10, 70, n)
    wear = rng.integers(0, 250, n)

    failure = (
        (torque > 60) |
        (wear > 220) |
        (rpm < 1300)
    ).astype(int)

    return pd.DataFrame({
        "UDI": range(1, n + 1),
        "Product ID": [f"M-{i:05d}" for i in range(1, n + 1)],
        "Type": rng.choice(["L", "M", "H"], n),
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Machine failure": failure
    })


df = load_data()

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

missing = [
    col for col in FEATURES + [TARGET]
    if col not in df.columns
]

if missing:
    st.error("Dataset columns are missing.")
    st.write("Missing columns:", missing)
    st.write("Available columns:", list(df.columns))
    st.stop()

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET], errors="coerce")

valid = X.notna().all(axis=1) & y.notna()
X = X.loc[valid]
y = y.loc[valid].astype(int)

if len(X) < 10 or y.nunique() < 2:
    st.error("Not enough valid data to train the model.")
    st.stop()


# ---------------- TRAIN RANDOM FOREST ----------------
@st.cache_resource
def build_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    return classifier, accuracy, matrix


model, accuracy, cm = build_model(X, y)


# ---------------- SIDEBAR ----------------
st.sidebar.markdown("""
<div style="padding:12px 0 24px 0">
    <div style="font-size:38px">⚙️</div>
    <div style="font-size:25px;font-weight:800">Predictive AI</div>
    <div style="font-size:11px;letter-spacing:1px">
        INDUSTRIAL INTELLIGENCE
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Overview",
        "AI Prediction",
        "Machine Records",
        "Analytics",
        "Maintenance Alerts",
        "Project Details"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### SYSTEM STATUS")
st.sidebar.success("Application Running")
st.sidebar.caption(f"Dataset: {len(df):,} records")
st.sidebar.caption("Algorithm: Random Forest")
st.sidebar.caption("Version: 1.0")


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">INDUSTRIAL INTELLIGENCE PLATFORM</div>
    <h1>⚙️ AI-Based Predictive Maintenance</h1>
    <p>Machine monitoring • Failure prediction • Data analytics</p>
</div>
""", unsafe_allow_html=True)

total = len(df)
failure_count = int((df[TARGET] == 1).sum())
normal_count = total - failure_count
failure_rate = failure_count / total * 100 if total else 0


# ==================================================
# OVERVIEW
# ==================================================
if page == "Overview":

    st.subheader("Executive Dashboard")
    st.caption("Overview of historical machine performance data.")

    a, b, c, d = st.columns(4)

    a.metric("TOTAL RECORDS", f"{total:,}")
    b.metric("FAILURE RECORDS", f"{failure_count:,}")
    c.metric("NORMAL RECORDS", f"{normal_count:,}")
    d.metric("FAILURE RATE", f"{failure_rate:.2f}%")

    st.write("")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("### Failure Distribution")

            chart_data = pd.Series({
                "Normal": normal_count,
                "Failure": failure_count
            })

            st.bar_chart(chart_data, color="#405CC5")

    with right:
        with st.container(border=True):
            st.markdown("### Machine Types")

            if "Type" in df.columns:
                st.bar_chart(
                    df["Type"].value_counts(),
                    color="#7385BB"
                )
            else:
                st.info("Machine type data is unavailable.")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("### AI Model Performance")
            st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
            st.caption(
                "Measured on a held-out test dataset. "
                "Accuracy alone does not describe every model error."
            )

    with right:
        with st.container(border=True):
            st.markdown("### Recent Machine Records")
            st.dataframe(
                df.tail(5),
                use_container_width=True,
                hide_index=True
            )

    st.info(
        "This project uses historical data and is not connected "
        "to live industrial sensors."
    )


# ==================================================
# AI PREDICTION
# ==================================================
elif page == "AI Prediction":

    st.subheader("Machine Failure Risk Assessment")

    st.write(
        "Enter machine readings and click the button "
        "to get the model's prediction."
    )

    with st.container(border=True):

        st.markdown("### Sensor Input Panel")

        c1, c2, c3 = st.columns(3)

        with c1:
            air = st.number_input(
                "Air Temperature (K)",
                min_value=280.0,
                max_value=320.0,
                value=300.0
            )

            process = st.number_input(
                "Process Temperature (K)",
                min_value=280.0,
                max_value=330.0,
                value=310.0
            )

        with c2:
            rpm = st.number_input(
                "Rotational Speed (rpm)",
                min_value=500,
                max_value=4000,
                value=1500
            )

            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=100.0,
                value=40.0
            )

        with c3:
            wear = st.number_input(
                "Tool Wear (min)",
                min_value=0,
                max_value=300,
                value=100
            )

            run = st.button(
                "Run AI Prediction",
                use_container_width=True
            )

    if run:

        sample = pd.DataFrame([{
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear
        }])

        prediction = int(model.predict(sample)[0])

        probabilities = model.predict_proba(sample)[0]
        failure_probability = float(
            probabilities[list(model.classes_).index(1)]
        ) * 100

        st.markdown("### Prediction Result")

        left, right = st.columns(2)

        with left:
            if prediction == 1:
                st.markdown(
                    '<div class="risk">⚠️ Potential failure predicted</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="ok">✓ Normal operation predicted</div>',
                    unsafe_allow_html=True
                )

        with right:
            st.metric(
                "Estimated Failure Probability",
                f"{failure_probability:.1f}%"
            )

            st.progress(
                int(np.clip(round(failure_probability), 0, 100))
            )

        st.caption(
            "This is a model estimate, not a guarantee of actual machine "
            "condition. Verify important decisions with maintenance staff."
        )

        st.session_state["last_prediction"] = {
            "Time": datetime_now(),
            "Air Temperature": air,
            "Process Temperature": process,
            "Rotational Speed": rpm,
            "Torque": torque,
            "Tool Wear": wear,
            "Prediction": prediction,
            "Failure Probability (%)": round(failure_probability, 2)
        }

    if st.session_state.get("last_prediction"):
        st.markdown("### Latest Prediction")
        st.json(st.session_state["last_prediction"])


# ==================================================
# MACHINE RECORDS
# ==================================================
elif page == "Machine Records":

    st.subheader("Machine Data Explorer")

    query = st.text_input(
        "Search machine records",
        placeholder="Search by machine ID, temperature, status..."
    )

    shown = df.copy()

    if query:
        mask = shown.astype(str).apply(
            lambda col: col.str.contains(
                query,
                case=False,
                na=False
            )
        ).any(axis=1)

        shown = shown.loc[mask]

    st.caption(f"Showing {len(shown):,} of {len(df):,} records")

    st.dataframe(
        shown,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Records as CSV",
        data=shown.to_csv(index=False).encode("utf-8"),
        file_name="machine_records.csv",
        mime="text/csv"
    )

    st.divider()
    st.markdown("### Add a Demo Machine Record")

    with st.form("add_record_form"):

        c1, c2 = st.columns(2)

        with c1:
            new_air = st.number_input(
                "Air Temperature (K)",
                value=300.0,
                key="new_air"
            )

            new_process = st.number_input(
                "Process Temperature (K)",
                value=310.0,
                key="new_process"
            )

            new_rpm = st.number_input(
                "Rotational Speed (rpm)",
                min_value=0,
                value=1500,
                key="new_rpm"
            )

        with c2:
            new_torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                value=40.0,
                key="new_torque"
            )

            new_wear = st.number_input(
                "Tool Wear (min)",
                min_value=0,
                value=100,
                key="new_wear"
            )

            new_failure = st.selectbox(
                "Recorded Failure Status",
                [0, 1],
                key="new_failure"
            )

        add_record = st.form_submit_button("Add Record")

    if add_record:

        new_row = {
            "Air temperature [K]": new_air,
            "Process temperature [K]": new_process,
            "Rotational speed [rpm]": new_rpm,
            "Torque [Nm]": new_torque,
            "Tool wear [min]": new_wear,
            "Machine failure": new_failure
        }

        st.session_state.setdefault("demo_records", [])
        st.session_state["demo_records"].append(new_row)

        st.success("Demo record added to the current session.")

    if st.session_state.get("demo_records"):

        demo_df = pd.DataFrame(
            st.session_state["demo_records"]
        )

        st.markdown("### Newly Added Records")

        st.dataframe(
            demo_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download New Records",
            data=demo_df.to_csv(index=False).encode("utf-8"),
            file_name="new_machine_records.csv",
            mime="text/csv"
        )

    st.warning(
        "New demo records are kept in the current session only. "
        "Download them to save a copy."
    )


# ==================================================
# ANALYTICS
# ==================================================
elif page == "Analytics":

    st.subheader("Data Analytics Laboratory")

    with st.container(border=True):
        st.markdown("### Descriptive Statistics")

        st.dataframe(
            df[FEATURES].describe().round(2),
            use_container_width=True
        )

    with st.container(border=True):
        st.markdown("### Sensor Trend")

        selected_sensor = st.selectbox(
            "Select Sensor",
            FEATURES
        )

        st.line_chart(
            df[selected_sensor].reset_index(drop=True),
            color="#405CC5"
        )

    with st.container(border=True):
        st.markdown("### Correlation Matrix")

        correlation = df[FEATURES + [TARGET]].corr(
            numeric_only=True
        )

        st.dataframe(
            correlation.round(2),
            use_container_width=True
        )

    with st.container(border=True):
        st.markdown("### Important Features")

        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        st.bar_chart(
            importance.set_index("Feature"),
            color="#7385BB"
        )

    with st.container(border=True):
        st.markdown("### Sensor Averages by Failure Status")

        averages = df.groupby(TARGET)[FEATURES].mean().round(2)
        averages.index = averages.index.map({
            0: "Normal",
            1: "Failure"
        })

        st.dataframe(
            averages,
            use_container_width=True
        )


# ==================================================
# MAINTENANCE ALERTS
# ==================================================
elif page == "Maintenance Alerts":

    st.subheader("Maintenance Alert Center")

    st.write(
        "Find historical failures and records with high tool wear "
        "or torque according to demonstration thresholds."
    )

    alerts = df[
        (df[TARGET] == 1)
        | (df["Tool wear [min]"] >= 200)
        | (df["Torque [Nm]"] >= 60)
    ].copy()

    alerts["Alert Reason"] = np.where(
        alerts[TARGET] == 1,
        "Failure recorded",
        np.where(
            alerts["Tool wear [min]"] >= 200,
            "High tool wear",
            "High torque"
        )
    )

    a, b, c = st.columns(3)

    a.metric("Records Flagged", f"{len(alerts):,}")
    b.metric("Recorded Failures", f"{failure_count:,}")
    c.metric("Dataset Failure Rate", f"{failure_rate:.2f}%")

    if alerts.empty:
        st.success("No records matched these demonstration rules.")
    else:
        st.dataframe(
            alerts,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Maintenance Alerts",
            data=alerts.to_csv(index=False).encode("utf-8"),
            file_name="maintenance_alerts.csv",
            mime="text/csv"
        )

    st.caption(
        "These thresholds are for demonstration and are not certified "
        "industrial safety limits. No live sensor feed is connected."
    )


# ==================================================
# PROJECT DETAILS
# ==================================================
elif page == "Project Details":

    st.subheader("Project Information")

    st.markdown("""
    ### AI-Based Predictive Maintenance System

    **Objective**

    Analyze historical industrial machine data and use machine learning
    to estimate potential machine failures.

    **Core Features**
    - Interactive machine monitoring dashboard
    - Sensor-based AI prediction
    - Search and export machine records
    - Statistical analysis and visualization
    - Feature importance analysis
    - Maintenance alert generation

    **Technologies**
    - Python
    - Streamlit
    - Pandas and NumPy
    - Scikit-learn
    - Random Forest Classifier

    **Dataset**

    AI4I 2020 Predictive Maintenance Dataset.

    **Potential Applications**

    Manufacturing plants, textile industries, engineering workshops,
    and industrial equipment monitoring.
    """)

    a, b = st.columns(2)

    a.metric("Model Test Accuracy", f"{accuracy * 100:.2f}%")
    b.metric("Input Features", len(FEATURES))

    st.markdown("### Confusion Matrix")

    matrix_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Failure"],
        columns=["Predicted Normal", "Predicted Failure"]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    st.info(
        "Educational prototype only. Model performance depends on the "
        "dataset and may differ from real-world industrial performance."
    )


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    PREDICTIVE AI • INDUSTRIAL INTELLIGENCE • PYTHON + STREAMLIT
</div>
""", unsafe_allow_html=True)
