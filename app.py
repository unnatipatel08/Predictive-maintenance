
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# ---------------- PROFESSIONAL THEME ----------------
st.markdown("""
<style>
.stApp {
    background: #F3F5FA;
    color: #202B45;
}
[data-testid="stSidebar"] {
    background: #172554;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
h1, h2, h3 {
    color: #243B70;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DCE3F0;
    padding: 18px;
    border-radius: 12px;
}
.stButton > button {
    background: #334EAC;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATASET ----------------
DATA_FILE = Path(__file__).parent / "ai4i2020.csv"

@st.cache_data
def load_data():
    if DATA_FILE.exists():
        data = pd.read_csv(DATA_FILE)
    else:
        rng = np.random.default_rng(42)
        n = 500
        air = rng.normal(300, 2, n)
        process = air + rng.normal(10, 1.5, n)
        rpm = rng.integers(1200, 2800, n)
        torque = rng.uniform(10, 70, n)
        wear = rng.integers(0, 250, n)
        failure = (
            (torque > 60) | (wear > 220) | (rpm < 1300)
        ).astype(int)

        data = pd.DataFrame({
            "UDI": range(1, n + 1),
            "Product ID": [f"M-{i:04d}" for i in range(1, n + 1)],
            "Type": rng.choice(["L", "M", "H"], n),
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
            "Machine failure": failure
        })
    return data

df = load_data()

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
TARGET = "Machine failure"

st.sidebar.title("⚙️ Predictive AI")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Machine Records",
        "AI Prediction",
        "Analytics",
        "Maintenance Alerts",
        "Project Details"
    ]
)
st.sidebar.caption("Industrial Machine Monitoring")
st.sidebar.write(f"Dataset records: {len(df):,}")

st.title("⚙️ AI-Based Predictive Maintenance System")
st.caption("Machine Monitoring • Failure Prediction • Data Analytics")
st.divider()

# ---------------- DATA VALIDATION ----------------
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]

if missing:
    st.error("Dataset columns missing: " + ", ".join(missing))
    st.write("CSV columns found:", list(df.columns))
    st.stop()

# ---------------- PREPROCESSING ----------------
X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET], errors="coerce")
valid = X.notna().all(axis=1) & y.notna()

X = X.loc[valid]
y = y.loc[valid].astype(int)

if len(X) < 10 or y.nunique() < 2:
    st.error("Not enough valid data to train the AI model.")
    st.stop()

# ---------------- MACHINE LEARNING ----------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    return (
        clf,
        accuracy_score(y_test, predictions),
        confusion_matrix(y_test, predictions)
    )

model, accuracy, cm = train_model(X, y)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.subheader("📊 Dashboard Overview")

    total = len(df)
    failures = int((df[TARGET] == 1).sum())
    normal = total - failures
    rate = failures / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{total:,}")
    c2.metric("Failure Records", f"{failures:,}")
    c3.metric("Normal Records", f"{normal:,}")
    c4.metric("Failure Rate", f"{rate:.2f}%")

    left, right = st.columns(2)

    with left:
        st.subheader("Failure Distribution")
        counts = df[TARGET].value_counts().rename(
            index={0: "Normal", 1: "Failure"}
        )
        st.bar_chart(counts)

    with right:
        st.subheader("Machine Type Distribution")
        if "Type" in df.columns:
            st.bar_chart(df["Type"].value_counts())

    st.subheader("Recent Machine Records")
    st.dataframe(df.tail(10), use_container_width=True)

    st.download_button(
        "⬇️ Download Dataset CSV",
        df.to_csv(index=False).encode("utf-8"),
        "machine_records.csv",
        "text/csv"
    )

# ---------------- MACHINE RECORDS ----------------
elif page == "Machine Records":
    st.subheader("🗂️ Machine Records")

    search = st.text_input("Search machine records")
    shown = df.copy()

    if search:
        mask = shown.astype(str).apply(
            lambda col: col.str.contains(
                search, case=False, na=False
            )
        ).any(axis=1)
        shown = shown[mask]

    st.dataframe(shown, use_container_width=True)

    st.download_button(
        "Download CSV",
        shown.to_csv(index=False).encode("utf-8"),
        "filtered_machine_records.csv",
        "text/csv"
    )

    st.divider()
    st.subheader("➕ Add Machine Record")

    with st.form("add_machine"):
        air = st.number_input("Air Temperature (K)", value=300.0)
        process = st.number_input("Process Temperature (K)", value=310.0)
        rpm = st.number_input("Rotational Speed (rpm)", min_value=0, value=1500)
        torque = st.number_input("Torque (Nm)", min_value=0.0, value=40.0)
        wear = st.number_input("Tool Wear (min)", min_value=0, value=100)
        failure = st.selectbox("Actual Failure Status", [0, 1])
        save = st.form_submit_button("Add Record")

    if save:
        record = {
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
            "Machine failure": failure
        }
        st.session_state.setdefault("new_records", [])
        st.session_state["new_records"].append(record)
        st.success("Machine record added to this session.")

    if st.session_state.get("new_records"):
        st.subheader("New Records")
        new_df = pd.DataFrame(st.session_state["new_records"])
        st.dataframe(new_df, use_container_width=True)
        st.download_button(
            "Download New Records",
            new_df.to_csv(index=False).encode("utf-8"),
            "new_machine_records.csv",
            "text/csv"
        )

    st.info("New records remain in the current session; download them to keep a copy.")

# ---------------- AI PREDICTION ----------------
elif page == "AI Prediction":
    st.subheader("🤖 Predict Machine Failure")
    st.write("Enter machine sensor values below.")

    c1, c2 = st.columns(2)

    with c1:
        air = st.number_input("Air Temperature (K)", 280.0, 320.0, 300.0)
        process = st.number_input("Process Temperature (K)", 280.0, 330.0, 310.0)
        rpm = st.number_input("Rotational Speed (rpm)", 500, 4000, 1500)

    with c2:
        torque = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0)
        wear = st.number_input("Tool Wear (min)", 0, 300, 100)

    if st.button("🔍 Predict Machine Status", use_container_width=True):
        sample = pd.DataFrame([{
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear
        }])

        prediction = int(model.predict(sample)[0])
        probs = model.predict_proba(sample)[0]
        failure_index = list(model.classes_).index(1)
        risk = float(probs[failure_index]) * 100

        if prediction == 1:
            st.error("⚠️ Potential Machine Failure")
        else:
            st.success("✅ Normal Operation")

        st.metric("Estimated Failure Probability", f"{risk:.1f}%")
        st.progress(int(round(risk)))
        st.caption("Model estimates are not a guarantee of real machine condition.")

# ---------------- ANALYTICS ----------------
elif page == "Analytics":
    st.subheader("📈 Data Analytics")

    st.write("### Descriptive Statistics")
    st.dataframe(df[FEATURES].describe().round(2), use_container_width=True)

    selected = st.selectbox("Choose sensor for visualization", FEATURES)
    st.write("### Sensor Trend")
    st.line_chart(df[selected].reset_index(drop=True))

    st.write("### Correlation Matrix")
    st.dataframe(
        df[FEATURES + [TARGET]].corr(numeric_only=True).round(2),
        use_container_width=True
    )

    st.write("### Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))

    st.write("### Average Sensor Values by Failure Status")
    comparison = df.groupby(TARGET)[FEATURES].mean().round(2)
    comparison.index = comparison.index.map({
        0: "Normal", 1: "Failure"
    })
    st.dataframe(comparison, use_container_width=True)

# ---------------- MAINTENANCE ALERTS ----------------
elif page == "Maintenance Alerts":
    st.subheader("🚨 Maintenance Alerts")

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

    a, b = st.columns(2)
    a.metric("Records Requiring Attention", len(alerts))
    b.metric("Recorded Failures", int((df[TARGET] == 1).sum()))

    if alerts.empty:
        st.success("No alerts found using these rules.")
    else:
        st.dataframe(alerts, use_container_width=True)
        st.download_button(
            "Download Alerts",
            alerts.to_csv(index=False).encode("utf-8"),
            "maintenance_alerts.csv",
            "text/csv"
        )

    st.caption("Alerts are based on historical data, not live sensors.")

# ---------------- PROJECT DETAILS ----------------
elif page == "Project Details":
    st.subheader("📘 Project Information")

    st.markdown("""
    **Project Name:** AI-Based Predictive Maintenance System

    **Objective:** Analyze historical industrial machine data and
    estimate potential machine failure using machine learning.

    **Technologies:** Python, Streamlit, Pandas, NumPy, Scikit-learn.

    **Algorithm:** Random Forest Classifier.

    **Dataset:** AI4I 2020 Predictive Maintenance Dataset.

    **Applications:** Manufacturing plants, textile industries,
    engineering workshops and industrial equipment monitoring.
    """)

    st.metric("Model Test Accuracy", f"{accuracy * 100:.2f}%")
    st.write("### Confusion Matrix")
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual Normal", "Actual Failure"],
            columns=["Predicted Normal", "Predicted Failure"]
        ),
        use_container_width=True
    )

    st.warning(
        "This is an educational prototype. Do not use predictions "
        "as the only basis for real industrial safety decisions."
    )

st.divider()
st.caption("AI-Based Predictive Maintenance | Python + Streamlit")
