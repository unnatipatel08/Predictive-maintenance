import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI-Based Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f1f8ff;
}
[data-testid="stSidebar"] {
    background-color: #dceeff;
}
h1, h2, h3 {
    color: #174a72;
}
</style>
""", unsafe_allow_html=True)

st.title("AI-Based Predictive Maintenance System")
st.caption("Python for Data Science | College Project")


# ---------------- DATABASE ----------------

DB_FILE = "maintenance.db"
DATA_FILE = "ai4i2020.csv"


def get_connection():
    return sqlite3.connect(DB_FILE)


def create_tables():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                machine_type TEXT,
                temperature REAL,
                vibration REAL,
                hours_used REAL,
                status TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS maintenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id INTEGER,
                result TEXT,
                temperature REAL,
                vibration REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def add_machine(name, machine_type, temperature,
                vibration, hours_used, status):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO machines
            (name, machine_type, temperature, vibration,
             hours_used, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, machine_type, temperature,
              vibration, hours_used, status))


def get_machines():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM machines ORDER BY id DESC", conn
        )


def update_machine(machine_id, name, machine_type,
                   temperature, vibration, hours_used, status):
    with get_connection() as conn:
        conn.execute("""
            UPDATE machines
            SET name=?, machine_type=?, temperature=?,
                vibration=?, hours_used=?, status=?
            WHERE id=?
        """, (name, machine_type, temperature, vibration,
              hours_used, status, machine_id))


def delete_machine(machine_id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM maintenance WHERE machine_id=?",
            (machine_id,)
        )
        conn.execute(
            "DELETE FROM machines WHERE id=?",
            (machine_id,)
        )


def add_maintenance(machine_id, result, temperature, vibration):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO maintenance
            (machine_id, result, temperature, vibration)
            VALUES (?, ?, ?, ?)
        """, (machine_id, result, temperature, vibration))


def get_maintenance():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM maintenance ORDER BY id DESC", conn
        )


create_tables()


# ---------------- DATASET ----------------

@st.cache_data
def load_dataset():
    if Path(DATA_FILE).exists():
        data = pd.read_csv(DATA_FILE)

        required = {
            "temperature", "vibration",
            "hours_used", "failure"
        }

        if required.issubset(data.columns):
            return data[list(required)].copy()

        # Support the standard AI4I 2020 dataset.
        names = {
            str(col).strip().lower(): col
            for col in data.columns
        }

        original_columns = [
            "air temperature [k]",
            "rotational speed [rpm]",
            "tool wear [min]",
            "machine failure"
        ]

        if all(col in names for col in original_columns):
            return pd.DataFrame({
                "temperature":
                    pd.to_numeric(
                        data[names["air temperature [k]"]],
                        errors="coerce"
                    ) - 273.15,

                "vibration":
                    pd.to_numeric(
                        data[names["rotational speed [rpm]"]],
                        errors="coerce"
                    ) / 500.0,

                "hours_used":
                    pd.to_numeric(
                        data[names["tool wear [min]"]],
                        errors="coerce"
                    ) * 10.0,

                "failure":
                    pd.to_numeric(
                        data[names["machine failure"]],
                        errors="coerce"
                    )
            })

    # Reproducible educational demo dataset.
    rng = np.random.default_rng(42)

    rows = 500
    temperature = rng.uniform(35, 100, rows)
    vibration = rng.uniform(0.1, 10, rows)
    hours_used = rng.uniform(100, 10000, rows)

    risk = (
        (temperature > 75).astype(int)
        + (vibration > 7).astype(int)
        + (hours_used > 8000).astype(int)
    )

    failure = (risk >= 2).astype(int)

    data = pd.DataFrame({
        "temperature": temperature.round(2),
        "vibration": vibration.round(2),
        "hours_used": hours_used.round(2),
        "failure": failure
    })

    data.to_csv(DATA_FILE, index=False)
    return data


def preprocess_data(data):
    data = data.copy()

    columns = [
        "temperature", "vibration",
        "hours_used", "failure"
    ]

    data = data[columns].drop_duplicates()

    for column in columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna()
    data = data[data["failure"].isin([0, 1])]

    return data


@st.cache_resource
def train_model():
    data = preprocess_data(load_dataset())

    if len(data) < 10 or data["failure"].nunique() < 2:
        raise ValueError(
            "Dataset needs at least 10 valid rows and both classes."
        )

    X = data[[
        "temperature", "vibration", "hours_used"
    ]]

    y = data["failure"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


def failure_probability(model, temperature, vibration, hours_used):
    sample = pd.DataFrame([{
        "temperature": temperature,
        "vibration": vibration,
        "hours_used": hours_used
    }])

    probabilities = model.predict_proba(sample)[0]

    classes = list(
        model.named_steps["randomforestclassifier"].classes_
    )

    return float(probabilities[classes.index(1)])


def hypothesis_test(data):
    failed = data.loc[
        data["failure"] == 1, "temperature"
    ]

    normal = data.loc[
        data["failure"] == 0, "temperature"
    ]

    if len(failed) < 2 or len(normal) < 2:
        raise ValueError("Both groups need at least two records.")

    result = stats.ttest_ind(
        failed, normal, equal_var=False
    )

    return normal.mean(), failed.mean(), result.pvalue


def scrape_page(url):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Enter a valid HTTP or HTTPS URL.")

    response = requests.get(
        url,
        timeout=8,
        headers={"User-Agent": "Student-PDS-Demo/1.0"}
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title = (
        soup.title.get_text(strip=True)
        if soup.title else "No title found"
    )

    headings = [
        heading.get_text(" ", strip=True)
        for heading in soup.find_all(["h1", "h2"])[:5]
    ]

    return title, headings


# ---------------- SIDEBAR ----------------

section = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Machine Management",
        "Prediction & Analysis"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Student PDS Project")


# ==================================================
# 1. DASHBOARD
# ==================================================

if section == "Dashboard":

    st.header("Dashboard")

    machines = get_machines()
    history = get_maintenance()

    total = len(machines)

    running = (
        int((machines["status"] == "Running").sum())
        if total else 0
    )

    attention = (
        int(
            (machines["status"] == "Needs Maintenance").sum()
        )
        if total else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Machines", total)
    col2.metric("Running Machines", running)
    col3.metric("Needs Maintenance", attention)

    st.subheader("Machine Status")

    if machines.empty:
        st.info("No machines added. Go to Machine Management.")
    else:
        counts = machines["status"].value_counts()

        fig, ax = plt.subplots()

        counts.plot(kind="bar", ax=ax)

        ax.set_xlabel("Status")
        ax.set_ylabel("Number of Machines")
        ax.set_title("Machines by Status")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("All Machine Records")
        st.dataframe(machines, use_container_width=True)

    st.subheader("Recent Maintenance History")

    if history.empty:
        st.info("No maintenance records available.")
    else:
        st.dataframe(
            history.head(10),
            use_container_width=True
        )


# ==================================================
# 2. MACHINE MANAGEMENT
# ==================================================

elif section == "Machine Management":

    st.header("Machine Management")

    with st.expander("Add New Machine", expanded=True):

        with st.form("add_form"):

            name = st.text_input("Machine Name")

            machine_type = st.selectbox(
                "Machine Type",
                ["Motor", "Pump", "Fan", "Compressor", "Other"]
            )

            col1, col2 = st.columns(2)

            temperature = col1.number_input(
                "Temperature (°C)",
                min_value=0.0,
                max_value=200.0,
                value=50.0
            )

            vibration = col2.number_input(
                "Vibration",
                min_value=0.0,
                max_value=20.0,
                value=2.0
            )

            hours_used = st.number_input(
                "Operating Hours",
                min_value=0.0,
                value=1000.0,
                step=100.0
            )

            status = st.selectbox(
                "Status",
                ["Running", "Needs Maintenance", "Stopped"]
            )

            submitted = st.form_submit_button("Add Machine")

            if submitted:
                if not name.strip():
                    st.error("Please enter a machine name.")
                else:
                    try:
                        add_machine(
                            name.strip(),
                            machine_type,
                            temperature,
                            vibration,
                            hours_used,
                            status
                        )

                        st.success("Machine added successfully.")
                        st.rerun()

                    except sqlite3.Error as error:
                        st.error(f"Database error: {error}")

    machines = get_machines()

    st.divider()
    st.subheader("Search and View Machines")

    search = st.text_input("Search machine by name")

    if machines.empty:
        st.info("No machine records found.")
    else:
        filtered = machines[
            machines["name"].str.contains(
                search, case=False, na=False
            )
        ]

        st.dataframe(filtered, use_container_width=True)

        st.divider()
        st.subheader("Update Machine")

        machine_id = st.selectbox(
            "Select machine ID to update",
            machines["id"].tolist(),
            key="update_id"
        )

        selected = machines[
            machines["id"] == machine_id
        ].iloc[0]

        types = [
            "Motor", "Pump", "Fan", "Compressor", "Other"
        ]

        statuses = [
            "Running", "Needs Maintenance", "Stopped"
        ]

        with st.form("update_form"):

            new_name = st.text_input(
                "Machine Name",
                str(selected["name"])
            )

            old_type = selected["machine_type"]
            type_index = types.index(old_type) if old_type in types else 0

            new_type = st.selectbox(
                "Machine Type",
                types,
                index=type_index
            )

            new_temp = st.number_input(
                "Temperature",
                min_value=0.0,
                max_value=200.0,
                value=float(selected["temperature"])
            )

            new_vibration = st.number_input(
                "Vibration",
                min_value=0.0,
                max_value=20.0,
                value=float(selected["vibration"])
            )

            new_hours = st.number_input(
                "Operating Hours",
                min_value=0.0,
                value=float(selected["hours_used"])
            )

            old_status = selected["status"]
            status_index = (
                statuses.index(old_status)
                if old_status in statuses else 0
            )

            new_status = st.selectbox(
                "Status",
                statuses,
                index=status_index
            )

            update_button = st.form_submit_button("Update Machine")

            if update_button:
                if not new_name.strip():
                    st.error("Machine name cannot be empty.")
                else:
                    try:
                        update_machine(
                            int(machine_id),
                            new_name.strip(),
                            new_type,
                            new_temp,
                            new_vibration,
                            new_hours,
                            new_status
                        )

                        st.success("Machine updated successfully.")
                        st.rerun()

                    except sqlite3.Error as error:
                        st.error(f"Database error: {error}")

        st.divider()
        st.subheader("Delete Machine")

        delete_id = st.selectbox(
            "Select machine ID to delete",
            machines["id"].tolist(),
            key="delete_id"
        )

        confirm = st.checkbox(
            "I confirm that I want to delete this machine."
        )

        if st.button("Delete Machine"):
            if confirm:
                try:
                    delete_machine(int(delete_id))
                    st.success("Machine deleted.")
                    st.rerun()

                except sqlite3.Error as error:
                  st.error(f"Database error: {error}")
