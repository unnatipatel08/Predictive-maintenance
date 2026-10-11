import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

DB = "maintenance.db"

# ---------------- LIGHT BLUE THEME ----------------
st.markdown("""
<style>
.stApp {
    background-color: #F3F8FF;
    color: #243B53;
}
[data-testid="stSidebar"] {
    background-color: #E2EEFF;
}
h1, h2, h3 {
    color: #245B91;
}
.stButton > button,
.stDownloadButton > button {
    background-color: #77BDFB;
    color: #12304A;
    border-radius: 8px;
    border: none;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #D6E8FF;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- DATABASE ----------------
def connect_db():
    return sqlite3.connect(DB)


def setup_database():
    with connect_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            machine_type TEXT,
            location TEXT,
            temperature REAL,
            vibration REAL,
            hours_used REAL,
            status TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id INTEGER,
            result TEXT,
            risk REAL,
            created_at TEXT
        )
        """)


def get_machines():
    with connect_db() as conn:
        return pd.read_sql_query(
            "SELECT * FROM machines ORDER BY id DESC", conn
        )


def add_machine(name, machine_type, location, temperature,
                vibration, hours_used, status):
    with connect_db() as conn:
        conn.execute("""
        INSERT INTO machines
        (name, machine_type, location, temperature,
         vibration, hours_used, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            name, machine_type, location,
            temperature, vibration, hours_used, status
        ))


def update_machine(machine_id, name, machine_type, location,
                   temperature, vibration, hours_used, status):
    with connect_db() as conn:
        conn.execute("""
        UPDATE machines
        SET name=?, machine_type=?, location=?,
            temperature=?, vibration=?, hours_used=?, status=?
        WHERE id=?
        """, (
            name, machine_type, location,
            temperature, vibration, hours_used, status, machine_id
        ))


def delete_machine(machine_id):
    with connect_db() as conn:
        conn.execute("DELETE FROM predictions WHERE machine_id=?",
                     (machine_id,))
        conn.execute("DELETE FROM machines WHERE id=?", (machine_id,))


def save_prediction(machine_id, result, risk):
    with connect_db() as conn:
        conn.execute("""
        INSERT INTO predictions
        (machine_id, result, risk, created_at)
        VALUES (?, ?, ?, ?)
        """, (
            machine_id, result, risk,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))


setup_database()


# ---------------- DEMO DATASET ----------------
@st.cache_data
def create_dataset():
    rng = np.random.default_rng(42)
    count = 500

    temperature = rng.normal(65, 12, count).clip(25, 110)
    vibration = rng.normal(3, 1.4, count).clip(0.1, 10)
    hours = rng.integers(100, 6000, count)

    risk_score = (
        (temperature > 82).astype(int)
        + (vibration > 5).astype(int)
        + (hours > 4200).astype(int)
    )

    probability = np.clip(0.05 + risk_score * 0.25, 0.02, 0.95)
    failure = rng.binomial(1, probability)

    return pd.DataFrame({
        "temperature": temperature,
        "vibration": vibration,
        "hours_used": hours,
        "failure": failure
    })


@st.cache_resource
def train_model():
    df = create_dataset()

    X = df[["temperature", "vibration", "hours_used"]]
    y = df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, df, accuracy


model, dataset, accuracy = train_model()


# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Maintenance System")
st.sidebar.caption("AI-based machine monitoring")

page = st.sidebar.radio(
    "Select Section",
    ["Dashboard", "Machine Management", "Prediction & Analysis"]
)

st.sidebar.divider()
st.sidebar.caption("Python • Streamlit • SQLite • Random Forest")


# ==================================================
# DASHBOARD
# ==================================================
if page == "Dashboard":

    st.title("AI-Based Predictive Maintenance System")
    st.write("Monitor machine health and estimate failure risk.")

    machines = get_machines()

    total = len(machines)
    active = (
        int((machines["status"] == "Active").sum())
        if total else 0
    )
    needs_check = (
        int((machines["status"] == "Needs Check").sum())
        if total else 0
    )

    with connect_db() as conn:
        prediction_count = conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Machines", total)
    c2.metric("Active Machines", active)
    c3.metric("Needs Check", needs_check)
    c4.metric("Predictions", prediction_count)

    st.subheader("Machine Overview")

    if not machines.empty:
        st.dataframe(machines, use_container_width=True,
                     hide_index=True)

        fig, ax = plt.subplots()
        machines["status"].value_counts().plot(
            kind="bar", ax=ax
        )
        ax.set_ylabel("Number of Machines")
        ax.set_title("Machine Status")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No machines added yet. Open Machine Management.")

    st.subheader("Training Dataset")
    st.write(f"Dataset records: {len(dataset)}")
    st.write(f"Random Forest model accuracy: {accuracy:.2%}")

    st.download_button(
        "Download Machine Data CSV",
        machines.to_csv(index=False).encode("utf-8"),
        file_name="machine_data.csv",
        mime="text/csv"
    )


# ==================================================
# MACHINE MANAGEMENT
# ==================================================
elif page == "Machine Management":

    st.title("Machine Management")
    st.write("Add, view, update and delete machine records.")

    add_tab, view_tab, update_tab, delete_tab = st.tabs([
        "Add Machine",
        "View Machines",
        "Update Machine",
        "Delete Machine"
    ])

    # ADD
    with add_tab:
        with st.form("add_form"):
            name = st.text_input("Machine Name")

            machine_type = st.selectbox(
                "Machine Type",
                ["Motor", "Pump", "Fan", "Compressor", "Other"]
            )

            location = st.text_input("Location")

            c1, c2, c3 = st.columns(3)

            temperature = c1.number_input(
                "Temperature (°C)", 0.0, 200.0, 65.0
            )
            vibration = c2.number_input(
                "Vibration Level", 0.0, 20.0, 3.0
            )
            hours_used = c3.number_input(
                "Hours Used", 0.0, 200000.0, 1000.0
            )

            status = st.selectbox(
                "Status",
                ["Active", "Needs Check", "Under Maintenance"]
            )

            submitted = st.form_submit_button("Add Machine")

            if submitted:
                if not name.strip():
                    st.error("Please enter a machine name.")
                else:
                    try:
                        add_machine(
                            name.strip(), machine_type, location.strip(),
                            temperature, vibration, hours_used, status
                        )
                        st.success("Machine added successfully.")
                        st.rerun()
                    except sqlite3.Error as error:
                        st.error(f"Database error: {error}")

    # VIEW
    with view_tab:
        machines = get_machines()

        if machines.empty:
            st.info("No machine records found.")
        else:
            st.dataframe(
                machines,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Export Machines",
                machines.to_csv(index=False).encode("utf-8"),
                file_name="machines.csv",
                mime="text/csv"
            )

    # UPDATE
    with update_tab:
        machines = get_machines()

        if machines.empty:
            st.info("Add a machine before updating.")
        else:
            machine_ids = machines["id"].tolist()

            machine_id = st.selectbox(
                "Select Machine ID",
                machine_ids,
                key="update_id"
            )

            row = machines[machines["id"] == machine_id].iloc[0]

            with st.form("update_form"):
                new_name = st.text_input(
                    "Machine Name", value=str(row["name"])
                )

                types = [
                    "Motor", "Pump", "Fan", "Compressor", "Other"
                ]
                old_type = str(row["machine_type"])
                type_index = types.index(old_type) if old_type in types else 0

                new_type = st.selectbox(
                    "Machine Type", types, index=type_index
                )

                new_location = st.text_input(
                    "Location", value=str(row["location"] or "")
                )

                c1, c2, c3 = st.columns(3)

                new_temp = c1.number_input(
                    "Temperature", min_value=0.0, max_value=200.0,
                    value=float(row["temperature"])
                )
                new_vib = c2.number_input(
                    "Vibration", min_value=0.0, max_value=20.0,
                    value=float(row["vibration"])
                )
                new_hours = c3.number_input(
                    "Hours Used", min_value=0.0, max_value=200000.0,
                    value=float(row["hours_used"])
                )

                statuses = [
                    "Active", "Needs Check", "Under Maintenance"
                ]
                old_status = str(row["status"])
                status_index = statuses.index(old_status) if old_status in statuses else 0

                new_status = st.selectbox(
                    "Status", statuses, index=status_index
                )

                update_clicked = st.form_submit_button("Save Changes")

                if update_clicked:
                    if not new_name.strip():
                        st.error("Machine name cannot be empty.")
                    else:
                        try:
                            update_machine(
                                machine_id, new_name.strip(), new_type,
                                new_location.strip(), new_temp, new_vib,
                                new_hours, new_status
                            )
                            st.success("Machine updated successfully.")
                            st.rerun()
                        except sqlite3.Error as error:
                            st.error(f"Database error: {error}")

    # DELETE
    with delete_tab:
        machines = get_machines()

        if machines.empty:
            st.info("No machines available to delete.")
        else:
            machine_ids = machines["id"].tolist()

            delete_id = st.selectbox(
                "Select Machine ID",
                machine_ids,
                key="delete_id"
            )

            selected_row = machines[
                machines["id"] == delete_id
            ].iloc[0]

            st.write(f"Selected machine: **{selected_row['name']}**")
            confirm = st.checkbox(
                "I confirm that I want to delete this machine."
            )

            if st.button("Delete Selected Machine"):
                if confirm:
                    try:
                        delete_machine(delete_id)
                        st.success("Machine deleted successfully.")
                        st.rerun()
                    except sqlite3.Error as error:
                        st.error(f"Database error: {error}")
                else:
                    st.warning("Please confirm before deleting.")


# ==================================================
# PREDICTION & ANALYSIS
# ==================================================
else:

    st.title("Prediction & Analysis")
    st.write("Enter machine sensor values to estimate failure risk.")

    machines = get_machines()

    machine_options = {"Manual Input": None}

    for _, row in machines.iterrows():
        machine_options[
            f"{row['id']} - {row['name']}"
        ] = int(row["id"])

    selected_machine = st.selectbox(
        "Select Machine",
        list(machine_options.keys())
    )

    machine_id = machine_options[selected_machine]
    selected_row = None

    if machine_id is not None:
        selected_row = machines[
            machines["id"] == machine_id
        ].iloc[0]

    default_temp = (
        float(selected_row["temperature"])
        if selected_row is not None else 65.0
    )
    default_vib = (
        float(selected_row["vibration"])
        if selected_row is not None else 3.0
    )
    default_hours = (
        float(selected_row["hours_used"])
        if selected_row is not None else 1000.0
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        input_temp = c1.number_input(
            "Temperature (°C)", 0.0, 200.0, default_temp
        )
        input_vib = c2.number_input(
            "Vibration Level", 0.0, 20.0, default_vib
        )
        input_hours = c3.number_input(
            "Hours Used", 0.0, 200000.0, default_hours
        )

        predict_clicked = st.form_submit_button(
            "Predict Maintenance Risk"
        )

    if predict_clicked:
        try:
            features = pd.DataFrame([{
                "temperature": input_temp,
                "vibration": input_vib,
                "hours_used": input_hours
            }])

            risk = float(model.predict_proba(features)[0][1] * 100)

            result = "High Risk" if risk >= 50 else "Low Risk"

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)
            col1.metric("Estimated Failure Risk", f"{risk:.1f}%")
            col2.metric("Prediction", result)

            st.progress(min(int(risk), 100))

            if risk >= 50:
                st.warning(
                    "Higher estimated risk. Consider checking the machine."
                )
            else:
                st.success(
                    "Lower estimated risk. Continue routine monitoring."
                )

            if machine_id is not None:
                save_prediction(machine_id, result, risk)
                st.caption("Prediction saved in the database.")

        except Exception as error:
            st.error(f"Prediction error: {error}")

    st.divider()
    st.subheader("Dataset Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Records", len(dataset))
    c2.metric("Failure Cases", int(dataset["failure"].sum()))
    c3.metric("Model Accuracy", f"{accuracy:.2%}")

    st.subheader("Sensor Data Preview")
    st.dataframe(dataset.head(10), use_container_width=True,
                 hide_index=True)

    st.subheader("Data Visualization")

    chart1, chart2 = st.columns(2)

    with chart1:
        fig, ax = plt.subplots()
        ax.hist(dataset["temperature"], bins=20)
        ax.set_title("Temperature Distribution")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    with chart2:
        fig, ax = plt.subplots()
        dataset["failure"].value_counts().sort_index().plot(
            kind="bar", ax=ax
        )
        ax.set_title("Failure vs No Failure")
        ax.set_xlabel("0 = No Failure, 1 = Failure")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": ["Temperature", "Vibration", "Hours Used"],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.dataframe(importance, use_container_width=True,
                 hide_index=True)

    fig, ax = plt.subplots()
    ax.barh(
        importance["Feature"][::-1],
        importance["Importance"][::-1]
    )
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Statistical Analysis")

    st.write("**Descriptive Statistics**")
    st.dataframe(
        dataset[["temperature", "vibration", "hours_used"]]
        .describe().round(2),
        use_container_width=True
    )

    normal_temp = dataset.loc[
        dataset["failure"] == 0, "temperature"
    ]
    failure_temp = dataset.loc[
        dataset["failure"] == 1, "temperature"
    ]

    if len(normal_temp) >= 2 and len(failure_temp) >= 2:
        test = stats.ttest_ind(
            failure_temp, normal_temp, equal_var=False
        )

        st.write("**Hypothesis Testing: Temperature**")
        st.write("H₀: Both groups have equal mean temperature.")
        st.write("H₁: Both groups have different mean temperature.")
        st.write(f"p-value: **{test.pvalue:.4f}**")

        if test.pvalue < 0.05:
            st.info(
                "The difference is statistically significant "
                "at the 0.05 level."
            )
        else:
            st.info(
                "There is not enough evidence of a difference "
                "at the 0.05 level."
            )

    st.download_button(
        "Download Training Dataset",
        dataset.to_csv(index=False).encode("utf-8"),
        file_name="predictive_maintenance_data.csv",
        mime="text/csv"
    )

    st.caption(
        "Note: This demo uses generated sample data unless a compatible "
        "dataset is provided. Its predictions are for learning only, "
        "not for real industrial safety decisions."
    )
'''
print(code)
