import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import io
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background-color: #F4F6FB;}
[data-testid="stSidebar"] {background-color: #E8ECF7;}
h1, h2, h3 {color: #263A63;}
div.stButton > button {
    background-color: #344D83;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚙️ AI-Based Predictive Maintenance System")
st.caption("Machine monitoring • Failure prediction • Maintenance analytics")

DB_FILE = "maintenance.db"
DATA_FILE = "ai4i2020.csv"

# ---------------- DATABASE ----------------
def get_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def initialize_database():
    with get_connection() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_name TEXT NOT NULL,
                machine_type TEXT,
                air_temp REAL,
                process_temp REAL,
                rpm REAL,
                torque REAL,
                tool_wear REAL,
                status TEXT DEFAULT 'Active',
                created_at TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS maintenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id INTEGER,
                service_date TEXT,
                description TEXT,
                cost REAL,
                status TEXT
            )
        """)

initialize_database()

# ---------------- DATASET ----------------
@st.cache_data
def load_dataset():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    return pd.DataFrame()

df = load_dataset()

def find_column(possible_names):
    for col in df.columns:
        for name in possible_names:
            if col.lower().replace("_", " ") == name.lower().replace("_", " "):
                return col
    return None

# Supports the standard AI4I 2020 dataset column names.
air_col = find_column(["Air temperature [K]", "Air temperature"])
process_col = find_column(["Process temperature [K]", "Process temperature"])
rpm_col = find_column(["Rotational speed [rpm]", "Rotational speed"])
torque_col = find_column(["Torque [Nm]", "Torque"])
wear_col = find_column(["Tool wear [min]", "Tool wear"])
failure_col = find_column(["Machine failure", "Failure"])

# ---------------- AI MODEL ----------------
@st.cache_resource
def train_model(file_modified_time):
    data = load_dataset()

    required = [air_col, process_col, rpm_col, torque_col, wear_col, failure_col]
    if data.empty or any(c is None for c in required):
        return None, None, None

    model_data = data[required].copy()
    model_data.columns = [
        "air_temp", "process_temp", "rpm", "torque", "tool_wear", "failure"
    ]

    for col in model_data.columns:
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

    model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()
    model_data = model_data[
        model_data["failure"].isin([0, 1])
    ]

    if len(model_data) < 20 or model_data["failure"].nunique() < 2:
        return None, None, None

    X = model_data.drop(columns=["failure"])
    y = model_data["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    return model, accuracy, matrix

modified_time = (
    os.path.getmtime(DATA_FILE) if os.path.exists(DATA_FILE) else 0
)
model, model_accuracy, model_matrix = train_model(modified_time)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Choose Module",
    [
        "Dashboard",
        "Machine Management",
        "AI Failure Prediction",
        "Data Analysis",
        "Statistical Analysis",
        "Maintenance History",
        "Web Scraping",
        "Reports",
        "Project Information"
    ]
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.header("Dashboard Overview")

    with get_connection() as con:
        machines = pd.read_sql_query("SELECT * FROM machines", con)
        services = pd.read_sql_query("SELECT * FROM maintenance", con)

    total = len(machines)
    active = int((machines["status"] == "Active").sum()) if total else 0
    alerts = int((machines["status"] == "Maintenance Required").sum()) if total else 0

    a, b, c, d = st.columns(4)
    a.metric("Total Machines", total)
    b.metric("Active Machines", active)
    c.metric("Maintenance Alerts", alerts)
    d.metric("Service Records", len(services))

    st.subheader("Recent Machine Records")
    if not machines.empty:
        st.dataframe(machines.tail(10), use_container_width=True)
    else:
        st.info("No machine records yet. Add a machine from Machine Management.")

    if not df.empty and failure_col:
        st.subheader("Training Dataset Summary")
        st.write("Dataset rows:", len(df))
        st.write("Dataset columns:", len(df.columns))
        st.write("Dataset failure records:", int(
            pd.to_numeric(df[failure_col], errors="coerce").fillna(0).sum()
        ))
    else:
        st.warning("Dataset not found or its column names are unsupported.")

# ---------------- MACHINE CRUD ----------------
elif page == "Machine Management":
    st.header("🛠️ Machine Management (CRUD)")

    with get_connection() as con:
        machines = pd.read_sql_query("SELECT * FROM machines", con)

    tab1, tab2, tab3 = st.tabs(["Add Machine", "View / Update", "Delete Machine"])

    with tab1:
        with st.form("add_machine"):
            name = st.text_input("Machine Name", placeholder="e.g. Textile Machine 01")
            kind = st.selectbox("Machine Type", ["M", "L", "H", "Other"])
            air = st.number_input("Air Temperature (K)", value=298.0)
            process = st.number_input("Process Temperature (K)", value=308.0)
            rpm = st.number_input("Rotational Speed (rpm)", min_value=0.0, value=1500.0)
            torque = st.number_input("Torque (Nm)", min_value=0.0, value=40.0)
            wear = st.number_input("Tool Wear (min)", min_value=0.0, value=10.0)
            submitted = st.form_submit_button("Add Machine")

            if submitted:
                if not name.strip():
                    st.error("Machine name is required.")
                else:
                    with get_connection() as con:
                        con.execute("""
                            INSERT INTO machines
                            (machine_name, machine_type, air_temp, process_temp,
                             rpm, torque, tool_wear, status, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            name.strip(), kind, air, process, rpm, torque,
                            wear, "Active", datetime.now().strftime("%Y-%m-%d %H:%M")
                        ))
                    st.success("Machine added and saved permanently.")
                    st.rerun()

    with tab2:
        if machines.empty:
            st.info("No machines available to update.")
        else:
            st.dataframe(machines, use_container_width=True)
            selected_id = st.selectbox("Select Machine ID", machines["id"].tolist())
            row = machines[machines["id"] == selected_id].iloc[0]

            with st.form("update_machine"):
                new_name = st.text_input("Machine Name", value=row["machine_name"])
                new_status = st.selectbox(
                    "Status",
                    ["Active", "Maintenance Required", "Under Repair", "Inactive"],
                    index=["Active", "Maintenance Required", "Under Repair", "Inactive"].index(
                        row["status"] if row["status"] in
                        ["Active", "Maintenance Required", "Under Repair", "Inactive"]
                        else "Active"
                    )
                )
                new_air = st.number_input("Air Temperature (K)", value=float(row["air_temp"]))
                new_process = st.number_input("Process Temperature (K)", value=float(row["process_temp"]))
                new_rpm = st.number_input("RPM", min_value=0.0, value=float(row["rpm"]))
                new_torque = st.number_input("Torque (Nm)", min_value=0.0, value=float(row["torque"]))
                new_wear = st.number_input("Tool Wear (min)", min_value=0.0, value=float(row["tool_wear"]))

                if st.form_submit_button("Update Machine"):
                    if not new_name.strip():
                        st.error("Machine name cannot be empty.")
                    else:
                        with get_connection() as con:
                            con.execute("""
                                UPDATE machines
                                SET machine_name=?, status=?, air_temp=?, process_temp=?,
                                    rpm=?, torque=?, tool_wear=?
                                WHERE id=?
                            """, (
                                new_name.strip(), new_status, new_air, new_process,
                                new_rpm, new_torque, new_wear, selected_id
                            ))
                        st.success("Machine updated.")
                        st.rerun()

    with tab3:
        if machines.empty:
            st.info("No machines to delete.")
        else:
            delete_id = st.selectbox(
                "Machine ID to delete", machines["id"].tolist(), key="delete_id"
            )
            st.warning("Deleting a machine cannot be undone.")
            if st.button("Delete Selected Machine"):
                with get_connection() as con:
                    con.execute("DELETE FROM machines WHERE id=?", (delete_id,))
                st.success("Machine deleted.")
                st.rerun()

# ---------------- AI FAILURE PREDICTION ----------------
elif page == "AI Failure Prediction":
    st.header("🤖 AI Machine Failure Prediction")
    st.write("Enter machine sensor readings to estimate failure risk.")

    if model is None:
        st.error(
            "AI model could not be trained. Put the standard AI4I 2020 CSV "
            "file named ai4i2020.csv in the same folder as app.py."
        )
    else:
        left, right = st.columns(2)
        with left:
            p_air = st.number_input("Air Temperature (K)", value=298.0)
            p_process = st.number_input("Process Temperature (K)", value=308.0)
            p_rpm = st.number_input("Rotational Speed (rpm)", min_value=0.0, value=1500.0)
        with right:
            p_torque = st.number_input("Torque (Nm)", min_value=0.0, value=40.0)
            p_wear = st.number_input("Tool Wear (min)", min_value=0.0, value=10.0)

        if st.button("Predict Machine Failure"):
            sample = pd.DataFrame([{
                "air_temp": p_air,
                "process_temp": p_process,
                "rpm": p_rpm,
                "torque": p_torque,
                "tool_wear": p_wear
            }])

            prediction = int(model.predict(sample)[0])
            probability = float(model.predict_proba(sample)[0][1])

            if prediction == 1:
                st.error(f"⚠️ Model predicts a possible failure. Estimated model risk: {probability:.1%}")
                st.info("Recommendation: inspect the machine and arrange a technician check.")
            else:
                st.success(f"✅ Model predicts no failure. Estimated model risk: {probability:.1%}")
                st.info("Continue routine monitoring and preventive maintenance.")

            st.caption(
                "This is a machine-learning estimate from the supplied dataset, "
                "not a guarantee of real-world machine safety."
            )

        st.metric("Held-out Test Accuracy", f"{model_accuracy:.1%}")

        if model_matrix is not None:
            st.subheader("Confusion Matrix")
            st.write("Rows = actual class (0 = no failure, 1 = failure); columns = predicted class.")
            st.dataframe(
                pd.DataFrame(
                    model_matrix,
                    index=["Actual No Failure", "Actual Failure"],
                    columns=["Predicted No Failure", "Predicted Failure"]
                )
            )

        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "Feature": ["Air Temperature", "Process Temperature", "RPM", "Torque", "Tool Wear"],
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.bar_chart(importance.set_index("Feature"))

# ---------------- DATA ANALYSIS ----------------
elif page == "Data Analysis":
    st.header("📊 Data Analysis & Preprocessing")

    if df.empty:
        st.warning("Dataset file ai4i2020.csv was not found.")
    else:
        st.subheader("Preview Dataset")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Dataset Information")
        summary = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum(),
            "Unique Values": df.nunique()
        })
        st.dataframe(summary)

        st.subheader("Clean Dataset")
        cleaned = df.copy()
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.drop_duplicates()
        cleaned = cleaned.dropna()
        st.write("Original rows:", len(df))
        st.write("Rows after removing duplicates and missing values:", len(cleaned))
        st.download_button(
            "Download Cleaned CSV",
            cleaned.to_csv(index=False).encode("utf-8"),
            "cleaned_machine_data.csv",
            "text/csv"
        )

        st.subheader("Numerical Statistics")
        st.dataframe(df.describe(include="all").T)

        numeric = df.select_dtypes(include=np.number)
        if not numeric.empty:
            st.subheader("Correlation Matrix")
            st.dataframe(numeric.corr())
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(9, 5))
            image = ax.imshow(numeric.corr(), aspect="auto")
            ax.set_xticks(range(len(numeric.columns)))
            ax.set_xticklabels(numeric.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(numeric.columns)))
            ax.set_yticklabels(numeric.columns, fontsize=7)
            fig.colorbar(image)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if failure_col:
            st.subheader("Machine Failure Distribution")
            counts = df[failure_col].value_counts(dropna=False).sort_index()
            st.bar_chart(counts)

# ---------------- STATISTICAL ANALYSIS ----------------
elif page == "Statistical Analysis":
    st.header("📈 Statistical Analysis")

    if df.empty or failure_col is None or torque_col is None:
        st.warning("Dataset with failure and torque columns is required.")
    else:
        stats_df = df[[failure_col, torque_col]].copy()
        stats_df[failure_col] = pd.to_numeric(stats_df[failure_col], errors="coerce")
        stats_df[torque_col] = pd.to_numeric(stats_df[torque_col], errors="coerce")
        stats_df = stats_df.dropna()

        no_failure = stats_df[stats_df[failure_col] == 0][torque_col]
        failure = stats_df[stats_df[failure_col] == 1][torque_col]

        if len(no_failure) > 1 and len(failure) > 1:
            st.write("Comparison: torque readings for failure vs. non-failure records.")
            st.metric("Average Torque — No Failure", f"{no_failure.mean():.2f}")
            st.metric("Average Torque — Failure", f"{failure.mean():.2f}")

            test_stat, p_value = stats.ttest_ind(
                no_failure, failure, equal_var=False, nan_policy="omit"
            )
            st.write(f"Welch t-test statistic: {test_stat:.4f}")
            st.write(f"P-value: {p_value:.6f}")

            if p_value < 0.05:
                st.success("At 5% significance, the two sample means differ statistically.")
            else:
                st.info("At 5% significance, there is not enough evidence of a difference in means.")

            mean_val = stats_df[torque_col].mean()
            sem_val = stats.sem(stats_df[torque_col])
            if len(stats_df) > 1 and np.isfinite(sem_val):
                ci = stats.t.interval(
                    0.95, len(stats_df) - 1, loc=mean_val, scale=sem_val
                )
                st.write(f"95% confidence interval for mean torque: {ci[0]:.2f} to {ci[1]:.2f}")
        else:
            st.info("Not enough records in both failure classes for the t-test.")

        st.caption(
            "The t-test compares observed dataset groups; it does not prove that torque causes failure."
        )

# ---------------- MAINTENANCE HISTORY ----------------
elif page == "Maintenance History":
    st.header("🔧 Maintenance History")

    with get_connection() as con:
        machines = pd.read_sql_query("SELECT id, machine_name FROM machines", con)
        history = pd.read_sql_query("SELECT * FROM maintenance ORDER BY id DESC", con)

    if machines.empty:
        st.info("Add a machine before creating a maintenance record.")
    else:
        with st.form("service_form"):
            machine_map = dict(zip(machines["machine_name"], machines["id"]))
            chosen_machine = st.selectbox("Machine", list(machine_map.keys()))
            service_date = st.date_input("Service Date", value=datetime.today().date())
            description = st.text_area("Service Description", placeholder="Inspection, lubrication, repair...")
            cost = st.number_input("Maintenance Cost", min_value=0.0, value=0.0)
            service_status = st.selectbox("Service Status", ["Completed", "Pending", "In Progress"])

            if st.form_submit_button("Save Maintenance Record"):
                if not description.strip():
                    st.error("Enter a service description.")
                else:
                    with get_connection() as con:
                        con.execute("""
                            INSERT INTO maintenance
                            (machine_id, service_date, description, cost, status)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            machine_map[chosen_machine], str(service_date),
                            description.strip(), cost, service_status
                        ))
                    st.success("Maintenance record saved.")
                    st.rerun()

    st.subheader("Saved Maintenance Records")
    if history.empty:
        st.info("No maintenance records yet.")
    else:
        st.dataframe(history, use_container_width=True)
        st.metric("Total Recorded Maintenance Cost", f"₹{history['cost'].sum():,.2f}")

# ---------------- WEB SCRAPING ----------------
elif page == "Web Scraping":
    st.header("🌐 Web Scraping Demo")
    st.write(
        "This module demonstrates Requests and BeautifulSoup. "
        "Only scrape websites you are allowed to access."
    )
    url = st.text_input("Public webpage URL", placeholder="https://example.com")

    if st.button("Fetch Webpage"):
        if not url.startswith(("https://", "http://")):
            st.error("Enter a valid URL starting with http:// or https://.")
        else:
            try:
                response = requests.get(
                    url,
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                title_text = soup.title.get_text(" ", strip=True) if soup.title else "No title"
                headings = [
                    h.get_text(" ", strip=True)
                    for h in soup.find_all(["h1", "h2", "h3"])[:20]
                ]
                st.success(f"Page title: {title_text}")
                st.write("Headings found:")
                st.write(headings if headings else "No headings found.")
            except requests.RequestException as error:
                st.error(f"Could not fetch webpage: {error}")
            except Exception as error:
                st.error(f"Unexpected error: {error}")

# ---------------- REPORTS ----------------
elif page == "Reports":
    st.header("📄 Reports & Export")

    with get_connection() as con:
        machines = pd.read_sql_query("SELECT * FROM machines", con)
        maintenance = pd.read_sql_query("SELECT * FROM maintenance", con)

    st.subheader("Machine Report")
    st.dataframe(machines, use_container_width=True)
    st.download_button(
        "Download Machine Report (CSV)",
        machines.to_csv(index=False).encode("utf-8"),
        "machine_report.csv",
        "text/csv"
    )

    st.subheader("Maintenance Report")
    st.dataframe(maintenance, use_container_width=True)
    st.download_button(
        "Download Maintenance Report (CSV)",
        maintenance.to_csv(index=False).encode("utf-8"),
        "maintenance_report.csv",
        "text/csv"
    )

# ---------------- PROJECT INFORMATION ----------------
elif page == "Project Information":
    st.header("📘 Project Information")
    st.markdown("""
    **Project:** AI-Based Predictive Maintenance System

    **Objective:** Estimate possible machine failure using sensor readings
    and help organize maintenance activities.

    **Python concepts demonstrated:**
    - Core concepts, functions, conditions, loops and exception handling
    - Lists, dictionaries, data frames and other data structures
    - Custom functions and modular program design within one file
    - File handling with CSV and SQLite database operations
    - CRUD operations: Create, Read, Update and Delete
    - NumPy, Pandas, SciPy, Matplotlib and Scikit-learn
    - Data cleaning, descriptive statistics, sampling and inference
    - Hypothesis testing and machine-learning classification
    - Web scraping using Requests and BeautifulSoup
    - CSV report generation and downloadable outputs

    **AI model:** Random Forest Classifier

    **Dataset:** AI4I 2020 Predictive Maintenance Dataset

    **Important:** Model performance depends on the training dataset.
    This academic demonstration is not a substitute for real industrial
    safety monitoring or certified maintenance procedures.
    """)
    st.info("SQLite records are stored locally in maintenance.db beside app.py.")
