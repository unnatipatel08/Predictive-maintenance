import os
import sqlite3
from datetime import date, datetime
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


# ==================================================
# 1. PAGE CONFIGURATION AND LIGHT BLUE THEME
# ==================================================

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #EAF5FF 0%, #F7FBFF 100%);
}
[data-testid="stSidebar"] {
    background-color: #D9ECFF;
    border-right: 1px solid #B8D9F5;
}
h1, h2, h3 {
    color: #174A7E !important;
}
p, label, .stMarkdown {
    color: #243B53;
}
div.stButton > button {
    background-color: #2476B9;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.45rem 1rem;
}
div.stButton > button:hover {
    background-color: #155A91;
    color: white;
}
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #C8E0F5;
    padding: 15px;
    border-radius: 12px;
}
div[data-testid="stForm"] {
    background-color: rgba(255,255,255,0.7);
    padding: 14px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚙️ AI-Based Predictive Maintenance System")
st.caption("Smart Machine Monitoring | AI Failure Prediction | Maintenance Analytics")

DB_FILE = "maintenance.db"
DATA_FILE = "ai4i2020.csv"


# ==================================================
# 2. DATABASE AND CRUD FUNCTIONS
# ==================================================

def connect_db():
    return sqlite3.connect(DB_FILE, timeout=20)


def initialize_database():
    with connect_db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
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


def get_machines():
    with connect_db() as con:
        return pd.read_sql_query(
            "SELECT * FROM machines ORDER BY id DESC", con
        )


def get_maintenance():
    with connect_db() as con:
        return pd.read_sql_query(
            "SELECT * FROM maintenance ORDER BY id DESC", con
        )


def add_machine(name, machine_type, air, process, rpm, torque, wear):
    with connect_db() as con:
        con.execute("""
            INSERT INTO machines
            (name, machine_type, air_temp, process_temp,
             rpm, torque, tool_wear, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, machine_type, air, process, rpm, torque, wear,
            "Active", datetime.now().strftime("%Y-%m-%d %H:%M")
        ))


def update_machine(machine_id, name, machine_type, air,
                   process, rpm, torque, wear, status):
    with connect_db() as con:
        con.execute("""
            UPDATE machines
            SET name=?, machine_type=?, air_temp=?, process_temp=?,
                rpm=?, torque=?, tool_wear=?, status=?
            WHERE id=?
        """, (
            name, machine_type, air, process, rpm, torque,
            wear, status, machine_id
        ))


def delete_machine(machine_id):
    with connect_db() as con:
        con.execute("DELETE FROM maintenance WHERE machine_id=?", (machine_id,))
        con.execute("DELETE FROM machines WHERE id=?", (machine_id,))


initialize_database()


# ==================================================
# 3. DATASET LOADING AND PREPROCESSING
# ==================================================

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()

    try:
        data = pd.read_csv(DATA_FILE)
        data.columns = [str(c).strip() for c in data.columns]
        return data
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


df = load_dataset()


def match_column(data, possible_names):
    normalized = {
        str(c).strip().lower(): c for c in data.columns
    }

    for name in possible_names:
        key = name.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


air_col = match_column(df, ["Air temperature [K]", "Air temperature"])
process_col = match_column(
    df, ["Process temperature [K]", "Process temperature"]
)
rpm_col = match_column(df, ["Rotational speed [rpm]", "Rotational speed"])
torque_col = match_column(df, ["Torque [Nm]", "Torque"])
wear_col = match_column(df, ["Tool wear [min]", "Tool wear"])
failure_col = match_column(df, ["Machine failure", "Failure"])

feature_columns = [
    air_col, process_col, rpm_col, torque_col, wear_col
]

dataset_ready = (
    not df.empty
    and all(c is not None for c in feature_columns)
    and failure_col is not None
)


# ==================================================
# 4. RANDOM FOREST AI MODEL
# ==================================================

@st.cache_resource
def train_ai_model(data):
    required = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure"
    ]

    # Standard AI4I column names are used for training.
    if not all(c in data.columns for c in required):
        return None, None, None, None

    training = data[required].copy()
    training.columns = [
        "air_temp", "process_temp", "rpm",
        "torque", "tool_wear", "failure"
    ]

    for col in training.columns:
        training[col] = pd.to_numeric(training[col], errors="coerce")

    training = training.replace([np.inf, -np.inf], np.nan).dropna()
    training = training[training["failure"].isin([0, 1])]

    if len(training) < 20 or training["failure"].nunique() != 2:
        return None, None, None, None

    X = training.drop(columns=["failure"])
    y = training["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, accuracy, matrix, importance


model = None
model_accuracy = None
model_matrix = None
feature_importance = None

if dataset_ready:
    model, model_accuracy, model_matrix, feature_importance = train_ai_model(df)


# ==================================================
# 5. NAVIGATION
# ==================================================

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Select Module",
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

st.sidebar.divider()
st.sidebar.caption("AI Predictive Maintenance")
st.sidebar.caption("Academic Project")


# ==================================================
# 6. DASHBOARD
# ==================================================

if page == "Dashboard":
    st.header("📊 Dashboard Overview")

    machines = get_machines()
    services = get_maintenance()

    total = len(machines)
    active = int((machines["status"] == "Active").sum()) if total else 0
    alerts = int(
        (machines["status"] == "Maintenance Required").sum()
    ) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Machines", total)
    c2.metric("Active Machines", active)
    c3.metric("Maintenance Alerts", alerts)
    c4.metric("Service Records", len(services))

    st.subheader("Recent Machines")

    if machines.empty:
        st.info("No machines found. Add a machine in Machine Management.")
    else:
        st.dataframe(machines.head(10), use_container_width=True)

        status_counts = machines["status"].value_counts()
        st.subheader("Machine Status Overview")
        st.bar_chart(status_counts)

    st.subheader("AI Dataset Status")

    if dataset_ready:
        st.success("AI training dataset loaded.")
        a, b, c = st.columns(3)
        a.metric("Dataset Rows", len(df))
        b.metric("Dataset Columns", len(df.columns))

        failure_total = pd.to_numeric(
            df[failure_col], errors="coerce"
        ).fillna(0).sum()

        c.metric("Failure Records", int(failure_total))

        if model_accuracy is not None:
            st.metric("AI Test Accuracy", f"{model_accuracy:.1%}")
    else:
        st.warning(
            "Dataset not loaded correctly. Check ai4i2020.csv and its columns."
        )


# ==================================================
# 7. MACHINE MANAGEMENT: CREATE, READ, UPDATE, DELETE
# ==================================================

elif page == "Machine Management":
    st.header("🛠️ Machine Management")
    st.write("Add, view, update and delete machine records.")

    tab_add, tab_view, tab_update, tab_delete = st.tabs(
        ["Add Machine", "View Machines", "Update Machine", "Delete Machine"]
    )

    with tab_add:
        with st.form("add_machine_form"):
            name = st.text_input("Machine Name")
            machine_type = st.selectbox("Machine Type", ["M", "L", "H", "Other"])

            col1, col2 = st.columns(2)

            with col1:
                air = st.number_input("Air Temperature (K)", value=298.0)
                rpm = st.number_input("Rotational Speed (rpm)", min_value=0.0, value=1500.0)
                wear = st.number_input("Tool Wear (min)", min_value=0.0, value=10.0)

            with col2:
                process = st.number_input("Process Temperature (K)", value=308.0)
                torque = st.number_input("Torque (Nm)", min_value=0.0, value=40.0)

            submit = st.form_submit_button("➕ Add Machine")

            if submit:
                if not name.strip():
                    st.error("Please enter a machine name.")
                else:
                    try:
                        add_machine(
                            name.strip(), machine_type,
                            air, process, rpm, torque, wear
                        )
                        st.success("Machine added successfully.")
                        st.rerun()
                    except sqlite3.Error as error:
                        st.error(f"Database error: {error}")

    with tab_view:
        machines = get_machines()

        if machines.empty:
            st.info("No machine records available.")
        else:
            st.dataframe(machines, use_container_width=True)

    with tab_update:
        machines = get_machines()

        if machines.empty:
            st.info("Add a machine before updating.")
        else:
            selected = st.selectbox(
                "Choose Machine ID", machines["id"].tolist(), key="update_id"
            )
            row = machines[machines["id"] == selected].iloc[0]

            with st.form("update_machine_form"):
                new_name = st.text_input("Machine Name", value=row["name"])
                types = ["M", "L", "H", "Other"]
                current_type = row["machine_type"]
                type_index = types.index(current_type) if current_type in types else 3
                new_type = st.selectbox("Machine Type", types, index=type_index)

                a, b = st.columns(2)
                with a:
                    new_air = st.number_input("Air Temperature (K)", value=float(row["air_temp"]))
                    new_rpm = st.number_input("RPM", min_value=0.0, value=float(row["rpm"]))
                    new_wear = st.number_input("Tool Wear (min)", min_value=0.0, value=float(row["tool_wear"]))
                with b:
                    new_process = st.number_input("Process Temperature (K)", value=float(row["process_temp"]))
                    new_torque = st.number_input("Torque (Nm)", min_value=0.0, value=float(row["torque"]))

                statuses = ["Active", "Maintenance Required", "Under Repair", "Inactive"]
                old_status = row["status"]
                status_index = statuses.index(old_status) if old_status in statuses else 0
                new_status = st.selectbox("Status", statuses, index=status_index)

                if st.form_submit_button("💾 Update Machine"):
                    if not new_name.strip():
                        st.error("Machine name is required.")
                    else:
                        try:
                            update_machine(
                                selected, new_name.strip(), new_type,
                                new_air, new_process, new_rpm,
                                new_torque, new_wear, new_status
                            )
                            st.success("Machine updated.")
                            st.rerun()
                        except sqlite3.Error as error:
                            st.error(f"Database error: {error}")

    with tab_delete:
        machines = get_machines()

        if machines.empty:
            st.info("No machines available to delete.")
        else:
            options = {
                f"{row['id']} — {row['name']}": row["id"]
                for _, row in machines.iterrows()
            }

            choice = st.selectbox("Select machine", list(options.keys()), key="delete_machine")
            confirm = st.checkbox("I confirm that I want to delete this machine.")

            if st.button("🗑️ Delete Machine"):
                if confirm:
                    delete_machine(options[choice])
                    st.success("Machine and its maintenance records deleted.")
                    st.rerun()
                else:
                    st.warning("Please confirm before deleting.")


# ==================================================
# 8. AI FAILURE PREDICTION
# ==================================================

elif page == "AI Failure Prediction":
    st.header("🤖 AI Machine Failure Prediction")

    st.write(
        "Enter sensor readings to estimate whether the machine may fail."
    )

    if model is None:
        st.error(
            "The AI model is unavailable. Upload the standard AI4I 2020 "
            "dataset with its original column names."
        )
    else:
        left, right = st.columns(2)

        with left:
            air = st.number_input("Air Temperature (K)", value=298.0)
            process = st.number_input("Process Temperature (K)", value=308.0)
            rpm = st.number_input("Rotational Speed (rpm)", min_value=0.0, value=1500.0)

        with right:
            torque = st.number_input("Torque (Nm)", min_value=0.0, value=40.0)
            wear = st.number_input("Tool Wear (min)", min_value=0.0, value=10.0)

        if st.button("🔍 Predict Failure"):
            if rpm <= 0:
                st.error("Rotational speed must be greater than zero.")
            else:
                try:
                    sample = pd.DataFrame([{
                        "air_temp": air,
                        "process_temp": process,
                        "rpm": rpm,
                        "torque": torque,
                        "tool_wear": wear
                    }])

                    prediction = int(model.predict(sample)[0])
                    probability = float(model.predict_proba(sample)[0][1])

                    st.subheader("Prediction Result")
                    st.metric("Estimated Failure Probability", f"{probability:.1%}")

                    if prediction == 1:
                        st.error("⚠️ AI predicts a possible machine failure.")
                        st.write(
                            "Recommendation: arrange a qualified technician inspection."
                        )
                    else:
                        st.success("✅ AI predicts no failure for these readings.")
                        st.write(
                            "Recommendation: continue routine monitoring and maintenance."
                        )

                    st.caption(
                        "This is a model estimate based on historical data, "
                        "not a guarantee of real-world machine safety."
                    )
                except (ValueError, TypeError) as error:
                    st.error(f"Could not predict from these inputs: {error}")

        if model_accuracy is not None:
            st.subheader("Model Evaluation")
            st.metric("Held-out Test Accuracy", f"{model_accuracy:.1%}")

            st.write("Confusion matrix: rows are actual labels; columns are predictions.")
            st.dataframe(pd.DataFrame(
                model_matrix,
                index=["Actual No Failure", "Actual Failure"],
                columns=["Predicted No Failure", "Predicted Failure"]
            ))

            st.subheader("Feature Importance")
            st.bar_chart(feature_importance.set_index("Feature"))


# ==================================================
# 9. DATA ANALYSIS AND VISUALIZATION
# ==================================================

elif page == "Data Analysis":
    st.header("📈 Data Analysis & Visualization")

    if df.empty:
        st.warning("Place ai4i2020.csv in the same folder as app.py.")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Dataset Details")
        st.write("Rows:", len(df))
        st.write("Columns:", len(df.columns))

        summary = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isna().sum(),
            "Unique Values": df.nunique()
        })
        st.dataframe(summary, use_container_width=True)

        st.subheader("Data Preprocessing")
        cleaned = df.replace([np.inf, -np.inf], np.nan).drop_duplicates().dropna()
        st.write("Original rows:", len(df))
        st.write("Rows after removing missing values and duplicates:", len(cleaned))

        st.download_button(
            "⬇️ Download Cleaned Dataset",
            cleaned.to_csv(index=False).encode("utf-8"),
            "cleaned_machine_data.csv",
            "text/csv"
        )

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().T, use_container_width=True)

        numeric = df.select_dtypes(include=np.number)

        if not numeric.empty:
            st.subheader("Choose a Chart")
            numeric_cols = list(numeric.columns)

            x_col = st.selectbox("X-axis / Category", numeric_cols, key="chart_x")
            y_col = st.selectbox(
                "Y-axis / Value",
                numeric_cols,
                index=min(1, len(numeric_cols) - 1),
                key="chart_y"
            )

            chart_type = st.selectbox(
                "Chart Type", ["Line Chart", "Bar Chart", "Scatter Chart"]
            )

            chart_data = numeric[[x_col, y_col]].dropna().head(500)

            if chart_type == "Line Chart":
                st.line_chart(chart_data.set_index(x_col))
            elif chart_type == "Bar Chart":
                st.bar_chart(chart_data.set_index(x_col))
            else:
                fig, ax = plt.subplots()
                ax.scatter(chart_data[x_col], chart_data[y_col], alpha=0.5)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} vs {x_col}")
                st.pyplot(fig)
                plt.close(fig)

            st.subheader("Correlation Matrix")
            corr = numeric.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(corr, aspect="auto", cmap="Blues")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns, fontsize=7)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if failure_col:
            st.subheader("Failure Distribution")
            failure_counts = df[failure_col].value_counts().sort_index()
            st.bar_chart(failure_counts)


# ==================================================
# 10. STATISTICAL ANALYSIS AND HYPOTHESIS TESTING
# ==================================================

elif page == "Statistical Analysis":
    st.header("🧮 Statistical Analysis")

    if df.empty or failure_col is None or torque_col is None:
        st.warning("A dataset with failure and torque columns is required.")
    else:
        data = df[[failure_col, torque_col]].copy()
        data[failure_col] = pd.to_numeric(data[failure_col], errors="coerce")
        data[torque_col] = pd.to_numeric(data[torque_col], errors="coerce")
        data = data.dropna()

        group0 = data.loc[data[failure_col] == 0, torque_col]
        group1 = data.loc[data[failure_col] == 1, torque_col]

        if len(group0) > 1 and len(group1) > 1:
            st.subheader("Torque: Failure vs No Failure")
            c1, c2 = st.columns(2)
            c1.metric("Mean Torque — No Failure", f"{group0.mean():.2f}")
            c2.metric("Mean Torque — Failure", f"{group1.mean():.2f}")

            test_stat, p_value = stats.ttest_ind(
                group0, group1, equal_var=False, nan_policy="omit"
            )

            st.write(f"Welch t-test statistic: {test_stat:.4f}")
            st.write(f"P-value: {p_value:.6f}")

            if p_value < 0.05:
                st.success(
                    "Reject the equal-means null hypothesis at the 5% significance level."
                )
            else:
                st.info(
                    "There is insufficient evidence to reject the equal-means "
                    "null hypothesis at the 5% significance level."
                )

            st.caption(
                "A statistical difference does not prove that torque causes machine failure."
            )

            values = data[torque_col].to_numpy(dtype=float)
            if len(values) > 1:
                ci_low, ci_high = stats.t.interval(
                    0.95,
                    df=len(values) - 1,
                    loc=np.mean(values),
                    scale=stats.sem(values)
                )
                st.write(
                    f"95% confidence interval for mean torque: "
                    f"{ci_low:.2f} to {ci_high:.2f}"
                )

            st.subheader("Random Sampling Demonstration")
            sample_size = st.slider(
                "Sample size", min_value=10,
                max_value=max(10, min(1000, len(data))),
                value=min(100, len(data))
            )

            sample_size = min(sample_size, len(data))
            sample_data = data.sample(n=sample_size, random_state=42)
            st.write("Sample records:", len(sample_data))
            st.dataframe(sample_data.head(20), use_container_width=True)
            st.write("Sample mean torque:", round(sample_data[torque_col].mean(), 3))
        else:
            st.info("There are not enough records in both failure groups.")


# ==================================================
# 11. MAINTENANCE HISTORY
# ==================================================

elif page == "Maintenance History":
    st.header("🔧 Maintenance History")

    machines = get_machines()
    history = get_maintenance()

    if machines.empty:
        st.info("Add a machine before adding its service history.")
    else:
        machine_options = {
            f"{row['id']} — {row['name']}": int(row["id"])
            for _, row in machines.iterrows()
        }

        with st.form("maintenance_form"):
            chosen = st.selectbox("Machine", list(machine_options.keys()))
            service_date = st.date_input("Service Date", value=date.today())
            description = st.text_area("Maintenance Description")
            cost = st.number_input("Service Cost (₹)", min_value=0.0, value=0.0)
            status = st.selectbox(
                "Service Status", ["Completed", "Pending", "In Progress"]
            )

            if st.form_submit_button("💾 Save Service Record"):
                if not description.strip():
                    st.error("Please enter the maintenance description.")
                else:
                    with connect_db() as con:
                        con.execute("""
                            INSERT INTO maintenance
                            (machine_id, service_date, description, cost, status)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            machine_options[chosen], str(service_date),
                            description.strip(), cost, status
                        ))
                    st.success("Maintenance record saved.")
                    st.rerun()

    st.subheader("Saved Service History")
    history = get_maintenance()

    if history.empty:
        st.info("No service records available.")
    else:
        st.dataframe(history, use_container_width=True)
        st.metric("Total Recorded Maintenance Cost", f"₹{history['cost'].sum():,.2f}")

        record_ids = history["id"].tolist()
        selected_record = st.selectbox(
            "Choose service record to delete", record_ids
        )

        if st.button("Delete Service Record"):
            with connect_db() as con:
                con.execute(
                    "DELETE FROM maintenance WHERE id=?", (selected_record,)
                )
            st.success("Service record deleted.")
            st.rerun()


# ==================================================
# 12. WEB SCRAPING
# ==================================================

elif page == "Web Scraping":
    st.header("🌐 Web Scraping Demo")
    st.write(
        "Demonstration of Requests and BeautifulSoup for reading public webpage headings."
    )

    url = st.text_input(
        "Public webpage URL", placeholder="https://example.com"
    )

    if st.button("Fetch Webpage"):
        parsed = urlparse(url.strip())

        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            st.error("Enter a valid URL starting with https:// or http://.")
        elif BeautifulSoup is None:
            st.error(
                "BeautifulSoup is not installed. Add beautifulsoup4 "
                "to requirements.txt and reboot the app."
            )
        else:
            try:
                response = requests.get(
                    url.strip(),
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                title = (
                    soup.title.get_text(" ", strip=True)
                    if soup.title else "No title found"
                )
                headings = [
                    h.get_text(" ", strip=True)
                    for h in soup.find_all(["h1", "h2", "h3"])[:20]
                ]

                st.success(f"Page title: {title}")
                st.write("Headings found:")
                st.write(headings if headings else "No headings found.")
            except requests.RequestException as error:
                st.error(f"Could not fetch the webpage: {error}")
            except Exception as error:
                st.error(f"Unexpected error: {error}")


# ==================================================
# 13. REPORTS AND CSV EXPORT
# ==================================================

elif page == "Reports":
    st.header("📄 Reports & Data Export")

    machines = get_machines()
    maintenance = get_maintenance()

    st.subheader("Machine Report")
    st.dataframe(machines, use_container_width=True)

    st.download_button(
        "⬇️ Download Machine Report",
        machines.to_csv(index=False).encode("utf-8"),
        "machine_report.csv",
        "text/csv"
    )

    st.divider()

    st.subheader("Maintenance Report")
    st.dataframe(maintenance, use_container_width=True)

    st.download_button(
        "⬇️ Download Maintenance Report",
        maintenance.to_csv(index=False).encode("utf-8"),
        "maintenance_report.csv",
        "text/csv"
    )

    if not df.empty:
        st.divider()
        st.subheader("Original Dataset Export")

        st.download_button(
            "⬇️ Download Dataset CSV",
            df.to_csv(index=False).encode("utf-8"),
            "predictive_maintenance_dataset.csv",
            "text/csv"
        )


# ==================================================
# 14. PROJECT INFORMATION
# ==================================================

elif page == "Project Information":
    st.header("📘 Project Information")

    st.markdown("""
    ### AI-Based Predictive Maintenance System

    **Objective:** Use machine sensor readings and historical data to
    estimate possible machine failure and organize preventive maintenance.

    **Main technologies**
    - Python core concepts, functions, conditions and exception handling
    - Lists, dictionaries, Pandas DataFrames and NumPy arrays
    - SQLite database and CSV file handling
    - CRUD operations for machine records
    - Pandas data cleaning and preprocessing
    - NumPy numerical calculations
    - SciPy statistical analysis, sampling and hypothesis testing
    - Matplotlib data visualization
    - Scikit-learn Random Forest machine-learning model
    - Requests and BeautifulSoup web-scraping demonstration
    - CSV report generation

    **AI model:** Random Forest Classifier

    **Dataset:** AI4I 2020 Predictive Maintenance Dataset

    **Database:** SQLite

    **Interface:** Streamlit

    **Limitations:** Predictions are based on the training dataset.
    They are not guaranteed to predict every real-world failure and
    must not replace qualified industrial inspection.
    """)

    st.info(
        "On a local computer, records are stored in maintenance.db beside app.py. "
        "On Streamlit Cloud, local database files may not persist across "
        "restarts or redeployments."
    )

    st.subheader("Dataset Check")
    if dataset_ready:
        st.success(f"Dataset loaded: {len(df)} rows.")
        st.write("Detected columns:", list(df.columns))
    else:
        st.error(
            "Dataset is missing or not in standard AI4I 2020 format. "
            "Check the filename and original column names."
        )

st.divider()
st.caption("AI-Based Predictive Maintenance System | Python & Streamlit")
