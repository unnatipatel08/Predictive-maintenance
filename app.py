
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ai4i2020.csv")
DB_PATH = os.path.join(BASE_DIR, "maintenance.db")

FEATURES = [
    "temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

CSV_COLUMNS = {
    "UDI": "machine_id",
    "Product ID": "product_id",
    "Type": "machine_type",
    "Air temperature [K]": "temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "failure",
}


# ==========================================================
# 2. PROFESSIONAL LIGHT BLUE THEME
# ==========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F3F7FC;
        color: #1F2937;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #17365D !important;
        font-weight: 700;
    }

    .stApp p,
    .stApp label,
    .stApp li {
        color: #263548;
    }

    [data-testid="stSidebar"] {
        background-color: #E2ECF8;
        border-right: 1px solid #D1DFEF;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #17365D !important;
    }

    [data-testid="stHeader"] {
        background-color: #F3F7FC;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D5E2F0;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(23, 54, 93, 0.05);
    }

    [data-testid="stMetricLabel"] p {
        color: #52657A !important;
    }

    [data-testid="stMetricValue"] {
        color: #1D4ED8 !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #17365D !important;
        font-weight: 600 !important;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background-color: #2563EB;
        color: #FFFFFF !important;
        border: 1px solid #2563EB;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1D4ED8;
        color: #FFFFFF !important;
    }

    input, textarea {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    [data-baseweb="select"] > div {
        background-color: #FFFFFF;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #D5E2F0;
        border-radius: 10px;
    }

    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #D5E2F0;
        border-radius: 10px;
    }

    [data-testid="stDownloadButton"] button {
        background-color: #FFFFFF;
        color: #1D4ED8 !important;
        border: 1px solid #93B4E8;
        border-radius: 8px;
        font-weight: 600;
    }

    [data-testid="stDownloadButton"] button:hover {
        background-color: #EAF2FF;
        color: #1D4ED8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 3. DATABASE
# Stores machine deletion choices and maintenance history.
# The original CSV file is never changed by these actions.
# ==========================================================

def db_connect():
    return sqlite3.connect(DB_PATH, timeout=30)


def initialize_database():
    with db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS deleted_machine_ids (
                machine_id TEXT PRIMARY KEY
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS maintenance_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT NOT NULL,
                product_id TEXT,
                maintenance_date TEXT,
                maintenance_type TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def get_deleted_ids():
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT machine_id FROM deleted_machine_ids"
        ).fetchall()

    return {str(row[0]) for row in rows}


def delete_machine_id(machine_id):
    with db_connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO deleted_machine_ids (machine_id)
            VALUES (?)
            """,
            (str(machine_id),),
        )


def restore_machine_id(machine_id):
    with db_connect() as conn:
        conn.execute(
            "DELETE FROM deleted_machine_ids WHERE machine_id = ?",
            (str(machine_id),),
        )


def save_maintenance_record(
    machine_id,
    product_id,
    maintenance_date,
    maintenance_type,
    notes,
):
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO maintenance_logs
            (
                machine_id,
                product_id,
                maintenance_date,
                maintenance_type,
                notes
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(machine_id),
                str(product_id),
                str(maintenance_date),
                str(maintenance_type),
                str(notes),
            ),
        )


def get_maintenance_history():
    with db_connect() as conn:
        return pd.read_sql_query(
            """
            SELECT
                log_id,
                machine_id,
                product_id,
                maintenance_date,
                maintenance_type,
                notes,
                created_at
            FROM maintenance_logs
            ORDER BY log_id DESC
            """,
            conn,
        )


initialize_database()


# ==========================================================
# 4. READ ORIGINAL CSV
# This DataFrame keeps all original column names and values.
# ==========================================================

if not os.path.isfile(CSV_PATH):
    st.error(
        "ai4i2020.csv was not found. Place it in the same "
        "GitHub repository folder as app.py."
    )
    st.stop()


@st.cache_data
def load_original_csv(path, modified_time):
    return pd.read_csv(path)


try:
    original_data = load_original_csv(
        CSV_PATH,
        os.path.getmtime(CSV_PATH),
    )
except Exception as error:
    st.error(f"Could not read ai4i2020.csv: {error}")
    st.stop()


# ==========================================================
# 5. PREPARE A SEPARATE DATASET FOR ANALYSIS AND AI
# ==========================================================

def prepare_data(raw_data):
    data = raw_data.rename(
        columns=CSV_COLUMNS
    ).copy()

    required_columns = [
        "machine_id",
        "product_id",
        "machine_type",
        "temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
        "failure",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing_columns)
        )

    data["machine_id"] = data["machine_id"].astype(str)
    data["product_id"] = data["product_id"].astype(str)
    data["machine_type"] = data["machine_type"].astype(str)

    numeric_columns = [
        "temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
        "failure",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=required_columns
    )

    data = data[
        data["failure"].isin([0, 1])
    ].copy()

    data["failure"] = data["failure"].astype(int)

    return data


try:
    dataset = prepare_data(original_data)
except Exception as error:
    st.error(f"Could not prepare the AI dataset: {error}")
    st.stop()


# ==========================================================
# 6. MACHINE SELECTION
# ==========================================================

deleted_ids = get_deleted_ids()

available_data = dataset[
    ~dataset["machine_id"].isin(deleted_ids)
].copy()

available_data = available_data.sort_values(
    "machine_id",
    key=lambda values: pd.to_numeric(
        values,
        errors="coerce",
    ),
    na_position="last",
).reset_index(drop=True)

machine_ids = available_data["machine_id"].tolist()


def machine_label(machine_id):
    rows = available_data[
        available_data["machine_id"].astype(str)
        == str(machine_id)
    ]

    if rows.empty:
        return str(machine_id)

    row = rows.iloc[0]

    return (
        f"UDI {row['machine_id']} | "
        f"Product ID {row['product_id']} | "
        f"Type {row['machine_type']}"
    )


def get_machine(machine_id):
    rows = available_data[
        available_data["machine_id"].astype(str)
        == str(machine_id)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


# ==========================================================
# 7. TRAIN RANDOM FOREST MODEL
# ==========================================================

@st.cache_resource
def train_failure_model(path, modified_time):
    raw_data = pd.read_csv(path)
    data = prepare_data(raw_data)

    training_data = data.dropna(
        subset=FEATURES + ["failure"]
    )

    if training_data.empty:
        raise ValueError(
            "No valid records are available for model training."
        )

    X = training_data[FEATURES]
    y = training_data["failure"]

    if y.nunique() < 2:
        raise ValueError(
            "The dataset needs both failure and no-failure records."
        )

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    random_state=42,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X, y)

    return model


def predict_failure(readings):
    model = train_failure_model(
        CSV_PATH,
        os.path.getmtime(CSV_PATH),
    )

    input_data = pd.DataFrame(
        [
            {
                feature: float(readings[feature])
                for feature in FEATURES
            }
        ]
    )

    probabilities = model.predict_proba(input_data)[0]
    model_classes = list(
        model.named_steps["classifier"].classes_
    )

    failure_index = model_classes.index(1)

    risk = float(
        probabilities[failure_index] * 100
    )

    if risk >= 50:
        level = "High Risk"
        color = "red"

    elif risk >= 20:
        level = "Medium Risk"
        color = "yellow"

    else:
        level = "Low Risk"
        color = "green"

    return {
        "risk": risk,
        "level": level,
        "color": color,
        "prediction": (
            "Failure predicted by model"
            if risk >= 50
            else "Failure not predicted by model"
        ),
    }


# ==========================================================
# 8. SMART MAINTENANCE SUGGESTIONS
# ==========================================================

def get_recommendations(readings, risk):
    suggestions = []

    temperature_limit = dataset["temperature"].quantile(0.95)
    speed_low = dataset["rotational_speed"].quantile(0.05)
    speed_high = dataset["rotational_speed"].quantile(0.95)
    torque_limit = dataset["torque"].quantile(0.95)
    wear_limit = dataset["tool_wear"].quantile(0.90)

    if risk >= 50:
        suggestions.append(
            "HIGH RISK: Arrange a qualified machine inspection promptly."
        )

    elif risk >= 20:
        suggestions.append(
            "MEDIUM RISK: Plan a maintenance inspection and monitor readings."
        )

    else:
        suggestions.append(
            "LOW RISK: Continue routine monitoring and regular inspections."
        )

    if readings["temperature"] > temperature_limit:
        suggestions.append(
            "Temperature is high compared with this dataset. "
            "Check cooling and verify the sensor reading."
        )

    if (
        readings["rotational_speed"] < speed_low
        or readings["rotational_speed"] > speed_high
    ):
        suggestions.append(
            "Rotational speed is outside the usual dataset range. "
            "Check operating conditions and the RPM sensor."
        )

    if readings["torque"] > torque_limit:
        suggestions.append(
            "Torque is high compared with this dataset. "
            "Check machine load and tooling."
        )

    if readings["tool_wear"] > wear_limit:
        suggestions.append(
            "Tool wear is high compared with this dataset. "
            "Inspect the tool and review its maintenance schedule."
        )

    return suggestions


# ==========================================================
# 9. SHOW ALERTS
# ==========================================================

def show_health_alert(result):
    risk = result["risk"]

    st.subheader("🚨 Machine Health Alert")

    if result["color"] == "red":
        st.error(
            f"🔴 HIGH RISK — Estimated failure risk: {risk:.2f}%"
        )

    elif result["color"] == "yellow":
        st.warning(
            f"🟡 MEDIUM RISK — Estimated failure risk: {risk:.2f}%"
        )

    else:
        st.success(
            f"🟢 LOW RISK — Estimated failure risk: {risk:.2f}%"
        )

    col1, col2 = st.columns(2)

    col1.metric(
        "Estimated Failure Risk",
        f"{risk:.2f}%",
    )

    col2.metric(
        "Model Result",
        result["prediction"],
    )

    st.caption(
        "The 50% threshold is used to label the model result; "
        "it is not a guarantee that a machine will or will not fail."
    )


# ==========================================================
# 10. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("⚙️ Predictive Maintenance")
st.sidebar.caption("AI-Based Industrial Equipment Monitoring")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Machine Management",
        "Prediction & Analysis",
        "Maintenance Report",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Educational demonstration project")


# ==========================================================
# PAGE 1 — DASHBOARD
# ORIGINAL CSV WITH ALL ORIGINAL COLUMNS
# ==========================================================

if page == "Dashboard":

    st.title("Industrial Machine Dashboard")

    st.write(
        "View and search the original AI4I machine dataset."
    )

    total = len(original_data)

    failures = int(
        (
            pd.to_numeric(
                original_data["Machine failure"],
                errors="coerce",
            ) == 1
        ).sum()
    )

    no_failures = total - failures

    c1, c2, c3 = st.columns(3)

    c1.metric("Total CSV Records", f"{total:,}")
    c2.metric("Recorded Failures", f"{failures:,}")
    c3.metric("No Recorded Failure", f"{no_failures:,}")

    st.subheader("Original Dataset — ai4i2020.csv")

    search = st.text_input(
        "Search Original Dataset",
        placeholder="Search UDI, Product ID, Type, or any value",
        key="dashboard_search",
    )

    if search.strip():
        mask = original_data.astype(str).apply(
            lambda column: column.str.contains(
                search.strip(),
                case=False,
                regex=False,
                na=False,
            )
        ).any(axis=1)

        display_data = original_data.loc[mask]

    else:
        display_data = original_data

    st.caption(
        f"Showing {len(display_data):,} of {total:,} records. "
        "All original CSV columns are retained."
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Original CSV",
        original_data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="ai4i2020.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Recorded Failure Distribution")

    status = (
        original_data["Machine failure"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["No Failure", "Failure"],
        status.values,
    )

    ax.set_ylabel("Number of Records")
    ax.set_title("Recorded Machine Failure")

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Temperature Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
        dataset.loc[
            dataset["failure"] == 0,
            "temperature",
        ].dropna(),
        bins=30,
        alpha=0.6,
        label="No Failure",
    )

    ax.hist(
        dataset.loc[
            dataset["failure"] == 1,
            "temperature",
        ].dropna(),
        bins=30,
        alpha=0.6,
        label="Failure",
    )

    ax.set_xlabel("Air Temperature [K]")
    ax.set_ylabel("Number of Records")
    ax.legend()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# ==========================================================
# PAGE 2 — MACHINE MANAGEMENT
# DELETE USING MACHINE ID + MAINTENANCE HISTORY
# ==========================================================

elif page == "Machine Management":

    st.title("Machine Management")

    st.write(
        "Select a machine by its original ID, remove it from the "
        "operational list, or record maintenance activity."
    )

    st.info(
        "The original ai4i2020.csv is not edited. Deleting a machine "
        "hides it from the operational selector; the full CSV remains "
        "visible on Dashboard."
    )

    st.subheader("Delete Machine")

    if machine_ids:

        id_to_delete = st.selectbox(
            "Select Machine to Delete",
            machine_ids,
            format_func=machine_label,
            key="delete_machine_selection",
        )

        confirm_delete = st.checkbox(
            "Confirm removal from the operational list."
        )

        if st.button("Delete Selected Machine"):

            if not confirm_delete:
                st.warning(
                    "Please confirm before deleting the selected machine."
                )

            else:
                delete_machine_id(id_to_delete)

                st.success(
                    f"Machine UDI {id_to_delete} removed from the "
                    "operational list."
                )

                st.rerun()

    else:
        st.info(
            "No available machines. Check the CSV or restore a machine below."
        )

    st.subheader("Restore Machine")

    deleted_ids = get_deleted_ids()

    if deleted_ids:

        restore_id = st.selectbox(
            "Select Machine to Restore",
            sorted(
                deleted_ids,
                key=lambda value: (
                    not value.isdigit(),
                    int(value) if value.isdigit() else value,
                ),
            ),
            key="restore_machine_selection",
        )

        if st.button("Restore Selected Machine"):
            restore_machine_id(restore_id)
            st.success(f"Machine UDI {restore_id} restored.")
            st.rerun()

    else:
        st.caption("There are no removed machines to restore.")

    st.divider()
    st.subheader("Add Maintenance Record")

    if machine_ids:

        maintenance_id = st.selectbox(
            "Select Machine for Maintenance",
            machine_ids,
            format_func=machine_label,
            key="maintenance_machine_selection",
        )

        machine_record = get_machine(maintenance_id)

        maintenance_date = st.date_input(
            "Maintenance Date"
        )

        maintenance_type = st.selectbox(
            "Maintenance Type",
            [
                "Inspection",
                "Cleaning",
                "Tool Replacement",
                "Repair",
                "Calibration",
                "Other",
            ],
        )

        notes = st.text_area(
            "Maintenance Notes",
            placeholder="Describe inspection or service work...",
        )

        if st.button("Save Maintenance Record"):

            if not notes.strip():
                st.warning("Enter maintenance notes before saving.")

            else:
                save_maintenance_record(
                    maintenance_id,
                    machine_record["product_id"],
                    str(maintenance_date),
                    maintenance_type,
                    notes.strip(),
                )

                st.success("Maintenance record saved successfully.")
                st.rerun()

    else:
        st.info(
            "No available machine to attach a maintenance record to."
        )

    st.subheader("Maintenance History")

    history = get_maintenance_history()

    if history.empty:

        st.info("No maintenance records have been added yet.")

    else:

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Maintenance History",
            history.to_csv(index=False).encode("utf-8"),
            file_name="maintenance_history.csv",
            mime="text/csv",
        )


# ==========================================================
# PAGE 3 — PREDICTION & ANALYSIS
# NO FULL DATASET TABLE ON THIS PAGE
# ==========================================================

elif page == "Prediction & Analysis":

    st.title("Prediction & Analysis")

    st.write(
        "Choose an existing machine. Enter new readings if you want "
        "to test another scenario; blank fields use its CSV readings."
    )

    if not machine_ids:

        st.warning(
            "No available machines. Check the dataset or restore a machine."
        )

    else:

        selected_id = st.selectbox(
            "Select Machine",
            machine_ids,
            format_func=machine_label,
            key="prediction_machine_selection",
        )

        selected_machine = get_machine(selected_id)

        st.subheader("Selected Machine's CSV Readings")

        sensor_cols = st.columns(4)

        for index, feature in enumerate(FEATURES):
            with sensor_cols[index]:
                sensor_cols[index].metric(
                    feature.replace("_", " ").title(),
                    f"{float(selected_machine[feature]):.2f}",
                )

        st.caption(
            "Leave a field blank to use the selected machine's original "
            "CSV value."
        )

        with st.form("manual_prediction_form"):

            st.subheader("Manual Machine Prediction")

            st.write(
                "Enter new sensor values only when you want to test "
                "a different scenario."
            )

            left, right = st.columns(2)

            with left:

                st.markdown("**Air Temperature [K]**")
                st.caption("Example: 298.1")

                temp_input = st.text_input(
                    "Temperature Value",
                    placeholder="Enter temperature, e.g. 298.1",
                    label_visibility="collapsed",
                )

                st.markdown("**Rotational Speed [rpm]**")
                st.caption("Example: 1551")

                speed_input = st.text_input(
                    "Rotational Speed Value",
                    placeholder="Enter speed, e.g. 1551",
                    label_visibility="collapsed",
                )

            with right:

                st.markdown("**Torque [Nm]**")
                st.caption("Example: 42.8")

                torque_input = st.text_input(
                    "Torque Value",
                    placeholder="Enter torque, e.g. 42.8",
                    label_visibility="collapsed",
                )

                st.markdown("**Tool Wear [min]**")
                st.caption("Example: 0")

                wear_input = st.text_input(
                    "Tool Wear Value",
                    placeholder="Enter tool wear, e.g. 0",
                    label_visibility="collapsed",
                )

            submitted = st.form_submit_button(
                "Predict Failure Risk"
            )

        if submitted:

            try:

                # Blank fields use the selected machine's dataset readings.
                readings = {
                    "temperature": float(
                        temp_input.strip()
                        if temp_input.strip()
                        else selected_machine["temperature"]
                    ),

                    "rotational_speed": float(
                        speed_input.strip()
                        if speed_input.strip()
                        else selected_machine["rotational_speed"]
                    ),

                    "torque": float(
                        torque_input.strip()
                        if torque_input.strip()
                        else selected_machine["torque"]
                    ),

                    "tool_wear": float(
                        wear_input.strip()
                        if wear_input.strip()
                        else selected_machine["tool_wear"]
                    ),
                }

                if not all(
                    np.isfinite(value)
                    for value in readings.values()
                ):
                    raise ValueError(
                        "All sensor values must be valid finite numbers."
                    )

                if readings["temperature"] <= 0:
                    raise ValueError(
                        "Temperature must be greater than zero Kelvin."
                    )

                if readings["rotational_speed"] < 0:
                    raise ValueError(
                        "Rotational speed cannot be negative."
                    )

                if readings["torque"] < 0:
                    raise ValueError(
                        "Torque cannot be negative."
                    )

                if readings["tool_wear"] < 0:
                    raise ValueError(
                        "Tool wear cannot be negative."
                    )

                prediction = predict_failure(readings)

                show_health_alert(prediction)

                st.subheader("Smart Maintenance Suggestions")

                suggestions = get_recommendations(
                    readings,
                    prediction["risk"],
                )

                for suggestion in suggestions:
                    st.write(f"• {suggestion}")

                report = pd.DataFrame(
                    [
                        {
                            "UDI": selected_machine["machine_id"],
                            "Product ID": selected_machine["product_id"],
                            "Type": selected_machine["machine_type"],
                            "Air temperature [K]": readings["temperature"],
                            "Rotational speed [rpm]": readings[
                                "rotational_speed"
                            ],
                            "Torque [Nm]": readings["torque"],
                            "Tool wear [min]": readings["tool_wear"],
                            "Estimated Failure Risk (%)": prediction["risk"],
                            "Risk Category": prediction["level"],
                            "Model Result": prediction["prediction"],
                            "Maintenance Suggestions": " | ".join(suggestions),
                        }
                    ]
                )

                st.download_button(
                    "Download Prediction Report",
                    report.to_csv(index=False).encode("utf-8"),
                    file_name=f"prediction_report_{selected_id}.csv",
                    mime="text/csv",
                )

            except ValueError as error:

                st.error(str(error))

            except Exception as error:

                st.error(f"Prediction failed: {error}")

    # ------------------------------------------------------
    # ANALYSIS: CHARTS, FEATURE IMPORTANCE AND STATISTICS
    # ------------------------------------------------------

    st.divider()
    st.subheader("Machine Failure Analysis")

    valid_data = dataset.dropna(
        subset=FEATURES + ["failure"]
    )

    try:

        st.markdown("**Random Forest Feature Importance**")

        model = train_failure_model(
            CSV_PATH,
            os.path.getmtime(CSV_PATH),
        )

        importances = model.named_steps[
            "classifier"
        ].feature_importances_

        importance_df = pd.DataFrame(
            {
                "Feature": FEATURES,
                "Importance": importances,
            }
        ).sort_values(
            "Importance",
            ascending=False,
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            importance_df["Feature"],
            importance_df["Importance"],
        )

        ax.set_ylabel("Importance")
        ax.set_xlabel("Sensor Feature")
        ax.tick_params(axis="x", rotation=20)

        fig.tight_layout()

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as error:

        st.warning(
            f"Feature importance unavailable: {error}"
        )

    st.subheader("Statistical Summary")

    st.dataframe(
        valid_data[FEATURES].describe().T,
        use_container_width=True,
    )

    st.subheader("Feature Correlation")

    correlation = valid_data[FEATURES].corr()

    fig, ax = plt.subplots(figsize=(7, 4))

    image = ax.imshow(
        correlation,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(
        FEATURES,
        rotation=25,
        ha="right",
    )

    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Temperature Hypothesis Test")

    no_failure_temp = valid_data.loc[
        valid_data["failure"] == 0,
        "temperature",
    ].dropna()

    failure_temp = valid_data.loc[
        valid_data["failure"] == 1,
        "temperature",
    ].dropna()

    if len(no_failure_temp) >= 2 and len(failure_temp) >= 2:

        test_result = ttest_ind(
            no_failure_temp,
            failure_temp,
            equal_var=False,
        )

        test_col1, test_col2 = st.columns(2)

        test_col1.metric(
            "Mean Temperature — No Failure",
            f"{no_failure_temp.mean():.2f} K",
        )

        test_col2.metric(
            "Mean Temperature — Recorded Failure",
            f"{failure_temp.mean():.2f} K",
        )

        st.write(
            f"**t-statistic:** {test_result.statistic:.3f}"
        )

        st.write(
            f"**p-value:** {test_result.pvalue:.4f}"
        )

        if test_result.pvalue < 0.05:

            st.info(
                "The group means differ significantly at the 5% level."
            )

        else:

            st.info(
                "The test did not find a significant difference at the 5% level."
            )

    else:

        st.info(
            "There are not enough records for the hypothesis test."
        )

    st.caption(
        "The full original CSV table is intentionally shown only on "
        "Dashboard, not on Prediction & Analysis."
    )


# ==========================================================
# PAGE 4 — PERSONALIZED MAINTENANCE REPORT
# ==========================================================

elif page == "Maintenance Report":

    st.title("Maintenance Report & Recommendations")

    st.write(
        "Select a machine to view its sensor readings, failure-risk "
        "estimate, alert, recommendations, and downloadable report."
    )

    if not machine_ids:

        st.warning(
            "No available machines. Check the dataset or restore a machine."
        )

    else:

        report_id = st.selectbox(
            "Select Machine",
            machine_ids,
            format_func=machine_label,
            key="report_machine_selection",
        )

        machine = get_machine(report_id)

        st.subheader(f"Maintenance Report — UDI {report_id}")

        st.write(f"**Product ID:** {machine['product_id']}")
        st.write(f"**Machine Type:** {machine['machine_type']}")

        report_readings = {
            "temperature": float(machine["temperature"]),
            "rotational_speed": float(machine["rotational_speed"]),
            "torque": float(machine["torque"]),
            "tool_wear": float(machine["tool_wear"]),
        }

        readings_table = pd.DataFrame(
            [
                {
                    "Sensor": "Air Temperature [K]",
                    "Reading": report_readings["temperature"],
                },
                {
                    "Sensor": "Rotational Speed [rpm]",
                    "Reading": report_readings["rotational_speed"],
                },
                {
                    "Sensor": "Torque [Nm]",
                    "Reading": report_readings["torque"],
                },
                {
                    "Sensor": "Tool Wear [min]",
                    "Reading": report_readings["tool_wear"],
                },
            ]
        )

        st.dataframe(
            readings_table,
            use_container_width=True,
            hide_index=True,
        )

        try:

            prediction = predict_failure(report_readings)

            show_health_alert(prediction)

            st.subheader("Smart Maintenance Suggestions")

            suggestions = get_recommendations(
                report_readings,
                prediction["risk"],
            )

            for suggestion in suggestions:
                st.write(f"• {suggestion}")

            report = pd.DataFrame(
                [
                    {
                        "UDI": machine["machine_id"],
                        "Product ID": machine["product_id"],
                        "Type": machine["machine_type"],
                        "Air temperature [K]": report_readings[
                            "temperature"
                        ],
                        "Rotational speed [rpm]": report_readings[
                            "rotational_speed"
                        ],
                        "Torque [Nm]": report_readings["torque"],
                        "Tool wear [min]": report_readings["tool_wear"],
                        "Recorded Failure": int(machine["failure"]),
                        "Estimated Failure Risk (%)": prediction["risk"],
                        "Risk Category": prediction["level"],
                        "Model Result": prediction["prediction"],
                        "Maintenance Suggestions": " | ".join(suggestions),
                    }
                ]
            )

            st.download_button(
                "Download Personalized Maintenance Report",
                report.to_csv(index=False).encode("utf-8"),
                file_name=f"maintenance_report_{report_id}.csv",
                mime="text/csv",
            )

        except Exception as error:

            st.error(
                f"Could not create the maintenance report: {error}"
            )

    st.divider()
    st.subheader("How Sensor Data Reaches This App")

    st.info(
        "This version uses historical readings from ai4i2020.csv. "
        "It is not yet connected to live physical sensors."
    )

    st.markdown(
        """
        **Future live-sensor workflow**

        1. Sensors measure temperature, rotational speed, and torque.
        2. An ESP32 or industrial PLC collects the measurements.
        3. An API or MQTT service sends the readings over a network.
        4. This Streamlit app receives new measurements, predicts risk,
           and displays health alerts.

        Tool wear may need operating-time tracking, inspection, or a
        specialised sensor. It is not automatically supplied by an
        ordinary temperature sensor.
        """
    )

    st.caption(
        "This is an educational demonstration. The model has not been "
        "validated for industrial safety decisions. A real-time alert "
        "requires an actual data connection and monitoring process."
    )
