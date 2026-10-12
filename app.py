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


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

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


# =========================================================
# 2. LIGHT BLUE THEME
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F3F7FC;
        color: #1F2937;
    }

    h1, h2, h3, h4 {
        color: #17365D !important;
    }

    [data-testid="stSidebar"] {
        background-color: #E2ECF8;
    }

    [data-testid="stSidebar"] * {
        color: #17365D;
    }

    [data-testid="stHeader"] {
        background-color: #F3F7FC;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D5E2F0;
        padding: 16px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #1D4ED8;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: #2563EB;
        color: white !important;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #1D4ED8;
        color: white !important;
    }

    input, textarea {
        background: white !important;
        color: #1F2937 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. DATABASE FOR DELETED IDS AND MAINTENANCE HISTORY
# =========================================================

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


def load_maintenance_history():
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


# =========================================================
# 4. LOAD ORIGINAL CSV WITHOUT CHANGING ITS COLUMNS
# =========================================================

@st.cache_data
def load_original_csv(path, modified_time):
    return pd.read_csv(path)


if not os.path.isfile(CSV_PATH):
    st.error(
        "ai4i2020.csv not found. Place it in the same "
        "GitHub repository folder as app.py."
    )
    st.stop()


try:
    original_data = load_original_csv(
        CSV_PATH,
        os.path.getmtime(CSV_PATH),
    )
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()


def prepare_data(raw):
    """Make a separate working copy for prediction and analysis."""

    data = raw.rename(columns=CSV_COLUMNS).copy()

    required = [
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

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            "Missing required original CSV columns: "
            + ", ".join(missing)
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

    return data


try:
    dataset = prepare_data(original_data)
except Exception as error:
    st.error(str(error))
    st.stop()


# =========================================================
# 5. MACHINE LIST AND DROPDOWN LABELS
# =========================================================

deleted_ids = get_deleted_ids()

available_data = dataset[
    ~dataset["machine_id"].isin(deleted_ids)
].copy()

available_data = available_data.sort_values(
    by="machine_id",
    key=lambda column: pd.to_numeric(
        column,
        errors="coerce",
    ),
    na_position="last",
).reset_index(drop=True)


def get_machine_label(machine_id):
    rows = available_data[
        available_data["machine_id"].astype(str)
        == str(machine_id)
    ]

    if rows.empty:
        return str(machine_id)

    row = rows.iloc[0]

    return (
        f"UDI {row['machine_id']} | "
        f"Product {row['product_id']} | "
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


machine_ids = available_data["machine_id"].tolist()


# =========================================================
# 6. RANDOM FOREST MACHINE LEARNING MODEL
# =========================================================

@st.cache_resource
def train_failure_model(path, modified_time):
    raw = pd.read_csv(path)
    data = prepare_data(raw)

    model_data = data.dropna(
        subset=FEATURES + ["failure"]
    )

    if model_data.empty:
        raise ValueError(
            "The CSV has no valid training records."
        )

    X = model_data[FEATURES]
    y = model_data["failure"].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The training data must contain both failure "
            "and no-failure records."
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

    X = pd.DataFrame(
        [
            {
                feature: readings[feature]
                for feature in FEATURES
            }
        ]
    )

    probabilities = model.predict_proba(X)[0]

    classes = model.named_steps["classifier"].classes_

    failure_index = list(classes).index(1)
    risk = float(probabilities[failure_index] * 100)

    # Health alert categories
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
            "Failure predicted"
            if risk >= 50
            else "Failure not predicted"
        ),
    }


# =========================================================
# 7. SMART MAINTENANCE RECOMMENDATIONS
# =========================================================

def get_recommendations(readings, risk):
    recommendations = []

    temp_limit = dataset["temperature"].quantile(0.95)
    speed_low = dataset["rotational_speed"].quantile(0.05)
    speed_high = dataset["rotational_speed"].quantile(0.95)
    torque_limit = dataset["torque"].quantile(0.95)
    wear_limit = dataset["tool_wear"].quantile(0.90)

    if risk >= 50:
        recommendations.append(
            "HIGH RISK: Arrange a qualified machine inspection "
            "as soon as practical."
        )

    elif risk >= 20:
        recommendations.append(
            "MEDIUM RISK: Plan a maintenance inspection "
            "and continue monitoring the sensor readings."
        )

    else:
        recommendations.append(
            "LOW RISK: Continue routine inspection and "
            "regular monitoring."
        )

    if readings["temperature"] > temp_limit:
        recommendations.append(
            "Temperature is unusually high compared with "
            "this dataset. Inspect the cooling system and "
            "verify the temperature sensor."
        )

    if (
        readings["rotational_speed"] < speed_low
        or readings["rotational_speed"] > speed_high
    ):
        recommendations.append(
            "Rotational speed is outside the usual dataset range. "
            "Check the operating conditions and RPM sensor."
        )

    if readings["torque"] > torque_limit:
        recommendations.append(
            "Torque is unusually high compared with this dataset. "
            "Check machine load and tooling."
        )

    if readings["tool_wear"] > wear_limit:
        recommendations.append(
            "Tool wear is high compared with this dataset. "
            "Inspect the tool and consider scheduled replacement."
        )

    return recommendations


# =========================================================
# 8. SIDEBAR NAVIGATION
# =========================================================

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


# =========================================================
# 9. PAGE: DASHBOARD
# Show ALL original CSV records and original columns.
# =========================================================

if page == "Dashboard":

    st.title("Industrial Machine Dashboard")

    st.write(
        "View the original AI4I dataset and its recorded failures."
    )

    total = len(original_data)

    if "Machine failure" in original_data.columns:

        failure_count = int(
            (
                pd.to_numeric(
                    original_data["Machine failure"],
                    errors="coerce",
                ) == 1
            ).sum()
        )

    else:
        failure_count = 0

    healthy_count = total - failure_count

    c1, c2, c3 = st.columns(3)

    c1.metric("Total CSV Records", f"{total:,}")
    c2.metric("Recorded Failures", f"{failure_count:,}")
    c3.metric("No Recorded Failure", f"{healthy_count:,}")

    st.subheader("Original Dataset — ai4i2020.csv")

    search = st.text_input(
        "Search in original dataset",
        placeholder="Search UDI, Product ID, Type, or any value",
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

        displayed_data = original_data.loc[mask]

    else:
        displayed_data = original_data

    st.caption(
        f"Showing {len(displayed_data):,} of {total:,} rows. "
        "All original CSV columns are retained."
    )

    st.dataframe(
        displayed_data,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Original CSV",
        original_data.to_csv(index=False).encode("utf-8"),
        file_name="ai4i2020.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Failure Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    status = (
        original_data["Machine failure"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    ax.bar(
        ["No Failure", "Failure"],
        status.values,
    )

    ax.set_ylabel("Number of Records")
    ax.set_title("Recorded Machine Failure Distribution")

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Temperature Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    for failure_value, label in [
        (0, "No Failure"),
        (1, "Failure"),
    ]:

        values = dataset.loc[
            dataset["failure"] == failure_value,
            "temperature",
        ].dropna()

        ax.hist(
            values,
            bins=30,
            alpha=0.6,
            label=label,
        )

    ax.set_xlabel("Air Temperature [K]")
    ax.set_ylabel("Number of Records")
    ax.legend()

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# =========================================================
# 10. PAGE: MACHINE MANAGEMENT
# Delete a machine by ID and add maintenance history.
# No sensor values are manually stored in the CSV.
# =========================================================

elif page == "Machine Management":

    st.title("Machine Management")

    st.write(
        "Manage available machine records and record maintenance activity."
    )

    st.info(
        "Machine sensor readings come from the repository CSV in this "
        "demo. The original CSV is not edited when a machine is removed "
        "from the operational list."
    )

    st.subheader("Delete Machine Using Machine ID")

    if machine_ids:

        id_to_delete = st.selectbox(
            "Select Machine",
            machine_ids,
            format_func=get_machine_label,
            key="delete_machine_selection",
        )

        confirmation = st.checkbox(
            "Confirm that I want to remove this machine "
            "from the operational list."
        )

        if st.button("Delete Selected Machine"):

            if not confirmation:
                st.warning(
                    "Please confirm before deleting."
                )

            else:
                delete_machine_id(id_to_delete)

                st.success(
                    f"Machine {id_to_delete} removed from the "
                    "operational list."
                )

                st.info(
                    "The original ai4i2020.csv file remains unchanged "
                    "and continues to appear in the Dashboard."
                )

                st.rerun()

    else:

        st.info(
            "No active machines. Restore machines below or check the CSV."
        )

    st.subheader("Restore a Removed Machine")

    if deleted_ids:

        restore_id = st.selectbox(
            "Select machine to restore",
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

            st.success(
                f"Machine {restore_id} restored."
            )

            st.rerun()

    else:

        st.caption("There are no removed machines to restore.")

    st.divider()
    st.subheader("Add Maintenance Record")

    st.write(
        "Select a machine and record its inspection, cleaning, "
        "repair, or other maintenance work."
    )

    if machine_ids:

        maintenance_id = st.selectbox(
            "Machine for Maintenance",
            machine_ids,
            format_func=get_machine_label,
            key="maintenance_machine_selection",
        )

        selected_for_maintenance = get_machine(
            maintenance_id
        )

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
            placeholder="Describe the maintenance performed",
        )

        if st.button("Save Maintenance Record"):

            if not notes.strip():
                st.warning(
                    "Please enter maintenance notes."
                )

            else:
                save_maintenance_record(
                    maintenance_id,
                    selected_for_maintenance["product_id"],
                    str(maintenance_date),
                    maintenance_type,
                    notes.strip(),
                )

                st.success(
                    "Maintenance record saved successfully."
                )

                st.rerun()

    else:

        st.info(
            "There are no available machines for maintenance records."
        )

    st.subheader("Maintenance History")

    history = load_maintenance_history()

    if history.empty:

        st.info(
            "No maintenance records have been saved yet."
        )

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


# =========================================================
# 11. PAGE: PREDICTION & ANALYSIS
# No full CSV table on this page.
# =========================================================

elif page == "Prediction & Analysis":

    st.title("Prediction & Analysis")

    st.write(
        "Choose a machine and optionally enter new sensor readings. "
        "Blank fields use the selected machine's CSV readings."
    )

    if not machine_ids:

        st.warning(
            "No available machines. Check the CSV or restore a machine "
            "from Machine Management."
        )

    else:

        selected_id = st.selectbox(
            "Select Machine",
            machine_ids,
            format_func=get_machine_label,
            key="prediction_machine_selection",
        )

        selected_machine = get_machine(selected_id)

        st.subheader("Selected Machine's CSV Readings")

        sensor_columns = st.columns(4)

        for index, feature in enumerate(FEATURES):

            with sensor_columns[index]:
                st.metric(
                    feature.replace("_", " ").title(),
                    f"{float(selected_machine[feature]):.2f}",
                )

        st.caption(
            "These readings are taken from the selected CSV record. "
            "No manual entry is needed to use this machine's readings."
        )

        with st.form("manual_prediction_form"):

            st.subheader("Manual Machine Prediction")

            st.write(
                "Enter new readings only to test a different scenario. "
                "Leave a field empty to use that CSV value."
            )

            left, right = st.columns(2)

            with left:

                temp_input = st.text_input(
                    "Air Temperature [K]",
                    placeholder="Blank = CSV value",
                )

                speed_input = st.text_input(
                    "Rotational Speed [rpm]",
                    placeholder="Blank = CSV value",
                )

            with right:

                torque_input = st.text_input(
                    "Torque [Nm]",
                    placeholder="Blank = CSV value",
                )

                wear_input = st.text_input(
                    "Tool Wear [min]",
                    placeholder="Blank = CSV value",
                )

            submitted = st.form_submit_button(
                "Predict Failure Risk"
            )

        if submitted:

            try:

                sensor_readings = {
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

                if any(
                    not np.isfinite(value)
                    for value in sensor_readings.values()
                ):
                    raise ValueError(
                        "All sensor values must be finite numbers."
                    )

                if sensor_readings["rotational_speed"] < 0:
                    raise ValueError(
                        "Rotational speed cannot be negative."
                    )

                if sensor_readings["torque"] < 0:
                    raise ValueError(
                        "Torque cannot be negative."
                    )

                if sensor_readings["tool_wear"] < 0:
                    raise ValueError(
                        "Tool wear cannot be negative."
                    )

                result = predict_failure(sensor_readings)

                st.divider()
                st.subheader("Machine Health Alert")

                if result["color"] == "red":

                    st.error(
                        f"🔴 {result['level']} — "
                        f"{result['risk']:.2f}% estimated risk"
                    )

                elif result["color"] == "yellow":

                    st.warning(
                        f"🟡 {result['level']} — "
                        f"{result['risk']:.2f}% estimated risk"
                    )

                else:

                    st.success(
                        f"🟢 {result['level']} — "
                        f"{result['risk']:.2f}% estimated risk"
                    )

                result_col1, result_col2 = st.columns(2)

                result_col1.metric(
                    "Estimated Failure Risk",
                    f"{result['risk']:.2f}%",
                )

                result_col2.metric(
                    "Model Result",
                    result["prediction"],
                )

                st.subheader("Smart Maintenance Suggestions")

                suggestions = get_recommendations(
                    sensor_readings,
                    result["risk"],
                )

                for suggestion in suggestions:
                    st.write(f"• {suggestion}")

                report = pd.DataFrame(
                    [
                        {
                            "UDI": selected_machine["machine_id"],
                            "Product ID": selected_machine["product_id"],
                            "Type": selected_machine["machine_type"],
                            "Air temperature [K]": sensor_readings[
                                "temperature"
                            ],
                            "Rotational speed [rpm]": sensor_readings[
                                "rotational_speed"
                            ],
                            "Torque [Nm]": sensor_readings["torque"],
                            "Tool wear [min]": sensor_readings["tool_wear"],
                            "Estimated Failure Risk (%)": result["risk"],
                            "Risk Category": result["level"],
                            "Model Result": result["prediction"],
                            "Maintenance Suggestions": " | ".join(
                                suggestions
                            ),
                        }
                    ]
                )

                st.download_button(
                    "Download Prediction Report",
                    report.to_csv(index=False).encode("utf-8"),
                    file_name=f"prediction_report_{selected_id}.csv",
                    mime="text/csv",
                )

                st.caption(
                    "This is a model estimate trained on historical sample "
                    "data, not live sensor data or a guarantee of failure."
                )

            except ValueError as error:

                st.error(str(error))

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

    # -----------------------------------------------------
    # ANALYSIS SECTION
    # -----------------------------------------------------

    st.divider()
    st.subheader("Statistical Analysis")

    valid_data = dataset.dropna(
        subset=FEATURES + ["failure"]
    )

    if not valid_data.empty:

        st.markdown("**Random Forest Feature Importance**")

        try:

            model = train_failure_model(
                CSV_PATH,
                os.path.getmtime(CSV_PATH),
            )

            importance = model.named_steps[
                "classifier"
            ].feature_importances_

            importance_df = pd.DataFrame(
                {
                    "Feature": FEATURES,
                    "Importance": importance,
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
            ax.set_xlabel("Feature")

            ax.tick_params(
                axis="x",
                rotation=20,
            )

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

        st.markdown("**Statistical Summary**")

        st.dataframe(
            valid_data[FEATURES].describe().T,
            use_container_width=True,
        )

        st.markdown("**Feature Correlation**")

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

        st.dataframe(
            correlation,
            use_container_width=True,
        )

        st.markdown("**Temperature Hypothesis Test**")

        no_failure_temp = valid_data.loc[
            valid_data["failure"] == 0,
            "temperature",
        ].dropna()

        failure_temp = valid_data.loc[
            valid_data["failure"] == 1,
            "temperature",
        ].dropna()

        if (
            len(no_failure_temp) >= 2
            and len(failure_temp) >= 2
        ):

            test_result = ttest_ind(
                no_failure_temp,
                failure_temp,
                equal_var=False,
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Mean Temperature — No Failure",
                f"{no_failure_temp.mean():.2f} K",
            )

            c2.metric(
                "Mean Temperature — Failure",
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
                    "The difference between group means is statistically "
                    "significant at the 5% level."
                )

            else:

                st.info(
                    "The test did not find a statistically significant "
                    "difference at the 5% level."
                )

        else:

            st.info(
                "Not enough records to perform the hypothesis test."
            )


# =========================================================
# 12. PAGE: PERSONALIZED MAINTENANCE REPORT
# =========================================================

elif page == "Maintenance Report":

    st.title("Maintenance Report & Recommendations")

    st.write(
        "Select a machine to create an individual report "
        "with readings, risk, and maintenance advice."
    )

    if not machine_ids:

        st.warning(
            "No available machines. Check the CSV or restore a machine."
        )

    else:

        report_machine_id = st.selectbox(
            "Select Machine",
            machine_ids,
            format_func=get_machine_label,
            key="report_machine_selection",
        )

        machine = get_machine(report_machine_id)

        st.subheader(
            f"Maintenance Report — UDI {report_machine_id}"
        )

        st.write(
            f"**Product ID:** {machine['product_id']}"
        )

        st.write(
            f"**Machine Type:** {machine['machine_type']}"
        )

        report_readings = {
            "temperature": float(machine["temperature"]),
            "rotational_speed": float(machine["rotational_speed"]),
            "torque": float(machine["torque"]),
            "tool_wear": float(machine["tool_wear"]),
        }

        reading_df = pd.DataFrame(
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
            reading_df,
            use_container_width=True,
            hide_index=True,
        )

        try:

            result = predict_failure(report_readings)

            if result["color"] == "red":

                st.error(
                    f"🔴 {result['level']} — "
                    f"{result['risk']:.2f}% estimated risk"
                )

            elif result["color"] == "yellow":

                st.warning(
                    f"🟡 {result['level']} — "
                    f"{result['risk']:.2f}% estimated risk"
                )

            else:

                st.success(
                    f"🟢 {result['level']} — "
                    f"{result['risk']:.2f}% estimated risk"
                )

            suggestions = get_recommendations(
                report_readings,
                result["risk"],
            )

            st.subheader("Smart Maintenance Suggestions")

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
                        "Estimated Failure Risk (%)": result["risk"],
                        "Risk Category": result["level"],
                        "Model Result": result["prediction"],
                        "Maintenance Suggestions": " | ".join(suggestions),
                    }
                ]
            )

            st.download_button(
                "Download Personalized Maintenance Report",
                report.to_csv(index=False).encode("utf-8"),
                file_name=f"maintenance_report_{report_machine_id}.csv",
                mime="text/csv",
            )

        except Exception as error:

            st.error(
                f"Could not create the report: {error}"
            )

    st.divider()
    st.subheader("How Sensor Data Reaches This App")

    st.info(
        "Currently this project reads historical readings from "
        "ai4i2020.csv. It is not connected to physical sensors."
    )

    st.markdown(
        """
        **Actual sensor data flow for a future version:**

        1. Sensors measure temperature, rotational speed and torque
           on real equipment.
        2. A controller such as an ESP32 or an industrial PLC collects
           the readings.
        3. The controller sends readings using Wi-Fi/Ethernet to an
           API or MQTT service.
        4. The Streamlit application receives those readings, sends
           them to the trained model, and displays alerts.

        Tool wear is often measured or estimated using tool monitoring,
        operating time, inspection or specialized sensors. It is not
        automatically available from an ordinary temperature sensor.
        """
    )

    st.caption(
        "A real sensor connection requires suitable hardware, a data "
        "transmission method, and additional integration code. "
        "Current predictions are educational estimates, not validated "
        "industrial safety decisions."
    )
