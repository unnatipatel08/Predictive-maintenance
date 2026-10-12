
import os
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from database import (
    initialize_database,
    get_machines,
    delete_machine,
    save_maintenance,
    get_maintenance,
    import_csv_data,
)

from machine_learning import (
    train_model,
    predict_risk,
    get_feature_importance,
)

from graph_analysis import (
    machine_status_chart,
    temperature_chart,
    feature_correlation,
    statistical_summary,
    temperature_hypothesis_test,
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)


# ==========================================================
# PROFESSIONAL LIGHT BLUE THEME
# ==========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F3F7FC;
        color: #1F2937;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #17365D !important;
        font-weight: 700;
    }

    .stApp p, .stApp label, .stApp li {
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
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(23, 54, 93, 0.05);
    }

    [data-testid="stMetricLabel"] p {
        color: #52657A !important;
    }

    [data-testid="stMetricValue"] {
        color: #1D4ED8 !important;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background-color: #2563EB;
        color: #FFFFFF !important;
        border: 1px solid #2563EB;
        border-radius: 9px;
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
        border-radius: 9px;
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
# CONSTANTS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ai4i2020.csv")
DB_PATH = os.path.join(BASE_DIR, "maintenance.db")

FEATURES = [
    "temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

REQUIRED_COLUMNS = [
    "machine_id",
    "temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "failure",
]


# ==========================================================
# DELETE HISTORY
# Remember deleted machines so they are not automatically
# imported again during ordinary app reruns.
# ==========================================================

def initialize_deleted_table():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS deleted_machine_ids (
                machine_id TEXT PRIMARY KEY
            )
            """
        )


def get_deleted_ids():
    initialize_deleted_table()

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT machine_id FROM deleted_machine_ids"
        ).fetchall()

    return {str(row[0]) for row in rows}


def remember_deleted_id(machine_id):
    initialize_deleted_table()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO deleted_machine_ids (machine_id)
            VALUES (?)
            """,
            (str(machine_id),),
        )


# ==========================================================
# LOAD THE ORIGINAL CSV
# The displayed original dataset is never renamed or edited.
# A separate copy is prepared for analysis and prediction.
# ==========================================================

def prepare_dataset(original_data):
    renamed = original_data.rename(
        columns={
            "UDI": "machine_id",
            "Air temperature [K]": "temperature",
            "Rotational speed [rpm]": "rotational_speed",
            "Torque [Nm]": "torque",
            "Tool wear [min]": "tool_wear",
            "Machine failure": "failure",
        }
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in renamed.columns
    ]

    if missing:
        raise ValueError(
            "Required dataset columns are missing: "
            + ", ".join(missing)
        )

    cleaned = renamed[REQUIRED_COLUMNS].copy()

    cleaned["machine_id"] = (
        cleaned["machine_id"].astype(str)
    )

    for column in FEATURES + ["failure"]:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned = cleaned.dropna(
        subset=REQUIRED_COLUMNS
    )

    cleaned["failure"] = cleaned["failure"].astype(int)

    cleaned = cleaned[
        cleaned["failure"].isin([0, 1])
    ]

    return cleaned


def load_original_csv():
    if not os.path.isfile(CSV_PATH):
        st.sidebar.error(
            "ai4i2020.csv was not found. "
            "Place it beside app.py in your GitHub repository."
        )
        return None

    try:
        # Keep original headers and columns for Dashboard.
        return pd.read_csv(CSV_PATH)

    except Exception as error:
        st.sidebar.error(
            f"Could not read ai4i2020.csv: {error}"
        )
        return None


# ==========================================================
# IMPORT CSV DATA INTO THE DATABASE
# ==========================================================

def sync_database_with_csv(original_data):
    cleaned = prepare_dataset(original_data)

    deleted_ids = get_deleted_ids()
    csv_ids = set(cleaned["machine_id"])

    current_data = get_machines()

    # Remove outdated records that are not present in the
    # current source dataset, and records already deleted.
    if not current_data.empty:
        for machine_id in current_data["machine_id"].astype(str):
            if (
                machine_id not in csv_ids
                or machine_id in deleted_ids
            ):
                delete_machine(machine_id)

    current_data = get_machines()

    current_ids = (
        set(current_data["machine_id"].astype(str))
        if not current_data.empty
        else set()
    )

    rows_to_import = cleaned[
        ~cleaned["machine_id"].isin(current_ids)
        & ~cleaned["machine_id"].isin(deleted_ids)
    ]

    if not rows_to_import.empty:
        import_csv_data(rows_to_import)


# Initialize database and original CSV.
initialize_database()
initialize_deleted_table()

original_csv = load_original_csv()

# Sync once per browser session. Ordinary reruns will not
# automatically restore machines deleted in that session.
if (
    original_csv is not None
    and "dataset_synced" not in st.session_state
):
    try:
        sync_database_with_csv(original_csv)
        st.session_state["dataset_synced"] = True

    except Exception as error:
        st.sidebar.error(
            f"Database import failed: {error}"
        )


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def get_analysis_data():
    """Return the original CSV as normalized analysis data."""
    if original_csv is None:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return prepare_dataset(original_csv)


def get_model():
    """Train the Random Forest model on operational records."""
    data = get_machines()

    if data.empty:
        raise ValueError(
            "No machine records found. Check ai4i2020.csv."
        )

    return train_model(data.copy())


def get_machine_by_id(machine_id):
    """Return one machine record from the operational database."""
    data = get_machines()

    if data.empty:
        return None

    match = data[
        data["machine_id"].astype(str) == str(machine_id)
    ]

    if match.empty:
        return None

    return match.iloc[0]


def parse_optional_number(text, original_value, field_name):
    """
    Empty input uses the selected machine's CSV value.
    A provided input is converted into a numeric value.
    """
    if not text or not text.strip():
        return float(original_value)

    try:
        value = float(text.strip())

    except ValueError as error:
        raise ValueError(
            f"{field_name} must be a valid number."
        ) from error

    if not pd.notna(value):
        raise ValueError(
            f"{field_name} must be a valid number."
        )

    return value


def show_prediction_result(model, readings):
    """Display risk, a threshold-based result, and advice."""
    result = predict_risk(model, readings)

    risk = float(result["risk"])

    metric_col, status_col = st.columns(2)

    metric_col.metric(
        "Estimated Failure Risk",
        f"{risk:.2f}%",
    )

    if risk >= 50:
        status_col.error(
            "Model result: Failure predicted"
        )
    else:
        status_col.success(
            "Model result: Failure not predicted"
        )

    st.write(
        f"**Risk category:** {result['level']}"
    )

    st.info(result["recommendation"])

    st.caption(
        "The failure/no-failure label uses a 50% estimated-risk "
        "threshold. This is a model estimate, not a guarantee "
        "of actual equipment behaviour."
    )

    return result


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("⚙️ Predictive Maintenance")

st.sidebar.caption(
    "AI-Based Industrial Equipment Monitoring"
)

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

st.sidebar.caption(
    "Educational demonstration project"
)


# ==========================================================
# PAGE 1: DASHBOARD
# Show all columns from original ai4i2020.csv.
# ==========================================================

if page == "Dashboard":

    st.title("Industrial Machine Dashboard")

    st.write(
        "Explore the original AI4I dataset and machine failure records."
    )

    if original_csv is None:
        st.warning(
            "The original dataset could not be loaded. "
            "Check ai4i2020.csv in your GitHub repository."
        )

    else:
        try:
            analysis_data = prepare_dataset(original_csv)

            total_records = len(original_csv)

            failed_records = int(
                (
                    analysis_data["failure"] == 1
                ).sum()
            )

            no_failure_records = (
                len(analysis_data) - failed_records
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total CSV Records",
                f"{total_records:,}",
            )

            c2.metric(
                "Recorded Failures",
                f"{failed_records:,}",
            )

            c3.metric(
                "No Recorded Failure",
                f"{no_failure_records:,}",
            )

        except Exception as error:
            st.warning(
                f"Could not calculate dataset metrics: {error}"
            )

        st.subheader("Original Dataset — ai4i2020.csv")

        search = st.text_input(
            "Search original dataset",
            placeholder="Search ID, Product ID, Type, or any value",
            key="dashboard_search",
        )

        # Filtering changes the view only, not the source CSV.
        if search.strip():
            mask = original_csv.astype(str).apply(
                lambda column: column.str.contains(
                    search.strip(),
                    case=False,
                    na=False,
                    regex=False,
                )
            ).any(axis=1)

            display_data = original_csv.loc[mask]

        else:
            display_data = original_csv

        st.caption(
            f"Showing {len(display_data):,} of "
            f"{len(original_csv):,} original records. "
            f"All {len(original_csv.columns)} original columns are retained."
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Original CSV",
            data=original_csv.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="ai4i2020.csv",
            mime="text/csv",
        )

        st.divider()

        st.subheader("Dataset Overview")

        try:
            analysis_data = prepare_dataset(original_csv)

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.pyplot(
                    machine_status_chart(analysis_data),
                    clear_figure=True,
                )

            with chart_col2:
                st.pyplot(
                    temperature_chart(analysis_data),
                    clear_figure=True,
                )

        except Exception as error:
            st.warning(
                f"Could not display dataset charts: {error}"
            )


# ==========================================================
# PAGE 2: MACHINE MANAGEMENT
# Machine ID deletion only. No default sensor-value form.
# ==========================================================

elif page == "Machine Management":

    st.title("Machine Management")

    st.write(
        "Delete a machine from the operational database "
        "using its Machine ID and record maintenance work."
    )

    data = get_machines()

    st.subheader("Delete Machine")

    st.caption(
        "Enter a Machine ID that exists in the dataset."
    )

    machine_id_to_delete = st.text_input(
        "Machine ID",
        placeholder="Enter Machine ID, for example 1",
        key="machine_delete_id",
    )

    if st.button("Delete Machine", key="delete_machine_button"):

        requested_id = machine_id_to_delete.strip()

        if not requested_id:
            st.warning("Enter a Machine ID first.")

        else:
            selected_machine = get_machine_by_id(requested_id)

            if selected_machine is None:
                st.error(
                    f"Machine ID {requested_id} was not found "
                    "in the operational database."
                )

            else:
                try:
                    delete_machine(requested_id)
                    remember_deleted_id(requested_id)

                    st.success(
                        f"Machine {requested_id} has been removed "
                        "from the operational database."
                    )

                    st.info(
                        "The original ai4i2020.csv file on GitHub "
                        "is not modified. Its original records "
                        "remain visible on the Dashboard."
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        f"Could not delete the machine: {error}"
                    )

    st.divider()

    st.subheader("Current Operational Machines")

    data = get_machines()

    if data.empty:
        st.info(
            "No operational machines are currently available."
        )

    else:
        st.caption(
            f"{len(data):,} machine records currently available "
            "for prediction and maintenance."
        )

        st.dataframe(
            data.head(100),
            use_container_width=True,
            hide_index=True,
        )

        if len(data) > 100:
            st.caption(
                "Showing the first 100 operational records here. "
                "The full original CSV remains available on Dashboard."
            )

    st.divider()

    st.subheader("Record Maintenance")

    maintenance_machine_id = st.text_input(
        "Maintenance Machine ID",
        placeholder="Enter an existing Machine ID",
        key="maintenance_machine_id",
    )

    maintenance_date = st.date_input(
        "Maintenance Date",
        key="maintenance_date",
    )

    maintenance_notes = st.text_area(
        "Maintenance Notes",
        placeholder="Describe maintenance, inspection, or service work...",
        key="maintenance_notes",
    )

    if st.button("Save Maintenance Record"):

        requested_id = maintenance_machine_id.strip()

        if not requested_id:
            st.warning(
                "Enter a Machine ID for the maintenance record."
            )

        elif get_machine_by_id(requested_id) is None:
            st.error(
                "That Machine ID is not in the operational database."
            )

        elif not maintenance_notes.strip():
            st.warning(
                "Enter maintenance notes before saving."
            )

        else:
            try:
                save_maintenance(
                    requested_id,
                    str(maintenance_date),
                    maintenance_notes.strip(),
                )

                st.success(
                    "Maintenance record saved successfully."
                )

                st.rerun()

            except Exception as error:
                st.error(
                    f"Could not save the maintenance record: {error}"
                )

    st.subheader("Maintenance History")

    history = get_maintenance()

    if history is None or history.empty:
        st.info(
            "No maintenance history is available yet."
        )

    else:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================
# PAGE 3: PREDICTION AND ANALYSIS
# No full CSV table on this page.
# ==========================================================

elif page == "Prediction & Analysis":

    st.title("Prediction & Analysis")

    st.write(
        "Predict failure risk for an existing machine or use "
        "new sensor readings, then review machine-data analysis."
    )

    data = get_machines()

    if data.empty:

        st.warning(
            "No machine records are available for prediction. "
            "Check ai4i2020.csv or Machine Management."
        )

    else:

        st.subheader("Machine Failure Prediction")

        machine_ids = data["machine_id"].astype(str).tolist()

        selected_id = st.selectbox(
            "Select Machine ID",
            machine_ids,
            key="prediction_machine_id",
        )

        selected_machine = get_machine_by_id(selected_id)

        st.caption(
            "Leave any new sensor field empty to use the selected "
            "machine's existing CSV reading."
        )

        with st.expander(
            "Selected Machine's Current Sensor Readings",
            expanded=True,
        ):

            current_cols = st.columns(4)

            for index, feature in enumerate(FEATURES):
                with current_cols[index]:
                    st.metric(
                        feature.replace("_", " ").title(),
                        f"{float(selected_machine[feature]):.2f}",
                    )

        with st.form("prediction_form"):

            st.write(
                "Optional: enter new readings to test different values."
            )

            p1, p2 = st.columns(2)

            with p1:
                temperature_input = st.text_input(
                    "Air Temperature [K]",
                    placeholder="Blank = use selected machine value",
                )

                speed_input = st.text_input(
                    "Rotational Speed [rpm]",
                    placeholder="Blank = use selected machine value",
                )

            with p2:
                torque_input = st.text_input(
                    "Torque [Nm]",
                    placeholder="Blank = use selected machine value",
                )

                wear_input = st.text_input(
                    "Tool Wear [min]",
                    placeholder="Blank = use selected machine value",
                )

            run_prediction = st.form_submit_button(
                "Predict Machine Failure Risk"
            )

        if run_prediction:

            try:
                readings = {
                    "temperature": parse_optional_number(
                        temperature_input,
                        selected_machine["temperature"],
                        "Air Temperature",
                    ),
                    "rotational_speed": parse_optional_number(
                        speed_input,
                        selected_machine["rotational_speed"],
                        "Rotational Speed",
                    ),
                    "torque": parse_optional_number(
                        torque_input,
                        selected_machine["torque"],
                        "Torque",
                    ),
                    "tool_wear": parse_optional_number(
                        wear_input,
                        selected_machine["tool_wear"],
                        "Tool Wear",
                    ),
                }

                model = get_model()

                st.subheader("Prediction Result")

                show_prediction_result(
                    model,
                    readings,
                )

                result_report = pd.DataFrame(
                    [
                        {
                            "machine_id": selected_id,
                            **readings,
                            "input_source": (
                                "CSV readings with optional overrides"
                            ),
                        }
                    ]
                )

                st.download_button(
                    "Download Prediction Input CSV",
                    data=result_report.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name=f"prediction_{selected_id}.csv",
                    mime="text/csv",
                )

            except Exception as error:
                st.error(
                    f"Prediction failed: {error}"
                )

        st.divider()

    # The following analysis uses the original CSV data,
    # and does not show the entire dataset table here.
    if original_csv is not None:

        try:
            analysis_data = prepare_dataset(original_csv)

            st.subheader("Machine Failure Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.pyplot(
                    machine_status_chart(analysis_data),
                    clear_figure=True,
                )

            with col2:
                st.pyplot(
                    temperature_chart(analysis_data),
                    clear_figure=True,
                )

            st.subheader("Random Forest Feature Importance")

            model = get_model()

            importance = get_feature_importance(model)

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.bar(
                importance["Feature"],
                importance["Importance"],
            )

            ax.set_xlabel("Machine Feature")
            ax.set_ylabel("Importance")
            ax.set_title("Feature Importance")

            ax.tick_params(
                axis="x",
                rotation=20,
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

            plt.close(fig)

            st.subheader("Statistical Summary")

            st.dataframe(
                statistical_summary(analysis_data),
                use_container_width=True,
            )

            st.subheader("Feature Correlation")

            st.dataframe(
                feature_correlation(analysis_data),
                use_container_width=True,
            )

            st.subheader("Temperature Hypothesis Test")

            test = temperature_hypothesis_test(
                analysis_data
            )

            if test is None:

                st.info(
                    "At least two records in both failure groups "
                    "are required for the hypothesis test."
                )

            else:

                test_col1, test_col2 = st.columns(2)

                test_col1.metric(
                    "Mean Temperature — No Failure",
                    f"{test['normal_mean']:.2f}",
                )

                test_col2.metric(
                    "Mean Temperature — Recorded Failure",
                    f"{test['failure_mean']:.2f}",
                )

                st.write(
                    f"**t-statistic:** "
                    f"{test['t_statistic']:.3f}"
                )

                st.write(
                    f"**p-value:** {test['p_value']:.4f}"
                )

                if test["p_value"] < 0.05:

                    st.success(
                        "The difference is statistically significant "
                        "at the 5% significance level."
                    )

                else:

                    st.info(
                        "There is insufficient evidence of a difference "
                        "at the 5% significance level."
                    )

            st.caption(
                "Statistical patterns do not prove causation. "
                "The model and results are for educational use."
            )

        except Exception as error:
            st.error(
                f"Analysis could not be completed: {error}"
            )


# ==========================================================
# PAGE 4: MAINTENANCE REPORT
# ==========================================================

elif page == "Maintenance Report":

    st.title("Maintenance Report & Recommendations")

    st.write(
        "Select a machine to view its readings, predicted failure "
        "risk, and recommended next steps."
    )

    data = get_machines()

    if data.empty:

        st.warning(
            "No machine records are available for a report."
        )

    else:

        selected_id = st.selectbox(
            "Select Machine ID",
            data["machine_id"].astype(str).tolist(),
            key="report_machine_id",
        )

        machine = get_machine_by_id(selected_id)

        st.subheader(
            f"Machine Report — {selected_id}"
        )

        report_readings = {
            "temperature": float(machine["temperature"]),
            "rotational_speed": float(machine["rotational_speed"]),
            "torque": float(machine["torque"]),
            "tool_wear": float(machine["tool_wear"]),
        }

        readings_table = pd.DataFrame(
            [
                {
                    "Sensor": "Temperature [K]",
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

            model = get_model()

            result = predict_risk(
                model,
                report_readings,
            )

            metric_col, status_col = st.columns(2)

            metric_col.metric(
                "Estimated Failure Risk",
                f"{result['risk']:.2f}%",
            )

            if result["risk"] >= 50:

                status_col.error(
                    "Failure predicted by model"
                )

            else:

                status_col.success(
                    "Failure not predicted by model"
                )

            st.write(
                f"**Risk category:** {result['level']}"
            )

            st.info(
                result["recommendation"]
            )

            report = pd.DataFrame(
                [
                    {
                        "machine_id": selected_id,
                        **report_readings,
                        "recorded_failure": int(
                            machine["failure"]
                        ),
                        "estimated_failure_risk_percent": result["risk"],
                        "risk_category": result["level"],
                        "recommendation": result["recommendation"],
                    }
                ]
            )

            st.download_button(
                "Download Maintenance Report CSV",
                data=report.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=f"maintenance_report_{selected_id}.csv",
                mime="text/csv",
            )

        except Exception as error:

            st.error(
                f"Could not create the maintenance report: {error}"
            )

        st.caption(
            "This is an educational demonstration using a sample dataset. "
            "Predictions are not validated for industrial use and do not "
            "replace professional inspections."
        )
