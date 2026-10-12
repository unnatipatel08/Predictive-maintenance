
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from database import (
    initialize_database,
    get_machines,
    add_machine,
    update_machine,
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
    page_title="Predictive Maintenance AI",
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

    input,
    textarea {
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
# DATABASE SETUP AND AUTOMATIC CSV LOADING
# ==========================================================

initialize_database()


def load_dataset_from_repository():
    """Automatically load ai4i2020.csv from the repository."""

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ai4i2020.csv",
    )

    if not os.path.exists(csv_path):
        st.sidebar.error(
            "ai4i2020.csv was not found. "
            "Place it in the same repository folder as app.py."
        )
        return

    # Avoid importing the same initial dataset every rerun.
    if not get_machines().empty:
        return

    try:
        csv_data = pd.read_csv(csv_path)

        # Convert original AI4I dataset column names
        # into the names used by this application.
        csv_data = csv_data.rename(
            columns={
                "UDI": "machine_id",
                "Machine failure": "failure",
                "Air temperature [K]": "temperature",
                "Rotational speed [rpm]": "rotational_speed",
                "Torque [Nm]": "torque",
                "Tool wear [min]": "tool_wear",
            }
        )

        required_columns = [
            "machine_id",
            "temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
            "failure",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in csv_data.columns
        ]

        if missing_columns:
            st.sidebar.error(
                "Missing CSV columns: "
                + ", ".join(missing_columns)
            )
            return

        csv_data = csv_data[required_columns].copy()

        csv_data["machine_id"] = (
            csv_data["machine_id"].astype(str)
        )

        for column in [
            "temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
            "failure",
        ]:
            csv_data[column] = pd.to_numeric(
                csv_data[column],
                errors="coerce",
            )

        csv_data = csv_data.dropna(
            subset=required_columns
        )

        import_csv_data(csv_data)

        st.sidebar.success(
            f"Loaded {len(csv_data):,} records from ai4i2020.csv."
        )

    except Exception as error:
        st.sidebar.error(
            f"Automatic CSV loading failed: {error}"
        )


load_dataset_from_repository()


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
# MACHINE LEARNING MODEL
# ==========================================================

def show_model():
    """Train a model using the machine records."""

    current_data = get_machines()

    if current_data.empty:
        raise ValueError(
            "No machine records found. Check ai4i2020.csv."
        )

    return train_model(current_data.copy())


# ==========================================================
# ADD / UPDATE MACHINE FORM
# ==========================================================

def machine_form(machine_id=None):
    """Create a form to add or update a machine."""

    current_data = get_machines()
    current = None

    if machine_id is not None and not current_data.empty:

        matches = current_data[
            current_data["machine_id"].astype(str)
            == str(machine_id)
        ]

        if not matches.empty:
            current = matches.iloc[0]

    with st.form(
        f"machine_form_{machine_id or 'new'}"
    ):

        machine_id_value = st.text_input(
            "Machine ID",
            value=(
                str(current["machine_id"])
                if current is not None
                else ""
            ),
        )

        temperature = st.number_input(
            "Temperature",
            value=(
                float(current["temperature"])
                if current is not None
                else 70.0
            ),
        )

        speed = st.number_input(
            "Rotational Speed",
            min_value=0.0,
            value=(
                float(current["rotational_speed"])
                if current is not None
                else 1500.0
            ),
        )

        torque = st.number_input(
            "Torque",
            min_value=0.0,
            value=(
                float(current["torque"])
                if current is not None
                else 40.0
            ),
        )

        wear = st.number_input(
            "Tool Wear",
            min_value=0.0,
            value=(
                float(current["tool_wear"])
                if current is not None
                else 100.0
            ),
        )

        failure = st.selectbox(
            "Recorded Failure",
            [0, 1],
            index=(
                int(current["failure"])
                if current is not None
                else 0
            ),
            format_func=lambda value: (
                "Yes (1)" if value == 1 else "No (0)"
            ),
        )

        submitted = st.form_submit_button(
            "Update Machine"
            if current is not None
            else "Add Machine"
        )

    if submitted:

        cleaned_id = machine_id_value.strip()

        if not cleaned_id:
            st.error("Machine ID is required.")
            return

        machine = {
            "machine_id": cleaned_id,
            "temperature": temperature,
            "rotational_speed": speed,
            "torque": torque,
            "tool_wear": wear,
            "failure": failure,
        }

        try:

            if current is not None:
                update_machine(machine)

                st.success(
                    "Machine updated successfully."
                )

            else:
                add_machine(machine)

                st.success(
                    "Machine added successfully."
                )

            st.rerun()

        except Exception as error:
            st.error(
                f"Could not save machine: {error}"
            )


# ==========================================================
# PAGE 1: DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.title("Industrial Machine Dashboard")

    st.write(
        "Monitor machine readings, review recorded failures, "
        "and export machine data."
    )

    data = get_machines()

    if data.empty:

        st.warning(
            "No machine records found. "
            "Check that ai4i2020.csv is in the same "
            "repository folder as app.py."
        )

    else:

        total = len(data)

        failed = int(
            (
                pd.to_numeric(
                    data["failure"],
                    errors="coerce",
                )
                == 1
            ).sum()
        )

        healthy = total - failed

        card1, card2, card3 = st.columns(3)

        card1.metric(
            "Total Machines",
            f"{total:,}",
        )

        card2.metric(
            "Recorded Failures",
            f"{failed:,}",
        )

        card3.metric(
            "No Recorded Failure",
            f"{healthy:,}",
        )

        st.subheader("Machine Records")

        search = st.text_input(
            "Search by Machine ID",
            placeholder="Enter machine ID",
        )

        searchable_ids = data[
            "machine_id"
        ].astype(str)

        view = data[
            searchable_ids.str.contains(
                search,
                case=False,
                na=False,
            )
        ]

        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Machine Data CSV",
            data.to_csv(index=False).encode("utf-8"),
            file_name="machine_data_export.csv",
            mime="text/csv",
        )

        st.subheader("Machine Overview")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.pyplot(
                machine_status_chart(data),
                clear_figure=True,
            )

        with chart_col2:
            st.pyplot(
                temperature_chart(data),
                clear_figure=True,
            )


# ==========================================================
# PAGE 2: MACHINE MANAGEMENT
# ==========================================================

elif page == "Machine Management":

    st.title("Machine Management")

    st.write(
        "Add, update, or delete machine records "
        "and maintain a maintenance history."
    )

    st.subheader("Add New Machine")

    machine_form()

    data = get_machines()

    if not data.empty:

        st.subheader("Update or Delete Machine")

        selected_machine = st.selectbox(
            "Select a machine",
            data["machine_id"].astype(str).tolist(),
            key="manage_machine",
        )

        machine_form(selected_machine)

        if st.button("Delete Selected Machine"):

            try:
                delete_machine(selected_machine)

                st.success(
                    "Machine deleted successfully."
                )

                st.rerun()

            except Exception as error:
                st.error(
                    f"Could not delete machine: {error}"
                )

    st.divider()

    st.subheader("Record Maintenance")

    current_data = get_machines()

    ids = (
        current_data["machine_id"].astype(str).tolist()
        if not current_data.empty
        else []
    )

    if ids:

        selected_id = st.selectbox(
            "Machine",
            ids,
            key="maintenance_machine",
        )

        maintenance_date = st.date_input(
            "Maintenance Date"
        )

        notes = st.text_area(
            "Maintenance Notes",
            placeholder="Describe the maintenance performed...",
        )

        if st.button("Save Maintenance Record"):

            try:
                save_maintenance(
                    selected_id,
                    str(maintenance_date),
                    notes,
                )

                st.success(
                    "Maintenance record saved."
                )

                st.rerun()

            except Exception as error:
                st.error(
                    f"Could not save maintenance record: {error}"
                )

    else:

        st.info(
            "Add a machine before recording maintenance."
        )

    st.subheader("Maintenance History")

    history = get_maintenance()

    if history is None or history.empty:

        st.info(
            "No maintenance records have been saved yet."
        )

    else:

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================
# PAGE 3: PREDICTION AND ANALYSIS
# ==========================================================

elif page == "Prediction & Analysis":

    st.title("Prediction & Analysis")

    st.write(
        "Estimate machine failure risk and analyze data patterns."
    )

    data = get_machines()

    if data.empty:

        st.warning(
            "No machine records found. "
            "Check that ai4i2020.csv is in the repository."
        )

    else:

        st.subheader("Random Forest Failure-Risk Estimates")

        try:

            model = show_model()

            predictions = data.copy()

            predictions[
                "estimated_failure_risk_percent"
            ] = [
                predict_risk(model, row)["risk"]
                for _, row in predictions.iterrows()
            ]

            st.dataframe(
                predictions,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Prediction CSV",
                predictions.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="machine_predictions.csv",
                mime="text/csv",
            )

            st.subheader("Feature Importance")

            importance = get_feature_importance(model)

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.bar(
                importance["Feature"],
                importance["Importance"],
            )

            ax.set_xlabel("Machine Feature")
            ax.set_ylabel("Importance")
            ax.tick_params(
                axis="x",
                rotation=25,
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

            plt.close(fig)

        except Exception as error:

            st.error(
                f"Model could not be trained: {error}"
            )

        st.divider()

        st.subheader("Statistical Summary")

        try:

            st.dataframe(
                statistical_summary(data),
                use_container_width=True,
            )

        except Exception as error:

            st.warning(
                f"Could not calculate statistics: {error}"
            )

        st.subheader("Correlation Matrix")

        try:

            st.dataframe(
                feature_correlation(data),
                use_container_width=True,
            )

        except Exception as error:

            st.warning(
                f"Could not calculate correlations: {error}"
            )

        st.subheader("Temperature Hypothesis Test")

        try:

            test = temperature_hypothesis_test(data)

            if test is None:

                st.info(
                    "At least two records in each failure group "
                    "are needed for the test."
                )

            else:

                result_col1, result_col2 = st.columns(2)

                result_col1.metric(
                    "Mean Temperature — No Failure",
                    f"{test['normal_mean']:.2f}",
                )

                result_col2.metric(
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
                        "The group means differ significantly "
                        "at the 5% significance level."
                    )

                else:

                    st.info(
                        "There is insufficient evidence of a "
                        "difference at the 5% significance level."
                    )

        except Exception as error:

            st.warning(
                f"Could not run the hypothesis test: {error}"
            )

        st.caption(
            "Statistical results describe patterns in the dataset; "
            "they do not prove causation."
        )


# ==========================================================
# PAGE 4: MAINTENANCE REPORT AND RECOMMENDATIONS
# ==========================================================

elif page == "Maintenance Report":

    st.title("Maintenance Report & Recommendations")

    st.write(
        "Review a machine's readings, estimated risk, "
        "and suggested action."
    )

    data = get_machines()

    if data.empty:

        st.warning(
            "No machine records found. "
            "Check that ai4i2020.csv is in the repository."
        )

    else:

        selected = st.selectbox(
            "Select Machine",
            data["machine_id"].astype(str).tolist(),
            key="report_machine",
        )

        machine = data[
            data["machine_id"].astype(str)
            == str(selected)
        ].iloc[0]

        st.subheader(f"Machine: {selected}")

        st.dataframe(
            machine.to_frame("Value"),
            use_container_width=True,
        )

        try:

            model = show_model()

            result = predict_risk(
                model,
                machine,
            )

            st.metric(
                "Estimated Failure Risk",
                f"{result['risk']:.2f}%",
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
                        "machine_id": selected,
                        "temperature": machine["temperature"],
                        "rotational_speed": machine["rotational_speed"],
                        "torque": machine["torque"],
                        "tool_wear": machine["tool_wear"],
                        "recorded_failure": machine["failure"],
                        "estimated_failure_risk_percent": result["risk"],
                        "risk_category": result["level"],
                        "recommendation": result["recommendation"],
                    }
                ]
            )

            st.download_button(
                "Download Machine Report CSV",
                report.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=f"{selected}_maintenance_report.csv",
                mime="text/csv",
            )

        except Exception as error:

            st.error(
                f"Report prediction unavailable: {error}"
            )

        st.caption(
            "This is an educational model using a sample dataset. "
            "Predictions are not validated for industrial use and "
            "should not replace professional equipment inspections."
        )
