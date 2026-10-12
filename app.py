import io
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from database import (
    initialize_database, get_machines, add_machine,
    update_machine, delete_machine, save_maintenance,
    get_maintenance, import_csv_data
)
from machine_learning import (
    train_model, predict_risk, get_feature_importance,
    FEATURES
)
from graph_analysis import (
    machine_status_chart, temperature_chart,
    feature_correlation, statistical_summary,
    temperature_hypothesis_test
)

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #F3F7FC;
    color: #1F2937;
}

[data-testid="stSidebar"] {
    background-color: #E2ECF8;
}

[data-testid="stSidebar"] * {
    color: #17365D !important;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp p,
.stApp label,
.stApp li {
    color: #1F2937;
}

[data-testid="stHeader"] {
    background-color: #F3F7FC;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #D8E3F0;
    padding: 18px;
    border-radius: 12px;
}

.stButton > button {
    background-color: #2563EB;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: #FFFFFF !important;
}

input, textarea {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
}
</style>
""", unsafe_allow_html=True)

initialize_database()

if "seeded" not in st.session_state:
    try:
        if get_machines().empty and os.path.exists("machine_data.csv"):
            import_csv_data(pd.read_csv("machine_data.csv"))
    except Exception:
        pass
    st.session_state["seeded"] = True

st.sidebar.title("⚙️ Predictive Maintenance")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Machine Management",
        "Prediction & Analysis",
        "Maintenance Report"
    ]
)

st.sidebar.caption("Educational demonstration project")
data = get_machines()

def show_model():
    source = data.copy()
    if source.empty:
        raise ValueError("Add or import machine data first.")
    return train_model(source)


def machine_form(machine_id=None):
    current = None

    if machine_id:
        matches = data[data["machine_id"] == machine_id]
        if not matches.empty:
            current = matches.iloc[0]

    with st.form("machine_form"):
        mid = st.text_input(
            "Machine ID",
            value=str(current["machine_id"]) if current is not None else ""
        )
        temperature = st.number_input(
            "Temperature",
            value=float(current["temperature"]) if current is not None else 70.0
        )
        speed = st.number_input(
            "Rotational Speed",
            min_value=0.0,
            value=float(current["rotational_speed"]) if current is not None else 1500.0
        )
        torque = st.number_input(
            "Torque",
            min_value=0.0,
            value=float(current["torque"]) if current is not None else 40.0
        )
        wear = st.number_input(
            "Tool Wear",
            min_value=0.0,
            value=float(current["tool_wear"]) if current is not None else 100.0
        )
        failure = st.selectbox(
            "Recorded Failure",
            [0, 1],
            index=int(current["failure"]) if current is not None else 0,
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
        )
        submitted = st.form_submit_button(
            "Update Machine" if current is not None else "Add Machine"
        )

    if submitted:
        mid = mid.strip()
        if not mid:
            st.error("Machine ID is required.")
            return

        machine = {
            "machine_id": mid,
            "temperature": temperature,
            "rotational_speed": speed,
            "torque": torque,
            "tool_wear": wear,
            "failure": failure
        }

        try:
            if current is not None:
                update_machine(machine)
                st.success("Machine updated.")
            else:
                add_machine(machine)
                st.success("Machine added.")
            st.rerun()
        except Exception as error:
            st.error(f"Could not save machine: {error}")


if page == "Dashboard":
    st.title("Industrial Machine Dashboard")
    st.write("Monitor machine readings and recorded failures.")

    data = get_machines()

    if data.empty:
        st.warning("No machine records found. Import a CSV in Machine Management.")
    else:
        total = len(data)
        failed = int((data["failure"] == 1).sum())
        healthy = total - failed

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Machines", total)
        c2.metric("Recorded Failures", failed)
        c3.metric("No Recorded Failure", healthy)

        search = st.text_input("Search Machine ID")
        view = data[
            data["machine_id"].str.contains(search, case=False, na=False)
        ]
        st.dataframe(view, use_container_width=True)

        st.download_button(
            "Download Machine Data CSV",
            data.to_csv(index=False).encode("utf-8"),
            "machine_data_export.csv",
            "text/csv"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(machine_status_chart(data))
        with col2:
            st.pyplot(temperature_chart(data))


elif page == "Machine Management":
    st.title("Machine Management")
    st.subheader("Import Machine CSV")

    uploaded = st.file_uploader(
        "Upload a CSV with machine columns",
        type=["csv"]
    )

    if uploaded is not None and st.button("Import CSV"):
        try:
            incoming = pd.read_csv(uploaded)
            required = [
                "machine_id", "temperature", "rotational_speed",
                "torque", "tool_wear", "failure"
            ]
            if not set(required).issubset(incoming.columns):
                st.error("CSV columns do not match the required format.")
            else:
                import_csv_data(incoming)
                st.success("CSV records imported.")
                st.rerun()
        except Exception as error:
            st.error(f"Import failed: {error}")

    st.subheader("Add Machine")
    machine_form()

    data = get_machines()
    if not data.empty:
        st.subheader("Update or Delete Machine")
        selected = st.selectbox(
            "Select machine",
            data["machine_id"].tolist(),
            key="manage_machine"
        )
        machine_form(selected)

        if st.button("Delete Selected Machine"):
            delete_machine(selected)
            st.success("Machine deleted.")
            st.rerun()

    st.subheader("Record Maintenance")
    ids = get_machines()["machine_id"].tolist()

    if ids:
        selected_id = st.selectbox("Machine", ids, key="maintenance_machine")
        date = st.date_input("Maintenance Date")
        notes = st.text_area("Maintenance Notes")

        if st.button("Save Maintenance Record"):
            save_maintenance(selected_id, str(date), notes)
            st.success("Maintenance record saved.")

    history = get_maintenance()
    st.subheader("Maintenance History")
    st.dataframe(history, use_container_width=True)


elif page == "Prediction & Analysis":
    st.title("Prediction & Analysis")
    data = get_machines()

    if data.empty:
        st.warning("Import or add machine records first.")
    else:
        st.subheader("Random Forest Failure-Risk Estimates")

        try:
            model = show_model()
            predictions = data.copy()
            predictions["estimated_failure_risk_percent"] = [
                predict_risk(model, row)["risk"]
                for _, row in predictions.iterrows()
            ]
            st.dataframe(predictions, use_container_width=True)

            st.download_button(
                "Download Prediction CSV",
                predictions.to_csv(index=False).encode("utf-8"),
                "machine_predictions.csv",
                "text/csv"
            )

            st.subheader("Feature Importance")
            importance = get_feature_importance(model)
            fig, ax = plt.subplots()
            ax.bar(importance["Feature"], importance["Importance"])
            ax.set_ylabel("Importance")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as error:
            st.error(f"Model could not be trained: {error}")

        st.subheader("Statistical Summary")
        st.dataframe(statistical_summary(data), use_container_width=True)

        st.subheader("Correlation Matrix")
        st.dataframe(feature_correlation(data), use_container_width=True)

        st.subheader("Temperature Hypothesis Test")
        test = temperature_hypothesis_test(data)

        if test is None:
            st.info("At least two records in each failure group are needed.")
        else:
            st.write(f"Mean temperature without recorded failure: {test['normal_mean']:.2f}")
            st.write(f"Mean temperature with recorded failure: {test['failure_mean']:.2f}")
            st.write(f"t-statistic: {test['t_statistic']:.3f}")
            st.write(f"p-value: {test['p_value']:.4f}")

            if test["p_value"] < 0.05:
                st.write("Result: The group means differ significantly at the 5% level.")
            else:
                st.write("Result: Insufficient evidence of a difference at the 5% level.")

        st.caption(
            "Statistical results show associations in this sample, not proof of causation."
        )


elif page == "Maintenance Report":
    st.title("Maintenance Report & Recommendations")
    data = get_machines()

    if data.empty:
        st.warning("Import or add machine records first.")
    else:
        selected = st.selectbox(
            "Select Machine",
            data["machine_id"].tolist(),
            key="report_machine"
        )
        machine = data[data["machine_id"] == selected].iloc[0]

        st.subheader(f"Machine {selected}")
        st.dataframe(machine.to_frame("Value"), use_container_width=True)

        try:
            model = show_model()
            result = predict_risk(model, machine)

            st.metric("Estimated Failure Risk", f"{result['risk']:.2f}%")
            st.write(f"Risk category: **{result['level']}**")
            st.info(result["recommendation"])

            report = pd.DataFrame([{
                "machine_id": selected,
                "temperature": machine["temperature"],
                "rotational_speed": machine["rotational_speed"],
                "torque": machine["torque"],
                "tool_wear": machine["tool_wear"],
                "recorded_failure": machine["failure"],
                "estimated_failure_risk_percent": result["risk"],
                "risk_category": result["level"],
                "recommendation": result["recommendation"]
            }])

            st.download_button(
                "Download Machine Report CSV",
                report.to_csv(index=False).encode("utf-8"),
                f"{selected}_maintenance_report.csv",
                "text/csv"
            )

        except Exception as error:
            st.error(f"Report prediction unavailable: {error}")

        st.caption(
            "This educational model uses a small sample dataset. "
            "Risk estimates and recommendations are not validated for industrial use."
        )
