            st.subheader("Feature Importance")
            importance = get_feature_importance(model)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(importance["Feature"], importance["Importance"])
            ax.set_ylabel("Importance")
            ax.set_xlabel("Machine Feature")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as error:
            st.error(f"Model could not be trained: {error}")

        st.divider()
        st.subheader("Statistical Summary")
        try:
            st.dataframe(statistical_summary(data), use_container_width=True)
        except Exception as error:
            st.warning(f"Could not calculate summary statistics: {error}")

        st.subheader("Correlation Matrix")
        try:
            st.dataframe(feature_correlation(data), use_container_width=True)
        except Exception as error:
            st.warning(f"Could not calculate correlation matrix: {error}")

        st.subheader("Temperature Hypothesis Test")
        try:
            test = temperature_hypothesis_test(data)
            if test is None:
                st.info("At least two records in each failure group are needed.")
            else:
                result_col1, result_col2 = st.columns(2)
                result_col1.metric(
                    "Mean Temperature — No Failure", f"{test['normal_mean']:.2f}"
                )
                result_col2.metric(
                    "Mean Temperature — Recorded Failure", f"{test['failure_mean']:.2f}"
                )
                st.write(f"**t-statistic:** {test['t_statistic']:.3f}")
                st.write(f"**p-value:** {test['p_value']:.4f}")
                if test["p_value"] < 0.05:
                    st.success(
                        "The group means differ significantly at the 5% significance level."
                    )
                else:
                    st.info(
                        "There is insufficient evidence of a difference at the 5% significance level."
                    )
        except Exception as error:
            st.warning(f"Could not run the hypothesis test: {error}")

        st.caption(
            "Statistical results describe patterns in this dataset; they do not prove causation."
        )


# ------------------------------------------------------------
# 4. Maintenance Report and Recommendations
# ------------------------------------------------------------
elif page == "Maintenance Report":
    st.title("Maintenance Report & Recommendations")
    st.write("Review the selected machine and download a report for it.")

    data = get_machines()
    if data.empty:
        st.warning("No machine records found. Check that ai4i2020.csv is in the same GitHub folder as app.py.")
    else:
        selected = st.selectbox(
            "Select Machine",
            data["machine_id"].astype(str).tolist(),
            key="report_machine",
        )
        machine = data[data["machine_id"].astype(str) == str(selected)].iloc[0]

        st.subheader(f"Machine: {selected}")
        st.dataframe(machine.to_frame("Value"), use_container_width=True)

        try:
            model = show_model()
            result = predict_risk(model, machine)
            st.metric("Estimated Failure Risk", f"{result['risk']:.2f}%")
            st.write(f"**Risk category:** {result['level']}")
            st.info(result["recommendation"])

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
                report.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected}_maintenance_report.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.error(f"Report prediction unavailable: {error}")

        st.caption(
            "This educational model uses a sample dataset. Predictions and recommendations "
            "are not validated for industrial use and should not replace professional inspection."
        )
