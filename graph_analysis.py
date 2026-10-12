
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def machine_status_chart(data):
    counts = data["failure"].map({
        0: "No Recorded Failure",
        1: "Recorded Failure"
    }).value_counts()

    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Machine Failure Records")
    ax.set_xlabel("Recorded Status")
    ax.set_ylabel("Number of Machines")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()

    return fig


def temperature_chart(data):
    fig, ax = plt.subplots()

    for status, label in [(0, "No Recorded Failure"),
                          (1, "Recorded Failure")]:
        values = data.loc[
            data["failure"] == status, "temperature"
        ].dropna()

        if not values.empty:
            ax.hist(values, alpha=0.6, label=label)

    ax.set_title("Temperature Distribution")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Number of Records")
    ax.legend()
    fig.tight_layout()

    return fig


def feature_correlation(data):
    columns = [
        "temperature", "rotational_speed",
        "torque", "tool_wear", "failure"
    ]

    return data[columns].corr(numeric_only=True)


def statistical_summary(data):
    return data[
        ["temperature", "rotational_speed", "torque", "tool_wear"]
    ].describe().T


def temperature_hypothesis_test(data):
    normal = data.loc[
        data["failure"] == 0, "temperature"
    ].dropna()

    failed = data.loc[
        data["failure"] == 1, "temperature"
    ].dropna()

    if len(normal) < 2 or len(failed) < 2:
        return None

    result = stats.ttest_ind(
        normal, failed, equal_var=False
    )

    return {
        "normal_mean": float(normal.mean()),
        "failure_mean": float(failed.mean()),
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue)
    }
