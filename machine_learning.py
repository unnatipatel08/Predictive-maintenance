
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURES = [
    "temperature",
    "rotational_speed",
    "torque",
    "tool_wear"
]


def prepare_data(data):
    data = data.copy()

    for column in FEATURES + ["failure"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna(subset=["failure"])
    data = data[data["failure"].isin([0, 1])]
    data = data.dropna(subset=FEATURES, how="all")

    return data


def train_model(data):
    data = prepare_data(data)

    if len(data) < 4 or data["failure"].nunique() < 2:
        raise ValueError(
            "At least 4 usable rows with both failure classes are required."
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("random_forest", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(data[FEATURES], data["failure"])
    return model


def predict_risk(model, machine):
    values = pd.DataFrame(
        [{feature: machine[feature] for feature in FEATURES}]
    )

    probabilities = model.predict_proba(values)[0]
    classes = list(model.named_steps["random_forest"].classes_)

    if 1 in classes:
        risk = float(probabilities[classes.index(1)] * 100)
    else:
        risk = 0.0

    if risk >= 70:
        level = "High"
        recommendation = (
            "Arrange prompt inspection and check sensor readings."
        )
    elif risk >= 40:
        level = "Medium"
        recommendation = (
            "Schedule an inspection and monitor machine readings."
        )
    else:
        level = "Low"
        recommendation = (
            "Continue routine monitoring and scheduled maintenance."
        )

    return {
        "risk": round(risk, 2),
        "level": level,
        "recommendation": recommendation
    }


def get_feature_importance(model):
    forest = model.named_steps["random_forest"]

    return pd.DataFrame({
        "Feature": FEATURES,
        "Importance": forest.feature_importances_
    }).sort_values("Importance", ascending=False)
