
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent

NASA_DATA_PATH = BASE_DIR / "nasa_combustion_data.csv"
SYNTHETIC_DATA_PATH = BASE_DIR / "nasa_parametric_grid.csv"
MODEL_PATH = BASE_DIR / "eigenmind_fire_model.pkl"


def calculate_physics_hazard_score(
    o2_percent,
    pressure_kpa,
    airflow_cm_s,
    is_concurrent,
    gravity_g
):
    """
    Project-defined illustrative hazard index.
    This is not a NASA-standard or experimentally validated fire-risk equation.
    """

    values = [
        o2_percent,
        pressure_kpa,
        airflow_cm_s,
        gravity_g
    ]

    if not all(np.isfinite(float(v)) for v in values):
        raise ValueError("All numerical scenario inputs must be finite.")

    o2_percent = float(o2_percent)
    pressure_kpa = float(pressure_kpa)
    airflow_cm_s = float(airflow_cm_s)
    gravity_g = float(gravity_g)

    if pressure_kpa <= 0 or airflow_cm_s < 0 or gravity_g < 0:
        raise ValueError("Pressure must be positive; airflow and gravity cannot be negative.")

    o2_factor = (
        (max(o2_percent, 12.0) - 12.0) / 73.0
    ) ** 0.85 * 55.0

    flow_factor = (
        airflow_cm_s / (airflow_cm_s + 8.0)
    ) * 22.0

    ppo2 = (o2_percent / 100.0) * pressure_kpa

    pressure_factor = np.clip(
        (ppo2 / 40.0) * 15.0,
        2.0,
        22.0
    )

    direction_mult = 1.35 if int(is_concurrent) == 1 else 0.85

    grav_mult = (
        1.15 if gravity_g <= 0.2
        else 1.05 if gravity_g <= 0.4
        else 0.90
    )

    score = (
        (o2_factor + flow_factor + pressure_factor)
        * direction_mult
        * grav_mult
    )

    return float(np.clip(score, 5.0, 99.5))


def get_partial_gravity_uncertainty(gravity_g):
    """
    Returns an illustrative project margin.
    These margins are not statistically calibrated confidence intervals.
    """

    if gravity_g == 0.0:
        return "Illustrative model margin (not experimentally calibrated)", 4.2

    elif gravity_g < 0.5:
        return "Illustrative partial-gravity scenario margin", 14.5

    return "Illustrative terrestrial scenario margin", 3.5


def build_real_nasa_psi_dataset():
    """
    Loads the curated NASA PSI source dataset.
    Does not invent missing measurements or overwrite the source CSV.
    """

    if not NASA_DATA_PATH.exists():
        raise FileNotFoundError(
            f"NASA source dataset not found: {NASA_DATA_PATH}"
        )

    base_df = pd.read_csv(NASA_DATA_PATH)

    required_columns = [
        "PSI_ID",
        "Investigation",
        "DOI",
        "Hardware",
        "Material",
        "Fuel_State",
        "Oxygen_Percent",
        "Pressure_kPa",
        "Airflow_cm_s",
        "Flow_Direction",
        "Burn_Time_s",
        "Gravity_g",
        "Key_Finding",
        "Earth_Application"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in base_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"NASA dataset is missing required columns: {missing_columns}"
        )

    return base_df


def build_synthetic_training_dataset(n_samples=3000):
    """
    Generates hypothetical EigenMind parameter scenarios.

    These are project-generated scenarios, not NASA experimental records.
    """

    rng = np.random.default_rng(42)

    o2_values = rng.uniform(14.0, 85.0, n_samples)
    pressure_values = rng.uniform(29.6, 120.0, n_samples)
    airflow_values = rng.uniform(0.5, 40.0, n_samples)

    direction_values = rng.choice(
        [0, 1],
        size=n_samples
    )

    gravity_values = rng.choice(
        [0.0, 0.165, 0.38, 1.0],
        size=n_samples,
        p=[0.50, 0.20, 0.20, 0.10]
    )

    scores = [
        calculate_physics_hazard_score(
            o2,
            pressure,
            airflow,
            direction,
            gravity
        )
        for o2, pressure, airflow, direction, gravity
        in zip(
            o2_values,
            pressure_values,
            airflow_values,
            direction_values,
            gravity_values
        )
    ]

    training_df = pd.DataFrame({
        "PSI_ID": ["SYNTHETIC"] * n_samples,
        "Investigation": [
            "EigenMind Synthetic Scenario Set"
        ] * n_samples,
        "Material": [
            "Hypothetical parameter combination"
        ] * n_samples,
        "Oxygen_Percent": np.round(o2_values, 2),
        "Pressure_kPa": np.round(pressure_values, 2),
        "Airflow_cm_s": np.round(airflow_values, 2),
        "Is_Concurrent": direction_values,
        "Gravity_g": gravity_values,
        "Fire_Hazard_Score": np.round(scores, 2)
    })

    return training_df


def train_ai_model():
    """
    Trains the Random Forest on synthetic mathematical targets.
    The NASA source CSV is read-only in this training workflow.
    """

    base_df = build_real_nasa_psi_dataset()

    training_df = build_synthetic_training_dataset(
        n_samples=3000
    )

    features = [
        "Oxygen_Percent",
        "Pressure_kPa",
        "Airflow_cm_s",
        "Is_Concurrent",
        "Gravity_g"
    ]

    X = training_df[features]
    y = training_df["Fire_Hazard_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    # This file contains synthetic scenarios, not NASA measurements.
    training_df.to_csv(
        SYNTHETIC_DATA_PATH,
        index=False
    )

    # Deliberately do NOT write to nasa_combustion_data.csv.

    print("SUCCESS: Synthetic training dataset generated.")
    print("SUCCESS: Random Forest model trained.")
    print("NASA source CSV preserved.")
    print(f"NASA records: {len(base_df)}")
    print(f"Synthetic scenarios: {len(training_df)}")

    return base_df, training_df, model


if __name__ == "__main__":
    train_ai_model()
