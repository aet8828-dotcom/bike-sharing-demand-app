import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUTPUT_DIR = Path("model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def find_dataset():
    possible_paths = [
        Path("hour.csv"),
        Path("data/hour.csv"),
        Path("bike_sharing_enhanced.csv"),
        Path("data/bike_sharing_enhanced.csv"),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the dataset. Put hour.csv in the same folder as this script, "
        "or inside a data/ folder."
    )


def add_readable_columns(df):
    month_map = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }

    weekday_map = {
        0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
        4: "Thursday", 5: "Friday", 6: "Saturday",
    }

    season_map = {
        1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter",
    }

    weather_map = {
        1: "Clear or Partly Cloudy",
        2: "Mist or Cloudy",
        3: "Light Snow or Light Rain",
        4: "Heavy Rain or Snow",
    }

    df = df.copy()
    df["Month"] = df["mnth"].map(month_map)
    df["Weekday_Name"] = df["weekday"].map(weekday_map)
    df["Season_Name"] = df["season"].map(season_map)
    df["Weather_Condition"] = df["weathersit"].map(weather_map)
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x in [0, 6] else 0)

    return df


def build_preprocessor(categorical_features, numerical_features):
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
        ]
    )


def evaluate_model(model_name, model, X_test, y_test):
    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }, predictions


def plot_actual_vs_predicted(y_test, predictions, file_name, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.35)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
    plt.title(title)
    plt.xlabel("Actual Rentals")
    plt.ylabel("Predicted Rentals")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / file_name, dpi=200)
    plt.close()


def plot_feature_importance(model, feature_names, file_name):
    rf_model = model.named_steps["model"]
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values("Importance", ascending=False)

    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    top_features = importance_df.head(15).sort_values("Importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.title("Top 15 Feature Importance - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / file_name, dpi=200)
    plt.close()


def get_feature_names(preprocessor):
    categorical_names = preprocessor.named_transformers_["categorical"].get_feature_names_out()
    numerical_names = preprocessor.transformers_[1][2]
    return list(categorical_names) + list(numerical_names)


def main():
    dataset_path = find_dataset()
    print(f"Loading dataset from: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = add_readable_columns(df)

    target = "cnt"

    # Do not use instant, dteday, casual, or registered as model predictors.
    categorical_features = [
        "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "is_weekend",
    ]

    numerical_features = ["temp", "atemp", "hum", "windspeed"]
    features = categorical_features + numerical_features

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    preprocessor = build_preprocessor(categorical_features, numerical_features)

    linear_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    # NOTE: n_estimators reduced from 100 -> 40 and max_depth capped at 15.
    # This keeps accuracy nearly identical while making the saved .pkl file
    # small enough to upload directly to GitHub (no Git LFS needed).
    random_forest_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_features, numerical_features)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=40,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining Linear Regression...")
    linear_model.fit(X_train, y_train)

    print("Training Random Forest (smaller model for GitHub upload)...")
    random_forest_model.fit(X_train, y_train)

    linear_metrics, linear_predictions = evaluate_model(
        "Linear Regression", linear_model, X_test, y_test,
    )

    rf_metrics, rf_predictions = evaluate_model(
        "Random Forest", random_forest_model, X_test, y_test,
    )

    results = pd.DataFrame([linear_metrics, rf_metrics])
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\nModel Comparison:")
    print(results)

    plot_actual_vs_predicted(
        y_test, linear_predictions,
        "linear_regression_actual_vs_predicted.png",
        "Linear Regression: Actual vs Predicted",
    )

    plot_actual_vs_predicted(
        y_test, rf_predictions,
        "random_forest_actual_vs_predicted.png",
        "Random Forest: Actual vs Predicted",
    )

    feature_names = get_feature_names(random_forest_model.named_steps["preprocessor"])
    plot_feature_importance(
        random_forest_model, feature_names,
        "random_forest_feature_importance.png",
    )

    joblib.dump(linear_model, OUTPUT_DIR / "linear_regression_model.pkl")
    joblib.dump(random_forest_model, OUTPUT_DIR / "random_forest_model.pkl")

    rf_size_mb = (OUTPUT_DIR / "random_forest_model.pkl").stat().st_size / (1024 * 1024)
    print(f"\nrandom_forest_model.pkl size: {rf_size_mb:.1f} MB")

    print("\nFinal Model Comparison:")
    print(results)

    print("\nSaved outputs inside:")
    print(os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
