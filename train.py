import os
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib


def load_data(paths):
    dfs = []
    for p in paths:
        if os.path.exists(p):
            try:
                dfs.append(pd.read_csv(p))
            except Exception:
                pass
    if not dfs:
        raise FileNotFoundError("No dataset files found")
    df = pd.concat(dfs, ignore_index=True)
    return df


def prepare(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for required in ("City", "BHK", "Size", "Rent"):
        if required not in df.columns:
            raise KeyError(f"Required column '{required}' not found in data")
    df = df[["City", "BHK", "Size", "Rent"]]
    df = df.dropna()
    df["BHK"] = pd.to_numeric(df["BHK"], errors="coerce")
    df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
    df["Rent"] = pd.to_numeric(df["Rent"], errors="coerce")
    df = df.dropna()
    df = df[(df["Size"] > 20) & (df["BHK"] > 0) & (df["Rent"] > 0) & (df["Rent"] < 1_000_000)]
    df["City"] = df["City"].astype(str).str.title().str.strip()
    return df


def train_and_save(df, out_dir="model"):
    os.makedirs(out_dir, exist_ok=True)
    X = df[["BHK", "Size", "City"]]
    y = df["Rent"]

    numeric_features = ["BHK", "Size"]
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_features = ["City"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

    pipeline.fit(X, y)

    joblib.dump(pipeline, os.path.join(out_dir, "pipeline.pkl"))

    try:
        cities = list(pipeline.named_steps["preprocessor"].named_transformers_["cat"].categories_[0])
        cities = [c.title() for c in cities]
    except Exception:
        cities = sorted(df["City"].unique().tolist())
    with open(os.path.join(out_dir, "cities.json"), "w") as f:
        json.dump(cities, f)

    print("Model saved to:", os.path.join(out_dir, "pipeline.pkl"))


def main():
    base = os.path.dirname(__file__)
    paths = [
        os.path.join(base, "rent_data.csv"),
        os.path.join(base, "rent2.csv"),
    ]
    df = load_data(paths)
    df = prepare(df)
    train_and_save(df)


if __name__ == "__main__":
    main()
