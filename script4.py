import argparse, json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
OUT_DIR = "saved_model1"
FEATURES = ["ret_1", "ret_5", "ret_10", "range_pct", "body_pct",
            "close_to_sma5", "close_to_sma20", "vol_chg", "volatility_10"]
def load_raw(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        from datasets import load_dataset
        ds = load_dataset("vishnun0027/indian-market-historical-ohlcv", "commodities")
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # tolerate naming differences
    rename = {"datetime": "date", "timestamp": "date", "ticker": "symbol",
              "adj_close": "adjusted_close"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    need = {"date", "open", "high", "low", "close", "volume", "symbol"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset is missing columns {missing}. Found: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date", "close"])
    return df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"]).reset_index(drop=True)
def add_features(df):
    def per_symbol(d):
        d = d.copy()
        c = d["close"]
        d["ret_1"] = c.pct_change(1)
        d["ret_5"] = c.pct_change(5)
        d["ret_10"] = c.pct_change(10)
        d["range_pct"] = (d["high"] - d["low"]) / c
        d["body_pct"] = (d["close"] - d["open"]) / d["open"]
        d["close_to_sma5"] = c / c.rolling(5).mean() - 1
        d["close_to_sma20"] = c / c.rolling(20).mean() - 1
        d["vol_chg"] = d["volume"].replace(0, np.nan).pct_change().clip(-5, 5)
        d["volatility_10"] = d["ret_1"].rolling(10).std()
        d["next_close"] = c.shift(-1)
        d["target"] = (d["next_close"] > c).astype(int)
        return d
    out = pd.concat([per_symbol(d) for _, d in df.groupby("symbol")], ignore_index=True)
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    out["vol_chg"] = out["vol_chg"].fillna(0)
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="local CSV instead of downloading from Hugging Face")
    args = ap.parse_args()
    raw = load_raw(args.csv)
    full = add_features(raw)
    train_df = full.dropna(subset=FEATURES + ["next_close"]).sort_values("date")
    print(f"{len(raw):,} rows, {raw['symbol'].nunique()} symbols, "
          f"{raw['date'].min().date()} -> {raw['date'].max().date()}")
    cutoff = train_df["date"].quantile(0.8)
    tr, te = train_df[train_df["date"] <= cutoff], train_df[train_df["date"] > cutoff]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("nn", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                             alpha=1e-3, early_stopping=True, validation_fraction=0.1,
                             max_iter=300, random_state=42)),
    ])
    model.fit(tr[FEATURES], tr["target"])
    pred = model.predict(te[FEATURES])
    acc = accuracy_score(te["target"], pred)
    baseline = max(te["target"].mean(), 1 - te["target"].mean())
    print(f"\nTest accuracy: {acc:.3f}  (always-guess-majority baseline: {baseline:.3f})")
    print(classification_report(te["target"], pred, target_names=["Decrease", "Increase"]))
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUT_DIR, "commodity_nn.joblib"))
    full[["date", "symbol", "close"] + FEATURES].to_csv(
        os.path.join(OUT_DIR, "history_features.csv"), index=False)
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump({"features": FEATURES, "test_accuracy": acc, "baseline": baseline,
                   "symbols": sorted(raw["symbol"].astype(str).unique().tolist())}, f, indent=2)
    print(f"\nModel + metadata saved in ./{OUT_DIR}/")
if __name__ == "__main__":
    main()