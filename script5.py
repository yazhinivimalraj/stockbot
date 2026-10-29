import json, os, re
import joblib
import pandas as pd
DIR = "saved_model1"
if not os.path.exists(os.path.join(DIR, "commodity_nn.joblib")):
    raise SystemExit("No trained model found. Run `python train.py` first.")
model = joblib.load(os.path.join(DIR, "commodity_nn.joblib"))
meta = json.load(open(os.path.join(DIR, "meta.json")))
hist = pd.read_csv(os.path.join(DIR, "history_features.csv"), parse_dates=["date"])
FEATURES, SYMBOLS = meta["features"], meta["symbols"]
HELP = ("Ask me things like 'will gold go up on 2026-03-10?' or 'predict silver 2026-03-10'.\n"
        "Type 'list' for commodities, 'about' for model info, 'quit' to exit.")
def find_symbol(text):
    t = text.lower()
    for s in SYMBOLS:
        if s.lower() in t:
            return s
    for word in re.findall(r"[a-z]{3,}", t):
        hits = [s for s in SYMBOLS if word in s.lower()]
        if len(hits) == 1 and word not in {"will", "the", "price", "go", "up", "down", "for", "predict"}:
            return hits[0]
    return None
def predict(symbol, date):
    d = hist[hist["symbol"] == symbol].sort_values("date")
    prior = d[d["date"] < date].dropna(subset=FEATURES)
    if prior.empty:
        return f"I don't have enough history for {symbol} before {date.date()}."
    row = prior.iloc[-1]
    p_up = model.predict_proba(pd.DataFrame([row[FEATURES]]))[0][1]
    label = "INCREASE" if p_up >= 0.5 else "DECREASE"
    conf = max(p_up, 1 - p_up)
    msg = (f"{symbol} on {date.date()}: I predict {label} vs the previous close "
           f"({conf:.0%} confidence).\n(Based on data up to {row['date'].date()}, close {row['close']:.2f}.)")
    actual = d[d["date"] == date]
    if not actual.empty:
        real = "INCREASE" if actual.iloc[0]["close"] > row["close"] else "DECREASE"
        msg += f"\nActual outcome in the dataset: {real}."
    if date > d["date"].max() + pd.Timedelta(days=5):
        msg += "\nNote: that date is well past my data, so this is a weak guess."
    return msg + "\nThis is a statistical estimate, not financial advice."
def reply(text):
    t = text.strip().lower()
    if t in {"help", "?"}: return HELP
    if t == "list": return "Commodities: " + ", ".join(SYMBOLS)
    if t == "about":
        return (f"Neural network (64-32 MLP). Test accuracy {meta['test_accuracy']:.1%} "
                f"vs {meta['baseline']:.1%} majority-class baseline.")
    if t in {"hi", "hello", "hey"}: return "Hello! " + HELP
    sym = find_symbol(t)
    m = re.search(r"\d{4}-\d{2}-\d{2}", t)
    if not sym: return "Which commodity? Type 'list' to see what's available."
    if not m: return f"Which date for {sym}? Use the format YYYY-MM-DD."
    return predict(sym, pd.Timestamp(m.group()))
if __name__ == "__main__":
    print("Commodity Chatbot — " + HELP)
    while True:
        try:
            q = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in {"quit", "exit", "bye"}:
            print("Bot: Goodbye!"); break
        print("Bot:", reply(q))