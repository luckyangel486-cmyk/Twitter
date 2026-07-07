"""
Backend analyse-script — wordt uitgevoerd door de GitHub Action bij elke
push naar data/uploads/. Leest alle nieuwe uploads, merged ze met de
bestaande dataset (dupes eruit), traint het AI-model, en schrijft:
  - data/dataset.xlsx   (de opgebouwde dataset, dupes verwijderd)
  - data/results.json   (wat de frontend-pagina laat zien)
Verwijdert daarna de verwerkte bestanden uit data/uploads/.
"""

import io
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

UPLOAD_DIR = "data/uploads"
DATASET_PATH = "data/dataset.xlsx"
RESULTS_PATH = "data/results.json"


# ==============================
# PARSERS
# ==============================

def parse_tweets_js(raw_bytes):
    text = raw_bytes.decode("utf-8", errors="ignore")
    match = re.search(r"=\s*(\[.*\])\s*$", text, re.DOTALL)
    json_str = match.group(1) if match else text
    items = json.loads(json_str)

    rows = []
    for item in items:
        t = item.get("tweet", item)
        media = t.get("entities", {}).get("media") or []
        rows.append({
            "id": str(t.get("id_str", t.get("id", ""))),
            "tijd": t.get("created_at"),
            "text": t.get("full_text", t.get("text", "")),
            "likes": int(t.get("favorite_count", 0) or 0),
            "retweets": int(t.get("retweet_count", 0) or 0),
            "replies": 0,
            "quotes": 0,
            "heeft_media": bool(media),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["tijd"] = pd.to_datetime(df["tijd"], errors="coerce", utc=True).dt.tz_convert(None)
    return df


def parse_csv_export(raw_bytes):
    df_raw = pd.read_csv(io.BytesIO(raw_bytes))
    cols = {c.lower().strip(): c for c in df_raw.columns}

    def pick(*names, default=0):
        for n in names:
            if n in cols:
                return df_raw[cols[n]]
        return default

    id_col = pick("id", "tweet id", "id_str", default=None)
    df = pd.DataFrame({
        "id": id_col.astype(str) if id_col is not None else [str(i) for i in range(len(df_raw))],
        "tijd": pick("created_at", "date", "time", "creation date"),
        "text": pick("text", "full_text", "content", "tweet content", default=""),
        "likes": pd.to_numeric(pick("likes", "like_count", "favorite_count", "like count", default=0), errors="coerce").fillna(0),
        "retweets": pd.to_numeric(pick("retweets", "retweet_count", "retweet count", default=0), errors="coerce").fillna(0),
        "replies": pd.to_numeric(pick("replies", "reply_count", "reply count", default=0), errors="coerce").fillna(0),
        "quotes": pd.to_numeric(pick("quotes", "quote_count", "quote count", default=0), errors="coerce").fillna(0),
    })
    df["heeft_media"] = False
    df["tijd"] = pd.to_datetime(df["tijd"], errors="coerce", utc=True).dt.tz_convert(None)
    return df


# ==============================
# FEATURE ENGINEERING + AI MODEL
# ==============================

def extract_hashtags(text):
    return re.findall(r"#(\w+)", str(text).lower())


def categorize_content(text):
    t = str(text).lower()
    if any(w in t for w in ["vraag", "?", "poll", "question"]):
        return "vraag/poll"
    elif any(w in t for w in ["tip", "advies", "hoe", "guide", "how"]):
        return "educatief"
    elif any(w in t for w in ["nieuw", "new", "launch", "dropping"]):
        return "aankondiging"
    elif any(w in t for w in ["dank", "thanks", "appreciate"]):
        return "interactie"
    elif any(w in t for w in ["link", "bio", "check", "subscribe"]):
        return "promotie"
    return "algemeen"


def caps_ratio(text):
    text = str(text)
    return 0 if len(text) == 0 else sum(1 for c in text if c.isupper()) / len(text)


def engineer_features(combined):
    combined["uur"] = combined["tijd"].dt.hour
    combined["dag"] = combined["tijd"].dt.dayofweek
    combined["tekst_lengte"] = combined["text"].astype(str).str.len()
    combined["woordenaantal"] = combined["text"].astype(str).apply(lambda x: len(x.split()))
    combined["aantal_hashtags"] = combined["text"].apply(lambda x: len(extract_hashtags(x)))
    combined["hashtag_density"] = combined["aantal_hashtags"] / combined["woordenaantal"].replace(0, 1)
    combined["is_vraag"] = combined["text"].astype(str).str.contains(r"\?").astype(int)
    combined["aantal_uitroep"] = combined["text"].astype(str).str.count("!")
    combined["caps_ratio"] = combined["text"].apply(caps_ratio)
    combined["emoji_count"] = combined["text"].astype(str).str.count(r"[^\w\s,]")
    combined["heeft_link"] = combined["text"].str.contains(r"https?://", case=False, na=False).astype(int)
    combined["content_type"] = combined["text"].apply(categorize_content)
    combined = pd.get_dummies(combined, columns=["content_type"], drop_first=True)
    combined["heeft_media"] = combined["heeft_media"].fillna(False).astype(int)
    combined["total_engagement"] = combined["likes"] + combined["retweets"] + combined["replies"] + combined["quotes"]
    return combined


def train_model(df):
    df = df[df["total_engagement"] > 0].copy()
    if len(df) < 8:
        return None, df, "Te weinig data voor een betrouwbaar AI-model (minimaal 8 tweets met engagement nodig). Upload meer data."

    y = np.log1p(df["total_engagement"])
    feature_cols = ["uur", "dag", "aantal_hashtags", "tekst_lengte", "woordenaantal",
                     "hashtag_density", "is_vraag", "aantal_uitroep", "caps_ratio",
                     "emoji_count", "heeft_media", "heeft_link"]
    feature_cols += [c for c in df.columns if c.startswith("content_type_")]
    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=400, max_depth=14, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
    cv_scores = cross_val_score(model, X, y, cv=min(5, max(2, len(df) // 2)))

    df["ai_prediction"] = np.expm1(model.predict(X))
    df["prediction_error"] = df["total_engagement"] - df["ai_prediction"]

    avg_vals = X.mean()
    best_score, best_combo = -1, (0, 0, 0)
    for uur in range(0, 24, 3):
        for hashtags in [0, 1, 2, 3]:
            for media in [0, 1]:
                row = avg_vals.copy()
                row["uur"], row["aantal_hashtags"], row["heeft_media"] = uur, hashtags, media
                pred = np.expm1(model.predict(row.to_frame().T)[0])
                if pred > best_score:
                    best_score, best_combo = pred, (uur, hashtags, media)

    metrics = {"r2": float(r2), "mae": float(mae), "cv_mean": float(cv_scores.mean())}
    return model, df, {"metrics": metrics, "best_combo": best_combo, "best_score": float(best_score)}


# ==============================
# MAIN
# ==============================

def main():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    upload_files = [f for f in os.listdir(UPLOAD_DIR) if not f.startswith(".")]
    if not upload_files:
        print("Geen nieuwe uploads gevonden, niets te doen.")
        return

    nieuwe_dfs = []
    for fname in upload_files:
        path = os.path.join(UPLOAD_DIR, fname)
        with open(path, "rb") as f:
            raw = f.read()
        try:
            if fname.endswith(".csv"):
                nieuwe_dfs.append(parse_csv_export(raw))
            else:
                nieuwe_dfs.append(parse_tweets_js(raw))
        except Exception as e:
            print(f"Kon {fname} niet parsen: {e}")

    if not nieuwe_dfs:
        print("Geen geldige data in de uploads.")
        return

    nieuwe_data = pd.concat(nieuwe_dfs, ignore_index=True)

    if os.path.exists(DATASET_PATH):
        oude_data = pd.read_excel(DATASET_PATH)
        oude_data["tijd"] = pd.to_datetime(oude_data["tijd"], errors="coerce")
        combined = pd.concat([oude_data, nieuwe_data], ignore_index=True)
    else:
        combined = nieuwe_data.copy()

    before = len(combined)
    if "id" in combined.columns:
        combined["id"] = combined["id"].astype(str)
        combined = combined.drop_duplicates(subset="id", keep="last")
    removed = before - len(combined)
    print(f"{removed} duplicaten verwijderd. Totaal: {len(combined)} tweets.")

    combined = combined.dropna(subset=["tijd"]).sort_values("tijd", ascending=False).reset_index(drop=True)

    # Bewaar een schone kopie (zonder features) voor volgende run
    combined[["id", "tijd", "text", "likes", "retweets", "replies", "quotes", "heeft_media"]].to_excel(
        DATASET_PATH, index=False
    )

    engineered = engineer_features(combined.copy())
    model, scored_df, result = train_model(engineered)

    output = {
        "generated_at": int(time.time() * 1000),
        "total_tweets": len(combined),
        "gem_engagement": float(engineered["total_engagement"].mean()),
        "pct_media": float(engineered["heeft_media"].mean()),
    }

    if model is None:
        output["model_trained"] = False
        output["model_message"] = result
    else:
        output["model_trained"] = True
        output["metrics"] = result["metrics"]
        output["best_combo"] = {
            "uur": result["best_combo"][0],
            "hashtags": result["best_combo"][1],
            "media": bool(result["best_combo"][2]),
        }
        output["best_score"] = result["best_score"]
        under = scored_df.sort_values("prediction_error").head(5)
        over = scored_df.sort_values("prediction_error", ascending=False).head(5)
        output["underperformers"] = under[["text", "total_engagement", "ai_prediction"]].to_dict("records")
        output["overperformers"] = over[["text", "total_engagement", "ai_prediction"]].to_dict("records")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Verwerkte uploads opruimen zodat ze niet nogmaals meegenomen worden
    for fname in upload_files:
        os.remove(os.path.join(UPLOAD_DIR, fname))
    # .gitkeep zodat de map blijft bestaan in git
    with open(os.path.join(UPLOAD_DIR, ".gitkeep"), "w") as f:
        f.write("")

    print("Klaar. dataset.xlsx en results.json bijgewerkt.")


if __name__ == "__main__":
    main()
