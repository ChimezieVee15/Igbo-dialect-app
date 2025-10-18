# dialect_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Igbo Dialect Explorer", page_icon="🗣️", layout="centered")
st.title("🗣️ Igbo Dialect Explorer")
st.markdown("Explore Igbo dialects (search, predict, and view similarity). The raw data table is kept in the backend.")

# -------------------------
# Load data (auto or upload)
# -------------------------
DATA_PATH = Path("igbo_dialects_reshaped.csv")
df = None
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    uploaded = st.file_uploader("Upload reshaped CSV (ENGLISH, word, dialect, S/N optional):", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

if df is None:
    st.info("No data available. Upload the reshaped CSV or place igbo_dialects_reshaped.csv next to this file.")
    st.stop()

# Clean and ensure columns
required = ["ENGLISH", "word", "dialect"]
for c in required:
    if c not in df.columns:
        st.error(f"Expected column '{c}' not found in CSV.")
        st.stop()

df["word"] = df["word"].astype(str).str.strip()
df["dialect"] = df["dialect"].astype(str).str.strip()

# -------------------------
# Load model if exists
# -------------------------
MODEL_PATH = Path("outputs/dialect_model.joblib")
model_obj = None
vec = None
clf = None
if MODEL_PATH.exists():
    try:
        model_obj = joblib.load(MODEL_PATH)
        vec = model_obj.get("vectorizer")
        clf = model_obj.get("model")
    except Exception:
        model_obj = None

# -------------------------
# Sidebar controls / summary
# -------------------------
st.sidebar.header("Explore")
dialects = sorted(df["dialect"].unique())
chosen = st.sidebar.selectbox("Select dialect to inspect (counts only):", ["(all)"] + dialects)
if chosen == "(all)":
    counts = df["dialect"].value_counts().reindex(dialects).fillna(0).astype(int)
else:
    counts = pd.Series({chosen: int((df["dialect"] == chosen).sum())})

st.sidebar.write("Dialect counts (sample):")
st.sidebar.table(counts.reset_index().rename(columns={"index":"dialect", 0:"count"}))

# -------------------------
# Search words (backend lookup)
# -------------------------
st.header("Search for an English gloss or Igbo word")
q = st.text_input("Type an English gloss or Igbo word (partial OK)")
if q:
    q_lower = q.strip().lower()
    # search ENGLISH or word columns
    matched = df[df["ENGLISH"].astype(str).str.lower().str.contains(q_lower) |
                 df["word"].astype(str).str.lower().str.contains(q_lower)]
    if matched.empty:
        st.write("No matches found.")
    else:
        # Show limited, aggregated results (not raw table)
        st.write(f"Found {len(matched)} matches. Showing aggregated forms per dialect:")
        agg = matched.groupby("dialect")["word"].apply(lambda x: ", ".join(x.unique()[:10])).reset_index()
        st.table(agg)

# -------------------------
# Prediction UI (uses saved model)
# -------------------------
st.header("Predict dialect from a single word")
word_input = st.text_input("Type a single Igbo word to predict dialect (keep diacritics if present):", value="")
if st.button("Predict"):
    if clf is None or vec is None:
        st.error("No trained model found in outputs/dialect_model.joblib. Train locally and place it in outputs/.")
    else:
        X = vec.transform([word_input.strip().lower()])
        pred = clf.predict(X)[0]
        st.success(f"Predicted dialect: **{pred}**")
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[0]
            dfp = pd.DataFrame({"dialect": clf.classes_, "prob": probs}).sort_values("prob", ascending=False)
            st.table(dfp.round(3))

# -------------------------
# Similarity heatmap (backend)
# -------------------------
st.header("Dialect similarity (backend)")
if st.button("Compute similarity heatmap"):
    # Build dialect-by-character-ngram TF-IDF vectors (approx) using simple char-frequency matrix
    # Aggregate words per dialect into one string
    texts = df.groupby("dialect")["word"].apply(lambda ws: " ".join(ws)).reindex(dialects).fillna("")
    # Create a simple char-profile (counts) for each dialect
    char_features = []
    chars = list({ch for s in texts for ch in s})
    for s in texts:
        counts = [s.count(ch) for ch in chars]
        char_features.append(counts)
    M = np.array(char_features, dtype=float)
    # cosine similarity
    if M.sum() == 0:
        st.write("Not enough textual data to compute similarity.")
    else:
        sim = cosine_similarity(M)
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(dialects))); ax.set_yticks(range(len(dialects)))
        ax.set_xticklabels(dialects, rotation=45, ha="right"); ax.set_yticklabels(dialects)
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", color="white" if sim[i,j]>0.5 else "black")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

# -------------------------
# Admin: show raw table only when authorized
# -------------------------
st.markdown("---")
st.write("Admin options")
show_admin = st.checkbox("I am admin (show raw data)")

# simple password prompt 
if show_admin:
    pwd = st.text_input("Enter admin password", type="password")
    if pwd == "pearlvee":      
        st.warning("Raw table visible to admin")
        st.dataframe(df)         # admin-only view
    else:
        st.error("Wrong password or not provided. Raw table hidden.")
