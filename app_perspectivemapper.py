import os
from pathlib import Path
import re
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Configure page FIRST
st.set_page_config(page_title="PerspectiveMapper", page_icon="🧭", layout="wide")

# NLP / ML
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# Tokenization & Stopwords
import stopwordsiso as stopwordsiso
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# WordCloud & Viz
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

# File handling
from docx import Document as DocxDocument  # python-docx
import pdfplumber  # <-- added for PDFs

# Sentence embeddings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sentence_transformers import SentenceTransformer

# Sentiment (CardiffNLP + fallback to VADER)
cardiff_error = None
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    CARDIFF_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    _cardiff_tokenizer = AutoTokenizer.from_pretrained(CARDIFF_MODEL_NAME)
    _cardiff_model = AutoModelForSequenceClassification.from_pretrained(CARDIFF_MODEL_NAME)
    _use_cardiff = True
except Exception as e:
    cardiff_error = str(e)
    _use_cardiff = False
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_analyzer = SentimentIntensityAnalyzer()

# -----------------------------
# Password Gate
# -----------------------------
def gate():
    """Simple username/password gate using st.secrets['passwords']"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    st.title("🔐 PerspectiveMapper – Access")
    st.write("Enter your username and password.")
    user = st.text_input("Username", key="auth_user")
    pwd = st.text_input("Password", type="password", key="auth_pwd")

    if st.button("Log in"):
        try:
            if "passwords" in st.secrets and user in st.secrets["passwords"]:
                if str(st.secrets["passwords"][user]) == str(pwd):
                    st.session_state.authenticated = True
                    st.success("Access granted ✅")
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            else:
                st.error("User not found in [passwords].")
        except Exception as e:
            st.error(f"Could not validate access: {e}")
    st.stop()

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    return SentenceTransformer(model_name)

def guess_lang(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"

def collect_stopwords(selected_langs: List[str], extra_stop: List[str]) -> set:
    sw = set()
    for lang in selected_langs:
        try:
            sw |= set(stopwordsiso.stopwords(lang))
        except:
            pass
    sw |= set([w.strip().lower() for w in extra_stop if w.strip()])
    sw |= set(["http", "https", "www", "com"])
    return sw

def simple_tokenize(text: str) -> List[str]:
    text = re.sub(r"http\S+|www\.\S+", " ", text, flags=re.I)
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9_]+", text.lower(), flags=re.U)
    return tokens

def preprocess_docs(docs: List[Dict], stop_set: set) -> List[str]:
    cleaned = []
    for d in docs:
        toks = [t for t in simple_tokenize(d["text"]) if t not in stop_set and len(t) > 2]
        cleaned.append(" ".join(toks))
    return cleaned

def read_file(upload) -> str:
    """Handles txt, docx, pdf"""
    name = upload.name.lower()
    if name.endswith(".txt"):
        return upload.read().decode("utf-8", errors="ignore")
    elif name.endswith(".docx"):
        doc = DocxDocument(upload)
        return "\n".join([p.text for p in doc.paragraphs])
    elif name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(upload) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        return ""

def make_wordcloud(text: str):
    wc = WordCloud(width=1000, height=600, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)

def top_words_per_topic(lda, feature_names, n_top=10):
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_top:][::-1]
        words = [feature_names[i] for i in top_indices]
        topics.append({"topic": topic_idx, "top_words": ", ".join(words)})
    return pd.DataFrame(topics)

def run_cardiff_sentiment(texts: List[str]) -> Tuple[List[str], List[float]]:
    labels, scores = [], []
    if not _use_cardiff:
        return labels, scores
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    for t in texts:
        try:
            inputs = _cardiff_tokenizer(t, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = _cardiff_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            labels.append(id2label[idx])
            scores.append(float(probs[idx]))
        except Exception:
            labels.append("neutral")
            scores.append(0.0)
    return labels, scores

def run_vader_sentiment(texts: List[str]) -> Tuple[List[str], List[float]]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    labels, scores = [], []
    for t in texts:
        res = an.polarity_scores(t)
        comp = res["compound"]
        if comp >= 0.05:
            labels.append("positive")
        elif comp <= -0.05:
            labels.append("negative")
        else:
            labels.append("neutral")
        scores.append(float(comp))
    return labels, scores

def build_bias_table(texts: List[str], bias_dict: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    lowered = [t.lower() for t in texts]
    for i, t in enumerate(lowered):
        row = {"doc_id": i}
        total = max(len(t.split()), 1)
        for label, kws in bias_dict.items():
            hits = 0
            for kw in kws:
                kw = kw.lower().strip()
                if not kw:
                    continue
                hits += len(re.findall(rf"\b{re.escape(kw)}\b", t))
            row[label] = hits / total
        rows.append(row)
    return pd.DataFrame(rows)

# -----------------------------
# APP
# -----------------------------
gate()  # require password

# Header
c1, c2 = st.columns([1,4])
with c1:
    logo_candidates = [
        Path(_file_).parent / "assets" / "logo.png",
        Path.cwd() / "assets" / "logo.png",
        "assets/logo.png",
    ]
    logo_path = None
    for p in logo_candidates:
        try:
            if isinstance(p, str):
                if os.path.exists(p):
                    logo_path = p
                    break
            else:
                if p.exists():
                    logo_path = str(p)
                    break
        except Exception:
            pass
    if logo_path:
        st.image(logo_path, use_container_width=True)
with c2:
    st.title("PerspectiveMapper")
    st.write("Discourse analysis: topics, sentiment, similarity, and bias.")

# Sidebar
st.sidebar.header("⚙ Settings")
uploads = st.sidebar.file_uploader(
    "Upload .txt, .docx, or .pdf files",
    type=["txt","docx","pdf"],
    accept_multiple_files=True
)
lang_codes = st.sidebar.multiselect("Stopword languages", ["en","es","it","fr","de","pt","ca","eu","gl"], default=["en","es","it"])
extra_sw = st.sidebar.text_area("Extra stopwords (comma-separated)", value="and, the, of, to")
show_wc = st.sidebar.checkbox("Show WordCloud per document", value=True)
n_topics = st.sidebar.slider("LDA topics", 2, 12, 5)
max_features = st.sidebar.slider("Max vocabulary size", 1000, 10000, 3000, step=500)
n_clusters = st.sidebar.slider("KMeans clusters", 2, 12, 4)
use_cardiff = st.sidebar.checkbox("Use CardiffNLP sentiment", value=True)
bias_json = st.sidebar.text_area("Bias dictionary JSON", value=json.dumps({
    "gender": ["woman","man","trans","equality"],
    "migration": ["immigrant","migrant","refugee","border"],
    "religion": ["church","islam","catholic","jewish"],
    "politics": ["left","right","liberal","conservative"]
}, indent=2))
want_csv = st.sidebar.checkbox("Enable CSV download", value=True)

# Main
if not uploads:
    st.info("⬅ Upload one or more files to begin the analysis.")
    st.stop()

docs = []
for up in uploads:
    text = read_file(up)
    lang = guess_lang(text) if text.strip() else "en"
    docs.append({"filename": up.name, "text": text, "lang": lang})

st.success(f"Loaded {len(docs)} document(s).")

stop_set = collect_stopwords(lang_codes, [w.strip() for w in extra_sw.split(",")])
cleaned = preprocess_docs(docs, stop_set)

# WordClouds
if show_wc:
    st.subheader("☁ WordCloud per document")
    for d, text in zip(docs, cleaned):
        st.markdown(f"{d['filename']}")
        if text.strip():
            make_wordcloud(text)
        else:
            st.caption("Empty after preprocessing.")

# LDA
st.subheader("🧵 Topic Modeling (LDA)")
vectorizer = CountVectorizer(max_features=max_features)
X = vectorizer.fit_transform(cleaned)
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
W = lda.fit_transform(X)
topics_df = top_words_per_topic(lda, vectorizer.get_feature_names_out(), n_top=12)
st.write(topics_df)
dominant_topic = np.argmax(W, axis=1).tolist()

# Clustering (safe PCA/KMeans)
st.subheader("🧭 Clustering (SBERT + PCA)")
embedder = get_embedder()
embeddings = embedder.encode([d["text"] for d in docs], show_progress_bar=False)
n_docs = len(docs)
if n_docs < 2:
    coords = np.zeros((n_docs, 2))
else:
    n_comp = min(2, n_docs, embeddings.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    coords_pca = pca.fit_transform(embeddings)
    coords = coords_pca if coords_pca.shape[1] == 2 else np.hstack([coords_pca, np.zeros((n_docs,1))])
df_plot = pd.DataFrame({
    "x": coords[:,0],
    "y": coords[:,1],
    "file":[d["filename"] for d in docs],
    "topic":[f"T{t}" for t in dominant_topic]
})
k_for_fit = min(max(1, n_clusters), n_docs)
clusters = np.zeros(n_docs, dtype=int) if k_for_fit<2 else KMeans(n_clusters=k_for_fit, random_state=42, n_init="auto").fit_predict(coords)
df_plot["cluster"] = clusters
st.plotly_chart(px.scatter(df_plot, x="x", y="y", text="file", color="cluster"), use_container_width=True)

# Hierarchical Clustering Dendrogram
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

st.subheader("🌳 Cluster Dendrogram (hierarchical clustering)")
dist_matrix = pdist(embeddings, metric="cosine")
linkage = sch.linkage(dist_matrix, method="ward")
fig, ax = plt.subplots(figsize=(8, 5))
sch.dendrogram(
    linkage,
    labels=[d["filename"] for d in docs],
    orientation="top",
    leaf_rotation=45,
    leaf_font_size=10,
    ax=ax,
)
st.pyplot(fig)

# Similarity
st.subheader("🔗 Similarity matrix")
sim = cosine_similarity(embeddings)
heatmap = ff.create_annotated_heatmap(
    z=sim,
    x=[d["filename"] for d in docs],
    y=[d["filename"] for d in docs],
    showscale=True,
    colorscale="Viridis"
)
st.plotly_chart(heatmap, use_container_width=True)

# Sentiment
st.subheader("💬 Sentiment")
texts=[d["text"] for d in docs]
if use_cardiff and _use_cardiff:
    labels, scores = run_cardiff_sentiment(texts)
else:
    labels, scores = run_vader_sentiment(texts)

# Bias indicator
st.subheader("🧷 Bias indicator")
try:
    bias_dict = json.loads(bias_json)
except Exception as e:
    st.error(f"Invalid JSON for bias keywords: {e}")
    bias_dict = {}
bias_df = build_bias_table(texts, bias_dict) if bias_dict else pd.DataFrame()

# Results
st.subheader("📊 Results")
results = pd.DataFrame({
    "file":[d["filename"] for d in docs],
    "language":[d["lang"] for d in docs],
    "tokens_len":[len(c.split()) for c in cleaned],
    "lda_dominant_topic":dominant_topic,
    "cluster":clusters,
    "pca_x":coords[:,0],
    "pca_y":coords[:,1]
})
if not bias_df.empty:
    results = results.merge(bias_df, left_index=True, right_on="doc_id", how="left").drop(columns=["doc_id"])
st.write(results)

if want_csv:
    st.download_button(
        "⬇ Download CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="perspectivemapper_results.csv"
    )
