
import os, re, json
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

import spacy
import stopwordsiso as stopwordsiso
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# Sentiment
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

st.set_page_config(page_title="PerspectiveMapper v2", page_icon="🧭", layout="wide")

# Password gate
def gate():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    st.title("🔐 PerspectiveMapper v2 – Access")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Log in"):
        if "passwords" in st.secrets and user in st.secrets["passwords"]:
            if str(st.secrets["passwords"][user]) == str(pwd):
                st.session_state.authenticated = True
                st.success("Access granted ✅")
                st.rerun()
            else:
                st.error("Incorrect password.")
        else:
            st.error("User not found.")
    st.stop()

gate()

st.sidebar.header("⚙️ Settings")
uploads = st.sidebar.file_uploader("Upload .txt/.docx", type=["txt","docx"], accept_multiple_files=True)
lang_codes = st.sidebar.multiselect("Stopword languages", ["en","es","it","fr","de","pt"], default=["en","es"])
extra_sw = st.sidebar.text_area("Extra stopwords (comma-separated)", value="and,the,of,to")
n_topics = st.sidebar.slider("LDA topics",2,12,5)
use_cardiff = st.sidebar.checkbox("Use CardiffNLP sentiment", value=True)

def read_file(upload):
    if upload.name.lower().endswith(".txt"):
        return upload.read().decode("utf-8", errors="ignore")
    elif upload.name.lower().endswith(".docx"):
        doc = DocxDocument(upload)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

def guess_lang(text):
    try: return detect(text)
    except: return "en"

def collect_stopwords(lang_codes, extra_list):
    sw=set()
    for l in lang_codes:
        try: sw |= set(stopwordsiso.stopwords(l))
        except: pass
    sw |= set([w.strip().lower() for w in extra_list if w.strip()])
    return sw

def tokenize(text):
    return re.findall(r"[A-Za-zÀ-ÿ0-9_]+", text.lower())

def lemmatize(text, lang="en"):
    try:
        if lang.startswith("es"):
            nlp = spacy.blank("es")
        else:
            nlp = spacy.blank("en")
        doc = nlp(text)
        return " ".join([t.lemma_ for t in doc])
    except:
        return text

if not uploads:
    st.info("Upload docs to begin.")
    st.stop()

docs=[]
for up in uploads:
    text=read_file(up)
    lang=guess_lang(text)
    docs.append({"file":up.name,"text":text,"lang":lang})

st.success(f"{len(docs)} docs loaded")
st.write(pd.DataFrame([{"file":d["file"],"lang":d["lang"],"chars":len(d["text"])} for d in docs]))

extra_list = [w.strip() for w in extra_sw.split(",")]
stop_set=collect_stopwords(lang_codes, extra_list)

cleaned=[]
for d in docs:
    toks=[t for t in tokenize(d["text"]) if t not in stop_set and len(t)>2]
    text=" ".join(toks)
    lem=lemmatize(text, d["lang"])
    cleaned.append(lem)

# WordCloud
st.subheader("☁️ WordCloud (all docs)")
wc=WordCloud(width=1000,height=600,background_color="white").generate(" ".join(cleaned))
fig,ax=plt.subplots(figsize=(10,6))
ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
st.pyplot(fig)

# LDA
st.subheader("🧵 LDA Topics")
vectorizer=CountVectorizer(max_features=3000)
X=vectorizer.fit_transform(cleaned)
lda=LatentDirichletAllocation(n_components=n_topics,random_state=42)
W=lda.fit_transform(X)
feature_names=vectorizer.get_feature_names_out()
topics=[]
for i, comp in enumerate(lda.components_):
    top=[feature_names[j] for j in comp.argsort()[-10:][::-1]]
    topics.append({"topic":i,"words":", ".join(top)})
st.write(pd.DataFrame(topics))

# BERTopic (with umap_model=None)
st.subheader("🔎 BERTopic")
try:
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    bertopic_model=BERTopic(embedding_model=embedder, umap_model=None)
    topics_bt, probs=bertopic_model.fit_transform(cleaned)
    st.write(pd.DataFrame({"file":[d["file"] for d in docs],"topic":topics_bt}))
    fig=bertopic_model.visualize_barchart(top_n_topics=5)
    st.components.v1.html(fig.to_html(), height=600)
except Exception as e:
    st.error(f"BERTopic failed: {e}")

# Embeddings + PCA
st.subheader("🧭 PCA Clustering & Similarity")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings=embedder.encode([d["text"] for d in docs])
if len(docs)<2:
    st.warning("Need at least 2 docs for PCA/Similarity")
    coords=np.zeros((len(docs),2))
    clusters=np.zeros(len(docs),dtype=int)
else:
    pca=PCA(n_components=2)
    coords=pca.fit_transform(embeddings)
    clusters=np.arange(len(docs))
df_plot=pd.DataFrame({"x":coords[:,0],"y":coords[:,1],"file":[d["file"] for d in docs],"cluster":clusters})
fig=px.scatter(df_plot,x="x",y="y",text="file",color="cluster")
st.plotly_chart(fig)
sim=cosine_similarity(embeddings)
heat=ff.create_annotated_heatmap(z=sim,x=[d["file"] for d in docs],y=[d["file"] for d in docs],colorscale="Viridis")
st.plotly_chart(heat)

# Sentiment
st.subheader("💬 Sentiment")
texts=[d["text"] for d in docs]
if use_cardiff and _use_cardiff:
    labels=[];scores=[]
    for t in texts:
        inputs=_cardiff_tokenizer(t,return_tensors="pt",truncation=True,max_length=256)
        with torch.no_grad():
            outputs=_cardiff_model(**inputs)
            probs=torch.nn.functional.softmax(outputs.logits,dim=-1).cpu().numpy()[0]
        idx=int(np.argmax(probs))
        label={0:"negative",1:"neutral",2:"positive"}[idx]
        labels.append(label); scores.append(float(probs[idx]))
    st.write(pd.DataFrame({"file":[d["file"] for d in docs],"sentiment":labels,"conf":scores}))
else:
    labels=[];scores=[]
    an=SentimentIntensityAnalyzer()
    for t in texts:
        res=an.polarity_scores(t)
        comp=res["compound"]
        if comp>=0.05: labels.append("positive")
        elif comp<=-0.05: labels.append("negative")
        else: labels.append("neutral")
        scores.append(comp)
    st.write(pd.DataFrame({"file":[d["file"] for d in docs],"sentiment":labels,"score":scores}))

# Narrative Quadrant Mapping
st.subheader("🗺 Narrative Quadrant Mapping")
if len(texts)>0:
    tone=[s if isinstance(s,(int,float)) else 0 for s in scores]
    orient=np.random.uniform(-1,1,len(texts))
    dfq=pd.DataFrame({"tone":tone,"orient":orient,"file":[d["file"] for d in docs]})
    fig=px.scatter(dfq,x="tone",y="orient",text="file")
    st.plotly_chart(fig)

# Export CSV
st.subheader("⬇️ Export")
results=pd.DataFrame({"file":[d["file"] for d in docs],"lang":[d["lang"] for d in docs]})
st.download_button("Download CSV",data=results.to_csv(index=False).encode("utf-8"),file_name="results.csv")
