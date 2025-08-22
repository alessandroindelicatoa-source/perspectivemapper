# PerspectiveMapper (Streamlit)

A text analysis app with **password-protected access**, **file upload (.txt, .docx)**, **multilingual stopword removal**, **WordCloud**, **LDA (scikit-learn)**, **Sentence-BERT + PCA + KMeans**, **semantic similarity matrix**, **sentiment (CardiffNLP with VADER fallback)**, **keyword-based bias indicator**, and **CSV export**.

## 🚀 Quick deploy on Streamlit Cloud
1. **Create a GitHub repo** and add:
   - `app_perspectivemapper.py`
   - `requirements.txt`
   - `README.md`
   - `assets/logo.png`
   - `.streamlit/secrets.toml` (see below)

2. On Streamlit Cloud → **New app**, point to your repo & branch, and set the **main file** to `app_perspectivemapper.py`.

3. **Set the password(s)** under **Manage app → Settings → Secrets**:
   ```toml
   [passwords]
   user = "your_secure_password"
   ```
   Add more users if needed:
   ```toml
   [passwords]
   alice = "abc123"
   bob = "anotherSecureOne"
   ```

4. **First model download**: the CardiffNLP model will download the first time (may take a bit). If it fails, the app will automatically use VADER.

## 🧪 Local usage
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app_perspectivemapper.py
```
Create `.streamlit/secrets.toml` with:
```toml
[passwords]
user = "your_secure_password"
```

## 📝 Features
- **Password-protected** via `st.secrets`.
- **Uploader** for `.txt` & `.docx`.
- **Simple tokenization** + **multilingual ISO stopwords** and custom extra stopwords.
- **WordCloud** per document.
- **LDA** with adjustable number of topics and vocabulary size.
- **Embeddings** with `SentenceTransformer` (`paraphrase-multilingual-MiniLM-L12-v2`).
- **PCA + KMeans** interactive scatter (Plotly).
- **Cosine similarity matrix** heatmap.
- **Sentiment** using **CardiffNLP** if available; otherwise **VADER**.
- **Bias indicator** customizable via JSON (`{category: [words,...]}`).
- **CSV export** of results.

## 🔒 Citation
> If you use this application, please cite: **Indelicato & Martín (2025), PerspectiveMapper App (v1.0)**.

## 🛠️ Notes
- Language detection uses `langdetect` (fast and simple). You can set the stopword languages in the **sidebar**.
- For **more advanced preprocessing** (lemmatization), plug in spaCy with multilingual models if you wish.
- You can manually disable CardiffNLP in the **sidebar** toggle.

---
© 2025 – PerspectiveMapper