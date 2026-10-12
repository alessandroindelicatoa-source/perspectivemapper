from __future__ import annotations

import io
import json
import math
import re
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    davies_bouldin_score, f1_score, silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

try:
    import simplemma
    _HAS_SIMPLEMMA = True
except Exception:
    _HAS_SIMPLEMMA = False


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTIMENT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

LANG_NLTK = {
    "en": "english", "es": "spanish", "it": "italian", "fr": "french",
    "de": "german", "pt": "portuguese", "nl": "dutch", "ru": "russian",
    "ar": "arabic", "tr": "turkish", "da": "danish", "fi": "finnish",
    "no": "norwegian", "sv": "swedish", "ro": "romanian", "hu": "hungarian",
}

FALLBACK_STOPWORDS = {
    "en": set("""a about above after again against all am an and any are as at be because been before being below between both but by can could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with would you your yours yourself yourselves""".split()),
    "es": set("""a al algo algunas algunos ante antes como con contra cual cuando de del desde donde durante e el ella ellas ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaba estaban estado estas este esto estos fue fueron ha hasta hay la las le les lo los más me mi mis mucho muy no nos o os para pero por porque que quien se sea ser si sin sobre su sus también te tiene todo tu tus un una unas uno unos ya y yo son son los las del""".split()),
    "it": set("""a ad al alla allo ai agli alle anche ancora avere aveva con contro cui da dal dalla dallo dai dagli dalle de del della dello dei degli delle di e ed era erano essere fa fra gli ha hanno i il in io la le lei lo loro ma mi mia mie miei mio ne nei nel nella nello no noi non o per più quale quando che chi se sei si sia sono su sul sulla tra un una uno vi voi""".split()),
    "fr": set("""a à au aux avec ce ces cette dans de des du elle elles en est et eux il ils je la le les leur leurs lui mais me mes moi mon ne nos notre nous on ou où par pas pour qu que quelle qui sa se ses son sur ta te tes toi ton tu un une vos votre vous""".split()),
    "pt": set("""a ao aos aquela aquelas aquele aqueles as até com como da das de dela delas dele deles do dos e ela elas ele eles em entre era eram essa essas esse esses esta estas este estes eu foi foram há isso isto já lhe lhes mais mas me meu meus minha minhas muito na nas não no nos nós o os ou para pela pelas pelo pelos por qual quando que quem se sem seu seus sua suas também te tem uma umas um uns você vocês""".split()),
    "de": set("""aber als am an auch auf aus bei bin bis bist da dadurch daher darum das dass dein deine dem den der des die dies diese ein eine einem einen einer eines er es für hat hatte haben hier ich im in ist ja kann kein keine mit muss nach nicht nun oder seid sein seine sind so über um und uns unser unter vom von vor war waren warst was weg weil weiter welche wenn werde werden wie wieder wir wird wo zu zum zur""".split()),
}

DEFAULT_FRAMES = {}



def _normalise_search_text(value: object, case_sensitive: bool = False, accent_insensitive: bool = True) -> str:
    text = str(value if value is not None else "")
    if accent_insensitive:
        text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    if not case_sensitive:
        text = text.casefold()
    return text


def parse_keyword_terms(raw: object) -> List[str]:
    """Parse a user keyword filter. Separate terms/phrases with comma, semicolon or newline.

    A value without separators is treated as one phrase, so ``climate change`` searches
    for that phrase rather than two independent tokens.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        parts = [str(x).strip() for x in raw]
    else:
        parts = [x.strip() for x in re.split(r"[,;\n]+", str(raw))]
    out = []
    seen = set()
    for x in parts:
        x = x.strip().strip('"').strip("'")
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out


def filter_documents_by_keywords(
    docs: pd.DataFrame, terms: Sequence[str], mode: str = "ANY", scope: str = "text",
    metadata_cols: Optional[Sequence[str]] = None, case_sensitive: bool = False,
    accent_insensitive: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter document rows by words/phrases while preserving the document as the unit of analysis.

    Returns the filtered documents plus an audit table with per-document matched terms and counts.
    ``scope`` can be ``text``, ``metadata`` or ``both``. Matching is boundary-aware and, by
    default, case- and accent-insensitive.
    """
    if docs is None or docs.empty:
        return pd.DataFrame(), pd.DataFrame()
    clean_terms = [str(t).strip() for t in terms if str(t).strip()]
    if not clean_terms:
        audit = docs[[c for c in ["doc_id"] if c in docs.columns]].copy()
        audit["keyword_matches"] = ""
        audit["keyword_match_count"] = 0
        audit["keyword_filter_kept"] = True
        return docs.copy(), audit

    scope = str(scope).lower()
    mode = str(mode).upper()
    meta = [c for c in (metadata_cols or []) if c in docs.columns and c != "text"]
    rows = []
    keep_idx = []
    for idx, r in docs.iterrows():
        fields = []
        if scope in {"text", "both"}:
            fields.append(str(r.get("text", "")))
        if scope in {"metadata", "both"}:
            fields.extend(str(r.get(c, "")) for c in meta if not pd.isna(r.get(c, np.nan)))
        hay = _normalise_search_text(" ".join(fields), case_sensitive, accent_insensitive)
        matched = []
        total = 0
        for term in clean_terms:
            needle = _normalise_search_text(term, case_sensitive, accent_insensitive)
            pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
            count = len(re.findall(pattern, hay, flags=re.UNICODE))
            if count > 0:
                matched.append(term)
                total += count
        kept = (len(matched) == len(clean_terms)) if mode == "ALL" else (len(matched) > 0)
        if kept:
            keep_idx.append(idx)
        rows.append({
            "doc_id": str(r.get("doc_id", idx)),
            "keyword_matches": ", ".join(matched),
            "keyword_match_count": int(total),
            "keyword_filter_kept": bool(kept),
        })
    audit = pd.DataFrame(rows)
    out = docs.loc[keep_idx].copy()
    if not out.empty and "doc_id" in out.columns:
        out = out.merge(audit[["doc_id","keyword_matches","keyword_match_count"]], on="doc_id", how="left")
    return out.reset_index(drop=True), audit


def grouped_numeric_summary(df: pd.DataFrame, group_col: str, outcome_col: str, n_boot: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Document-level descriptive summary by a metadata group, with bootstrap mean CIs."""
    if df is None or df.empty or group_col not in df.columns or outcome_col not in df.columns:
        return pd.DataFrame()
    work = df[[group_col, outcome_col]].copy()
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.dropna(subset=[group_col, outcome_col])
    rows = []
    for g, sub in work.groupby(group_col, dropna=False, sort=False):
        vals = sub[outcome_col].to_numpy(dtype=float)
        ci = bootstrap_mean_ci(vals, n_boot=n_boot, random_state=random_state)
        rows.append({
            group_col: g, "n_documents": int(len(vals)), "mean": float(np.mean(vals)),
            "median": float(np.median(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "ci_low_95": ci["ci_low"], "ci_high_95": ci["ci_high"],
        })
    return pd.DataFrame(rows)


def safe_language(text: str) -> str:
    sample = re.sub(r"\s+", " ", str(text or "")).strip()[:2500]
    if not sample:
        return "und"
    if _HAS_LANGDETECT:
        try:
            return detect(sample)
        except Exception:
            pass
    # Conservative fallback for the three main UI languages.
    lower = f" {sample.lower()} "
    scores = {
        "es": sum(lower.count(f" {w} ") for w in ["que","los","las","una","para","con","del"]),
        "it": sum(lower.count(f" {w} ") for w in ["che","gli","una","per","con","della","del"]),
        "en": sum(lower.count(f" {w} ") for w in ["the","and","that","for","with","from","this"]),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "und"


_STOPWORD_CACHE: Dict[str, set] = {}

def get_stopwords(lang: str, extra: Optional[Iterable[str]] = None) -> set:
    if lang not in _STOPWORD_CACHE:
        words = set(FALLBACK_STOPWORDS.get(lang, set()))
        if _HAS_NLTK and lang in LANG_NLTK:
            try:
                words |= set(nltk_stopwords.words(LANG_NLTK[lang]))
            except Exception:
                # Do not trigger network downloads at analysis time. The bundled fallback lists
                # keep the app usable in offline/private deployments.
                pass
        _STOPWORD_CACHE[lang] = words
    words = set(_STOPWORD_CACHE[lang])
    if extra:
        words |= {str(x).strip().lower() for x in extra if str(x).strip()}
    words |= {"http", "https", "www", "com"}
    return words


def tokenize(text: str) -> List[str]:
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text or ""), flags=re.I)
    return re.findall(r"[^\W\d_][^\W_]*(?:['’\-][^\W\d_]+)?", text.lower(), flags=re.UNICODE)


_LEMMA_CACHE: Dict[Tuple[str, str], str] = {}


def lemmatizer_available() -> bool:
    return _HAS_SIMPLEMMA


def lemmatize_token(token: str, lang: str) -> str:
    """Dictionary-based lemma (simplemma). Unknown languages/tokens are returned unchanged."""
    if not _HAS_SIMPLEMMA or not lang or lang == "und":
        return token
    key = (lang, token)
    hit = _LEMMA_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        lemma = str(simplemma.lemmatize(token, lang=lang)).lower()
    except Exception:
        lemma = token
    if len(_LEMMA_CACHE) < 500_000:
        _LEMMA_CACHE[key] = lemma
    return lemma


def clean_for_lexical(text: str, lang: str, extra_stop: Optional[Iterable[str]] = None,
                      lemmatize: bool = False) -> str:
    """Lower-cased content tokens for lexical models.

    Stopwords are removed before and, when ``lemmatize`` is true, again after lemmatisation,
    so that inflected function words (e.g. "sus" -> "su") do not re-enter the vocabulary.
    """
    sw = get_stopwords(lang, extra_stop)
    toks = [t for t in tokenize(text) if len(t) >= 3 and t not in sw]
    if lemmatize:
        toks = [lemmatize_token(t, lang) for t in toks]
        toks = [t for t in toks if len(t) >= 3 and t not in sw]
    return " ".join(toks)


# ---------------------------------------------------------------------------
# Extracted-text cleaning (PDF/DOCX). Designed for institutional PDFs such as the
# vatican.va encyclicals: running headers/footers, page numbers, hyphenation at line
# ends, hard line wraps and end-of-document notes.
# ---------------------------------------------------------------------------

_NOTES_HEADINGS = r"(?:notas|notes|note|anmerkungen|notes?\s+et\s+r[ée]f[ée]rences|notas\s+finales|riferimenti)"


def _header_key(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def find_repeated_page_lines(pages: Sequence[str], edge_lines: int = 2, min_share: float = 0.4) -> set:
    """Lines (digits normalised) that recur at the top/bottom of many pages."""
    n = len(pages)
    if n < 3:
        return set()
    counts: Counter = Counter()
    for p in pages:
        lines = [l for l in str(p).splitlines() if l.strip()]
        if len(lines) < 3:
            continue  # too short to tell headers from body text
        edge = lines[:edge_lines] + lines[-edge_lines:]
        # Running heads are short and rarely end like a sentence; body lines that happen to
        # recur (e.g. a repeated formula) usually do, so they are not treated as headers.
        counts.update({_header_key(l) for l in edge
                       if len(l.strip()) <= 120 and (len(l.strip()) <= 25 or not re.search(r"[.;:!?»”\"]$", l.strip()))})
    threshold = max(3, int(math.ceil(min_share * n)))
    return {k for k, c in counts.items() if c >= threshold}


def clean_extracted_text(text: str, repeated_lines: Optional[set] = None, strip_notes: bool = False) -> str:
    """Repair common PDF-extraction artefacts while keeping paragraph boundaries."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = t.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    lines = t.split("\n")
    rep = repeated_lines or set()
    kept = []
    for line in lines:
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if re.fullmatch(r"[-–—\s]*\d{1,4}[-–—\s]*", s) or re.fullmatch(r"(?i)(page|p[áa]g(?:ina)?\.?|pagina)\s*\d+(\s*(of|de|di)\s*\d+)?", s):
            continue
        if rep and _header_key(s) in rep:
            continue
        kept.append(s)
    t = "\n".join(kept)
    if strip_notes:
        m = None
        for m_ in re.finditer(rf"(?im)^\s*{_NOTES_HEADINGS}\s*:?\s*$", t):
            m = m_
        if m is not None and m.start() >= 0.5 * len(t):
            t = t[:m.start()]
        t = re.sub(r"\[\d{1,3}\]", "", t)
    # Hyphenation across line breaks: "eco-\nnomía" -> "economía" (only before lower case).
    t = re.sub(r"(\w)[-‐]\n(?=[a-zà-öø-ÿ])", r"\1", t)
    # Re-join hard-wrapped lines inside paragraphs; blank lines remain paragraph breaks.
    t = re.sub(r"(?<![.!?:;»\"”])\n(?!\n)(?=\S)", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Dictionary matching with wildcards and accent-insensitive search
# ---------------------------------------------------------------------------

def compile_term_pattern(term: str, accent_insensitive: bool = True) -> Optional[re.Pattern]:
    """Compile a dictionary entry. ``*`` is a word-internal wildcard: ``salari*``, ``*capital*``.

    Matching is whole-word, case-insensitive and (optionally) accent-insensitive. Multi-word
    entries tolerate any amount of whitespace between words.
    """
    raw = _normalise_search_text(str(term).strip(), False, accent_insensitive)
    if not raw or raw.replace("*", "").strip() == "":
        return None
    parts = [re.escape(p) for p in raw.split("*")]
    body = r"\w*".join(parts)
    body = re.sub(r"(\\ )+", r"\\s+", body)
    return re.compile(r"(?<!\w)" + body + r"(?!\w)", flags=re.UNICODE)


def dictionary_term_counts(docs: pd.DataFrame, frame_dict: Dict[str, List[str]],
                           accent_insensitive: bool = True) -> pd.DataFrame:
    """Long audit table: hits of every dictionary entry in every document (non-zero rows only)."""
    compiled = {(cat, str(term)): compile_term_pattern(term, accent_insensitive)
                for cat, terms in frame_dict.items() for term in (terms or [])}
    rows = []
    for _, r in docs.iterrows():
        hay = _normalise_search_text(r["text"], False, accent_insensitive)
        for (cat, term), pat in compiled.items():
            if pat is None:
                continue
            hits = pat.findall(hay)
            if hits:
                rows.append({"doc_id": r["doc_id"], "category": cat, "entry": term, "hits": len(hits),
                             "matched_forms": ", ".join(sorted(set(hits))[:15])})
    return pd.DataFrame(rows, columns=["doc_id", "category", "entry", "hits", "matched_forms"])


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    # Unicode-friendly lightweight sentence segmentation; avoids a language model dependency.
    parts = re.split(r"(?<=[.!?。！？])\s+(?=[A-ZÀ-ÖØ-Ý0-9¿¡\"“‘])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_words: int = 140, overlap_words: int = 20) -> List[str]:
    max_words = max(40, int(max_words))
    overlap_words = max(0, min(int(overlap_words), max_words // 2))
    sents = split_sentences(text)
    if not sents:
        return []
    chunks: List[str] = []
    current: List[str] = []
    current_n = 0
    for sent in sents:
        sw = sent.split()
        if len(sw) > max_words:
            # Flush sentence buffer first.
            if current:
                chunks.append(" ".join(current).strip())
                current, current_n = [], 0
            step = max_words - overlap_words if max_words > overlap_words else max_words
            for start in range(0, len(sw), step):
                piece = sw[start:start + max_words]
                if piece:
                    chunks.append(" ".join(piece))
                if start + max_words >= len(sw):
                    break
            continue
        if current and current_n + len(sw) > max_words:
            chunks.append(" ".join(current).strip())
            overlap = " ".join(current).split()[-overlap_words:] if overlap_words else []
            current = [" ".join(overlap)] if overlap else []
            current_n = len(overlap)
        current.append(sent)
        current_n += len(sw)
    if current:
        chunks.append(" ".join(current).strip())
    # Remove accidental duplicate/very short overlap-only tails.
    out = []
    for c in chunks:
        if len(c.split()) >= 10 or len(chunks) == 1:
            if not out or c != out[-1]:
                out.append(c)
    return out


def mattr(tokens: Sequence[str], window: int = 50) -> float:
    n = len(tokens)
    if n == 0:
        return float("nan")
    if n <= window:
        return len(set(tokens)) / n
    vals = [len(set(tokens[i:i+window])) / window for i in range(n - window + 1)]
    return float(np.mean(vals))


def document_statistics(docs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in docs.iterrows():
        text = str(r["text"])
        toks = tokenize(text)
        sents = split_sentences(text)
        unique = len(set(toks))
        rows.append({
            "doc_id": r["doc_id"],
            "source_file": r.get("source_file", ""),
            "language": r.get("language", "und"),
            "characters": len(text),
            "words": len(toks),
            "sentences": len(sents),
            "unique_words": unique,
            "type_token_ratio": unique / len(toks) if toks else np.nan,
            "mattr_50": mattr(toks, 50),
            "avg_sentence_words": len(toks) / len(sents) if sents else np.nan,
            "avg_word_length": float(np.mean([len(t) for t in toks])) if toks else np.nan,
            "hapax_share": (sum(1 for c in Counter(toks).values() if c == 1) / unique) if unique else np.nan,
        })
    return pd.DataFrame(rows)


def build_chunks(docs: pd.DataFrame, max_words: int, overlap_words: int) -> pd.DataFrame:
    rows = []
    meta_cols = [c for c in docs.columns if c not in {"text"}]
    for _, r in docs.iterrows():
        pieces = chunk_text(str(r["text"]), max_words, overlap_words)
        if not pieces and str(r["text"]).strip():
            pieces = [str(r["text"])]
        for i, piece in enumerate(pieces):
            row = {c: r[c] for c in meta_cols}
            row.update({
                "chunk_id": f"{r['doc_id']}::c{i+1}",
                "chunk_index": i + 1,
                "chunk_text": piece,
                "chunk_words": len(tokenize(piece)),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def top_tfidf_terms(cleaned_docs: Sequence[str], top_n: int = 30, max_features: int = 5000) -> pd.DataFrame:
    if not any(str(x).strip() for x in cleaned_docs):
        return pd.DataFrame(columns=["term", "mean_tfidf", "document_frequency"])
    try:
        v = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        X = v.fit_transform(cleaned_docs)
        names = v.get_feature_names_out()
        means = np.asarray(X.mean(axis=0)).ravel()
        dfs = np.asarray((X > 0).sum(axis=0)).ravel()
        idx = means.argsort()[::-1][:top_n]
        return pd.DataFrame({
            "term": names[idx],
            "mean_tfidf": means[idx],
            "document_frequency": dfs[idx],
        })
    except Exception:
        return pd.DataFrame(columns=["term", "mean_tfidf", "document_frequency"])


def collocations_pmi(cleaned_docs: Sequence[str], min_count: int = 3, top_n: int = 40) -> pd.DataFrame:
    unig = Counter()
    big = Counter()
    total_tokens = 0
    total_bigrams = 0
    for txt in cleaned_docs:
        toks = str(txt).split()
        unig.update(toks)
        total_tokens += len(toks)
        bs = list(zip(toks, toks[1:]))
        big.update(bs)
        total_bigrams += len(bs)
    rows = []
    if total_tokens == 0 or total_bigrams == 0:
        return pd.DataFrame(columns=["bigram", "count", "pmi"])
    for (a, b), c in big.items():
        if c < min_count:
            continue
        p_ab = c / total_bigrams
        p_a = unig[a] / total_tokens
        p_b = unig[b] / total_tokens
        pmi = math.log2(p_ab / (p_a * p_b)) if p_a > 0 and p_b > 0 else np.nan
        rows.append({"bigram": f"{a} {b}", "count": c, "pmi": pmi})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Require frequency, then rank by PMI to avoid one-off artefacts.
    return out.sort_values(["pmi", "count"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def topic_models(cleaned_docs: Sequence[str], n_topics: int, max_features: int, random_state: int = 42) -> Dict[str, object]:
    texts = [str(x) for x in cleaned_docs]
    n_units = len(texts)
    result: Dict[str, object] = {}
    if n_units < 2:
        return result
    n_topics_eff = max(2, min(int(n_topics), max(2, n_units)))
    # With many units, terms that occur in a single unit add noise and slow both models.
    min_df = 2 if n_units >= 30 else 1
    try:
        cv = CountVectorizer(max_features=max_features, min_df=min_df, max_df=0.95 if n_units >= 30 else 1.0)
        Xc = cv.fit_transform(texts)
        if Xc.shape[1] >= n_topics_eff:
            lda = LatentDirichletAllocation(
                n_components=n_topics_eff, max_iter=30, learning_method="batch", random_state=random_state
            )
            W = lda.fit_transform(Xc)
            names = cv.get_feature_names_out()
            rows = []
            for k, comp in enumerate(lda.components_):
                ix = comp.argsort()[::-1][:12]
                rows.append({"topic": k + 1, "top_terms": ", ".join(names[ix])})
            result["lda_topics"] = pd.DataFrame(rows)
            result["lda_doc_topic"] = pd.DataFrame(W, columns=[f"LDA_{i+1}" for i in range(W.shape[1])])
            result["lda_perplexity"] = float(lda.perplexity(Xc))
            result["lda_top_terms"] = [names[comp.argsort()[::-1][:15]].tolist() for comp in lda.components_]
    except Exception as e:
        result["lda_error"] = str(e)
    try:
        tv = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=0.95 if n_units >= 30 else 1.0, sublinear_tf=True)
        Xt = tv.fit_transform(texts)
        if Xt.shape[1] >= n_topics_eff:
            nmf = NMF(n_components=n_topics_eff, init="nndsvda", random_state=random_state, max_iter=1000, tol=1e-4)
            Wn = nmf.fit_transform(Xt)
            names = tv.get_feature_names_out()
            rows = []
            for k, comp in enumerate(nmf.components_):
                ix = comp.argsort()[::-1][:12]
                rows.append({"topic": k + 1, "top_terms": ", ".join(names[ix])})
            result["nmf_topics"] = pd.DataFrame(rows)
            result["nmf_doc_topic"] = pd.DataFrame(Wn, columns=[f"NMF_{i+1}" for i in range(Wn.shape[1])])
            result["nmf_reconstruction_error"] = float(nmf.reconstruction_err_)
            result["nmf_top_terms"] = [names[comp.argsort()[::-1][:15]].tolist() for comp in nmf.components_]
    except Exception as e:
        result["nmf_error"] = str(e)
    return result


def _n_init(n: int) -> int:
    return 20 if n <= 1000 else 5


def _silhouette(emb: np.ndarray, labels: np.ndarray, random_state: int = 42, max_n: int = 3000) -> float:
    """Cosine silhouette; estimated on a fixed random subsample for large corpora (O(n^2))."""
    n = len(emb)
    if n > max_n:
        return float(silhouette_score(emb, labels, metric="cosine", sample_size=max_n, random_state=random_state))
    return float(silhouette_score(emb, labels, metric="cosine"))


def choose_k(embeddings: np.ndarray, max_k: int = 8, random_state: int = 42) -> Tuple[int, pd.DataFrame]:
    n = len(embeddings)
    rows = []
    if n < 4:
        return (2 if n >= 2 else 1), pd.DataFrame()
    upper = min(max_k, n - 1)
    best_k, best_s = 2, -np.inf
    for k in range(2, upper + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=_n_init(n)).fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        s = _silhouette(embeddings, labels, random_state)
        db = davies_bouldin_score(embeddings, labels)
        rows.append({"k": k, "silhouette": float(s), "davies_bouldin": float(db)})
        if s > best_s:
            best_s, best_k = s, k
    return best_k, pd.DataFrame(rows)


def semantic_clustering(
    embeddings: np.ndarray, auto_k: bool = True, k_manual: int = 4, random_state: int = 42
) -> Dict[str, object]:
    n = len(embeddings)
    if n < 2:
        return {"labels": np.zeros(n, dtype=int), "coords": np.zeros((n, 2)), "k": 1}
    emb = normalize(np.asarray(embeddings))
    k_table = pd.DataFrame()
    if auto_k:
        k, k_table = choose_k(emb, random_state=random_state)
    else:
        k = max(2, min(int(k_manual), n - 1 if n > 2 else 2))
    if n <= k:
        k = max(1, n - 1)
    if k < 2:
        labels = np.zeros(n, dtype=int)
        sil = np.nan
        db = np.nan
    else:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=_n_init(n)).fit_predict(emb)
        sil = _silhouette(emb, labels, random_state) if len(set(labels)) > 1 and n > k else np.nan
        db = davies_bouldin_score(emb, labels) if len(set(labels)) > 1 else np.nan
    ncomp = min(2, n, emb.shape[1])
    coords0 = PCA(n_components=ncomp, random_state=random_state).fit_transform(emb)
    coords = coords0 if coords0.shape[1] == 2 else np.c_[coords0[:, 0], np.zeros(n)]
    return {
        "labels": labels,
        "coords": coords,
        "k": int(k),
        "silhouette": float(sil) if np.isfinite(sil) else np.nan,
        "davies_bouldin": float(db) if np.isfinite(db) else np.nan,
        "k_diagnostics": k_table,
    }


def aggregate_embeddings(chunk_embeddings: np.ndarray, chunks: pd.DataFrame, doc_ids: Sequence[str]) -> np.ndarray:
    arr = []
    for doc_id in doc_ids:
        idx = np.where(chunks["doc_id"].astype(str).values == str(doc_id))[0]
        arr.append(np.mean(chunk_embeddings[idx], axis=0) if len(idx) else np.zeros(chunk_embeddings.shape[1]))
    # Re-normalise: the mean of unit vectors is shorter than 1, so dot products of the raw
    # means are not cosine similarities.
    return normalize(np.vstack(arr))


def hierarchy_from_embeddings(embeddings: np.ndarray):
    if len(embeddings) < 2:
        return None
    # Average linkage is compatible with a precomputed cosine distance vector; Ward is not.
    dist = pdist(normalize(np.asarray(embeddings)), metric="cosine")
    return linkage(dist, method="average")


def frame_indicators(docs: pd.DataFrame, frame_dict: Dict[str, List[str]],
                     accent_insensitive: bool = True) -> pd.DataFrame:
    """Dictionary hits per document and per 1,000 tokens (wildcards ``*`` allowed)."""
    compiled = {frame: [p for p in (compile_term_pattern(t, accent_insensitive) for t in (terms or [])) if p is not None]
                for frame, terms in (frame_dict or {}).items()}
    rows = []
    for _, r in docs.iterrows():
        hay = _normalise_search_text(r["text"], False, accent_insensitive)
        n = max(len(tokenize(r["text"])), 1)
        row = {"doc_id": r["doc_id"]}
        for frame, pats in compiled.items():
            hits = sum(len(p.findall(hay)) for p in pats)
            row[f"frame_{frame}_count"] = hits
            row[f"frame_{frame}_per_1000"] = hits * 1000.0 / n
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["doc_id"])


def kwic(docs: pd.DataFrame, term: str, context_words: int = 12, max_hits: int = 100,
         accent_insensitive: bool = True) -> pd.DataFrame:
    """Keyword-in-context with whole-word, wildcard-aware, accent-insensitive matching.

    Matching runs on a normalised copy of the text; because NFKD accent stripping can change
    string length, matching is done token by token so the displayed context keeps the
    original spelling.
    """
    cols = ["doc_id", "left", "match", "right"]
    pat = compile_term_pattern(term, accent_insensitive)
    if pat is None:
        return pd.DataFrame(columns=cols)
    n_words = max(1, len(str(term).split()))
    rows = []
    for _, r in docs.iterrows():
        words = re.findall(r"\S+", str(r["text"]))
        norm = [_normalise_search_text(re.sub(r"^\W+|\W+$", "", w), False, accent_insensitive) for w in words]
        for i in range(len(words)):
            window = " ".join(norm[i:i + n_words])
            m = pat.match(window)
            if m and m.start() == 0 and (m.end() == len(window) or not window[m.end()].isalnum()):
                rows.append({
                    "doc_id": r["doc_id"],
                    "left": " ".join(words[max(0, i - context_words):i]),
                    "match": " ".join(words[i:i + n_words]),
                    "right": " ".join(words[i + n_words:i + n_words + context_words]),
                })
                if len(rows) >= max_hits:
                    return pd.DataFrame(rows, columns=cols)
    return pd.DataFrame(rows, columns=cols)


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    df = len(a) + len(b) - 2
    pooled = math.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / df) if df > 0 else np.nan
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0 if np.mean(a) == np.mean(b) else np.nan
    d = (np.mean(a) - np.mean(b)) / pooled
    J = 1 - 3 / (4 * df - 1) if df > 1 else 1
    return float(J * d)


def numeric_group_test(a: Sequence[float], b: Sequence[float]) -> pd.DataFrame:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return pd.DataFrame()
    welch = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    try:
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        mw_p = float(mw.pvalue)
    except Exception:
        mw_p = np.nan
    return pd.DataFrame([{
        "n_A": len(a), "n_B": len(b),
        "mean_A": float(np.mean(a)), "mean_B": float(np.mean(b)),
        "mean_difference_A_minus_B": float(np.mean(a)-np.mean(b)),
        "hedges_g": hedges_g(a, b),
        "welch_t": float(welch.statistic), "welch_p": float(welch.pvalue),
        "mann_whitney_p": mw_p,
    }])


def informative_log_odds(texts_a: Sequence[str], texts_b: Sequence[str], top_n: int = 40) -> pd.DataFrame:
    ca, cb = Counter(), Counter()
    for t in texts_a: ca.update(str(t).split())
    for t in texts_b: cb.update(str(t).split())
    vocab = sorted(set(ca) | set(cb))
    if not vocab:
        return pd.DataFrame(columns=["term", "z", "log_odds", "count_A", "count_B"])
    pooled = Counter(ca); pooled.update(cb)
    alpha0 = sum(pooled.values())
    na, nb = sum(ca.values()), sum(cb.values())
    rows = []
    for w in vocab:
        alpha = pooled[w]
        # Monroe et al.-style informative prior using pooled corpus counts.
        a_num = ca[w] + alpha
        b_num = cb[w] + alpha
        a_den = max((na + alpha0) - a_num, 1e-12)
        b_den = max((nb + alpha0) - b_num, 1e-12)
        delta = math.log(a_num / a_den) - math.log(b_num / b_den)
        var = 1.0 / a_num + 1.0 / b_num
        z = delta / math.sqrt(var)
        rows.append({"term": w, "z": z, "log_odds": delta, "count_A": ca[w], "count_B": cb[w]})
    out = pd.DataFrame(rows)
    out["abs_z"] = out["z"].abs()
    return out.sort_values("abs_z", ascending=False).head(top_n).drop(columns="abs_z").reset_index(drop=True)


def excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        used = set()
        for name, df in sheets.items():
            if df is None:
                continue
            safe = re.sub(r"[\\/*?:\[\]]", "_", str(name))[:31] or "Sheet"
            base = safe
            i = 2
            while safe in used:
                suffix = f"_{i}"
                safe = (base[:31-len(suffix)] + suffix)
                i += 1
            used.add(safe)
            df.to_excel(writer, sheet_name=safe, index=False)
    return bio.getvalue()


def json_safe(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.replace({np.nan: None}).to_dict(orient="records")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# v3.1 research utilities: bootstrap, temporal inference and supervised stance
# ---------------------------------------------------------------------------

def _boot_means(x: np.ndarray, n_boot: int, rng: np.random.Generator, block: int = 2_000_000) -> np.ndarray:
    """Vectorised bootstrap means, processed in blocks to bound memory."""
    n = len(x)
    out = np.empty(n_boot, dtype=float)
    step = max(1, block // max(n, 1))
    for start in range(0, n_boot, step):
        stop = min(n_boot, start + step)
        idx = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = x[idx].mean(axis=1)
    return out


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 2000, confidence: float = 0.95,
                      random_state: int = 42) -> Dict[str, float]:
    """Percentile bootstrap CI for a document-level mean."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if len(x) == 1:
        v = float(x[0])
        return {"n": 1, "mean": v, "ci_low": v, "ci_high": v}
    rng = np.random.default_rng(int(random_state))
    n_boot = max(200, int(n_boot))
    sims = _boot_means(x, n_boot, rng)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "ci_low": float(np.quantile(sims, alpha)),
        "ci_high": float(np.quantile(sims, 1.0 - alpha)),
    }


def bootstrap_difference_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 3000,
                            confidence: float = 0.95, random_state: int = 42) -> pd.DataFrame:
    """Bootstrap CI for mean(A)-mean(B), resampling documents within each group."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return pd.DataFrame()
    rng = np.random.default_rng(int(random_state))
    n_boot = max(500, int(n_boot))
    sims = _boot_means(a, n_boot, rng) - _boot_means(b, n_boot, rng)
    alpha = (1.0 - float(confidence)) / 2.0
    return pd.DataFrame([{
        "n_A": int(len(a)), "n_B": int(len(b)),
        "mean_A": float(np.mean(a)), "mean_B": float(np.mean(b)),
        "mean_difference_A_minus_B": float(np.mean(a) - np.mean(b)),
        "bootstrap_ci_low_95": float(np.quantile(sims, alpha)),
        "bootstrap_ci_high_95": float(np.quantile(sims, 1.0-alpha)),
        "bootstrap_replications": int(n_boot),
    }])


def infer_time_values(series: pd.Series) -> Tuple[pd.Series, str]:
    """Return a sortable time variable and a mode ('numeric' or 'datetime')."""
    s = series.copy()
    numeric = pd.to_numeric(s, errors="coerce")
    # Four-digit years and other mostly numeric time indexes are kept numeric.
    if numeric.notna().mean() >= 0.75:
        return numeric.astype(float), "numeric"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce", utc=False)
    if parsed.notna().mean() >= 0.60:
        return parsed, "datetime"
    return pd.Series(np.nan, index=s.index, dtype=float), "invalid"


def temporal_bootstrap_summary(df: pd.DataFrame, time_col: str, outcome_col: str,
                               group_col: Optional[str] = None, n_boot: int = 1500,
                               random_state: int = 42) -> pd.DataFrame:
    """Aggregate an outcome over time with percentile bootstrap CIs."""
    if time_col not in df or outcome_col not in df:
        return pd.DataFrame()
    work = df[[time_col, outcome_col] + ([group_col] if group_col and group_col in df else [])].copy()
    t, mode = infer_time_values(work[time_col])
    work["__time"] = t
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.dropna(subset=["__time", outcome_col])
    if work.empty or mode == "invalid":
        return pd.DataFrame()
    # Datetimes are grouped at monthly resolution when there is within-year variation,
    # otherwise at year resolution. Numeric indexes retain their observed values.
    if mode == "datetime":
        dt = pd.to_datetime(work["__time"])
        if dt.dt.to_period("M").nunique() > dt.dt.year.nunique():
            work["period"] = dt.dt.to_period("M").astype(str)
        else:
            work["period"] = dt.dt.year.astype(str)
    else:
        vals = work["__time"].astype(float)
        work["period"] = vals.map(lambda v: str(int(v)) if float(v).is_integer() else f"{v:g}")
    keys = ["period"] + ([group_col] if group_col and group_col in work else [])
    rows = []
    for key, sub in work.groupby(keys, dropna=False, sort=True):
        ci = bootstrap_mean_ci(sub[outcome_col].values, n_boot=n_boot, random_state=random_state)
        if not isinstance(key, tuple): key = (key,)
        rec = {"period": key[0], **ci}
        if len(keys) == 2: rec[group_col] = key[1]
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out


def robust_time_trend(df: pd.DataFrame, time_col: str, outcome_col: str,
                      group_col: Optional[str] = None) -> pd.DataFrame:
    """HC3 robust linear trend. Datetime coefficients are reported per year."""
    import statsmodels.api as sm
    cols = [time_col, outcome_col] + ([group_col] if group_col and group_col in df else [])
    work = df[cols].copy()
    t, mode = infer_time_values(work[time_col])
    y = pd.to_numeric(work[outcome_col], errors="coerce")
    if mode == "datetime":
        tt = pd.to_datetime(t)
        origin = tt.min()
        x = (tt - origin).dt.total_seconds() / (365.25 * 24 * 3600)
        unit = "year"
    elif mode == "numeric":
        x = pd.to_numeric(t, errors="coerce")
        unit = "time_unit"
    else:
        return pd.DataFrame()
    base = pd.DataFrame({"y": y, "time": x})
    if group_col and group_col in work:
        base[group_col] = work[group_col]
    rows = []
    groups = [("ALL", base)] if not group_col or group_col not in base else list(base.groupby(group_col, dropna=False))
    for g, sub in groups:
        sub = sub[["y", "time"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 4 or sub["time"].nunique() < 2:
            continue
        X = sm.add_constant(sub[["time"]], has_constant="add")
        fit = sm.OLS(sub["y"], X).fit(cov_type="HC3")
        ci = fit.conf_int().loc["time"]
        rows.append({
            "group": g, "n": int(fit.nobs), "time_unit": unit,
            "slope": float(fit.params["time"]), "robust_se_HC3": float(fit.bse["time"]),
            "t": float(fit.tvalues["time"]), "p_value": float(fit.pvalues["time"]),
            "ci_low_95": float(ci.iloc[0]), "ci_high_95": float(ci.iloc[1]),
            "r_squared": float(fit.rsquared),
        })
    return pd.DataFrame(rows)


def supervised_embedding_classifier(embeddings: np.ndarray, doc_ids: Sequence[str], labels: Sequence[object],
                                    random_state: int = 42) -> Dict[str, object]:
    """Research-oriented supervised classifier for user-coded stance/framing labels.

    Embeddings are fixed multilingual document representations. Evaluation uses stratified
    cross-validation on user-labelled documents only. The fitted model can then predict
    missing labels, but those predictions are not treated as ground truth.
    """
    emb = np.asarray(embeddings, dtype=float)
    ids = pd.Series(list(map(str, doc_ids)), dtype=str)
    lab = pd.Series(labels, dtype="object")
    valid = lab.notna() & lab.astype(str).str.strip().ne("")
    if valid.sum() < 4:
        return {"error": "At least four labelled documents are required."}
    y_raw = lab[valid].astype(str).values
    counts = pd.Series(y_raw).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return {"error": "At least two classes with at least two labelled documents per class are required."}
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    X = normalize(emb[valid.values])
    folds = int(min(5, counts.min()))
    if folds < 2:
        return {"error": "Insufficient labelled cases for cross-validation."}
    clf = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=int(random_state))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(random_state))
    pred_cv = cross_val_predict(clf, X, y, cv=cv, method="predict")
    report = classification_report(y, pred_cv, target_names=enc.classes_, output_dict=True, zero_division=0)
    metrics = pd.DataFrame([{
        "labelled_n": int(len(y)), "classes": int(len(enc.classes_)), "cv_folds": folds,
        "accuracy": float(accuracy_score(y, pred_cv)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred_cv)),
        "macro_f1": float(f1_score(y, pred_cv, average="macro")),
        "weighted_f1": float(f1_score(y, pred_cv, average="weighted")),
    }])
    cm = pd.DataFrame(confusion_matrix(y, pred_cv), index=enc.classes_, columns=enc.classes_).reset_index().rename(columns={"index":"actual"})
    rep_rows = []
    for k, v in report.items():
        if isinstance(v, dict):
            rep_rows.append({"class": k, **v})
    rep = pd.DataFrame(rep_rows)
    # Fit all labelled observations, then predict every document. This is deliberately
    # separate from CV metrics so in-sample predictions are never mistaken for validation.
    clf.fit(X, y)
    probs = clf.predict_proba(normalize(emb))
    pred = clf.predict(normalize(emb))
    pred_df = pd.DataFrame({"doc_id": ids, "predicted_label": enc.inverse_transform(pred), "was_labelled": valid.values})
    for i, cls in enumerate(enc.classes_):
        pred_df[f"p_{cls}"] = probs[:, i]
    pred_df["observed_label"] = lab.astype(object).values
    pred_df["prediction_confidence"] = probs.max(axis=1)
    return {
        "metrics": metrics, "confusion": cm, "classification_report": rep,
        "predictions": pred_df, "classes": list(enc.classes_), "cv_folds": folds,
    }


def aggregate_entities(entity_mentions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate NER mentions into canonical document/entity counts."""
    if entity_mentions is None or entity_mentions.empty:
        return pd.DataFrame(columns=["doc_id", "entity", "entity_type", "mentions", "mean_score"])
    e = entity_mentions.copy()
    e["entity"] = e["entity"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    e = e[e["entity"].str.len() >= 2]
    e["entity_key"] = e["entity"].str.casefold()
    rows = []
    for (doc_id, etype, key), sub in e.groupby(["doc_id", "entity_type", "entity_key"], dropna=False):
        surface = sub["entity"].value_counts().index[0]
        rows.append({"doc_id": str(doc_id), "entity": surface, "entity_type": str(etype),
                     "mentions": int(len(sub)), "mean_score": float(pd.to_numeric(sub["score"], errors="coerce").mean())})
    return pd.DataFrame(rows)


def actor_topic_edges(entities: pd.DataFrame, doc_topic: pd.DataFrame, doc_ids: Sequence[str],
                      entity_types: Optional[Sequence[str]] = None, top_entities: int = 30,
                      min_edge_weight: float = 0.05) -> pd.DataFrame:
    """Build weighted actor/entity ↔ topic edges from entity mentions and topic mixtures."""
    if entities is None or entities.empty or doc_topic is None or doc_topic.empty:
        return pd.DataFrame(columns=["entity", "entity_type", "topic", "weight", "documents"])
    ent = entities.copy()
    if entity_types:
        ent = ent[ent["entity_type"].isin(list(entity_types))]
    if ent.empty:
        return pd.DataFrame()
    totals = ent.groupby(["entity","entity_type"], as_index=False)["mentions"].sum().sort_values("mentions", ascending=False)
    keep = totals.head(max(3, int(top_entities)))[["entity","entity_type"]]
    ent = ent.merge(keep, on=["entity","entity_type"], how="inner")
    topic = doc_topic.copy().reset_index(drop=True)
    topic.insert(0, "doc_id", list(map(str, doc_ids))[:len(topic)])
    merged = ent.merge(topic, on="doc_id", how="inner")
    topic_cols = [c for c in topic.columns if c != "doc_id"]
    rows = []
    for (entity, etype), sub in merged.groupby(["entity","entity_type"]):
        for tc in topic_cols:
            weights = pd.to_numeric(sub[tc], errors="coerce").fillna(0) * pd.to_numeric(sub["mentions"], errors="coerce").fillna(0)
            w = float(weights.sum())
            if w >= float(min_edge_weight):
                rows.append({"entity": entity, "entity_type": etype, "topic": tc,
                             "weight": w, "documents": int(sub.loc[pd.to_numeric(sub[tc], errors="coerce").fillna(0) > 0, "doc_id"].nunique())})
    return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# v3.3: chunk-level topic models, coherence diagnostics and concept projection
# ---------------------------------------------------------------------------

def aggregate_unit_topics(unit_topic: pd.DataFrame, unit_doc_ids: Sequence[str], doc_ids: Sequence[str],
                          unit_weights: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Length-weighted mean of unit (chunk) topic mixtures per document, re-normalised to sum to 1.

    NMF weights are not probabilities; after row-normalisation they are read as each topic's
    share of the document's topical loading.
    """
    if unit_topic is None or unit_topic.empty:
        return pd.DataFrame()
    W = unit_topic.to_numpy(dtype=float)
    W = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
    w = np.ones(len(W)) if unit_weights is None else np.asarray(unit_weights, dtype=float)
    ids = np.asarray(list(map(str, unit_doc_ids)))
    rows = []
    for d in map(str, doc_ids):
        m = ids == d
        if not m.any() or w[m].sum() <= 0:
            rows.append(np.full(W.shape[1], np.nan))
            continue
        v = (W[m] * w[m, None]).sum(axis=0) / w[m].sum()
        rows.append(v / max(v.sum(), 1e-12))
    return pd.DataFrame(rows, columns=unit_topic.columns)


def topic_coherence_npmi(top_terms: Sequence[Sequence[str]], units: Sequence[str], top_n: int = 10) -> pd.DataFrame:
    """Mean pairwise NPMI of each topic's top terms, using unit-level co-occurrence.

    Computed on the same units used to fit the model (an intrinsic measure). Values range
    from -1 to 1; higher means the top terms co-occur more than chance.
    """
    tops = [[t for t in list(terms)[:top_n] if " " not in t] for terms in top_terms]
    vocab = sorted({t for ts in tops for t in ts})
    N = len(units)
    if N == 0 or not vocab:
        return pd.DataFrame({"topic": range(1, len(tops) + 1), "npmi": np.nan})
    pos = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((N, len(vocab)), dtype=np.float32)
    for r, u in enumerate(units):
        for w in set(str(u).split()):
            j = pos.get(w)
            if j is not None:
                X[r, j] = 1.0
    df_ = X.sum(axis=0)
    co = X.T @ X
    rows = []
    for k, terms in enumerate(tops):
        vals = []
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = pos[terms[i]], pos[terms[j]]
                da, db, dab = df_[a], df_[b], co[a, b]
                if da == 0 or db == 0:
                    continue
                if dab == 0:
                    vals.append(-1.0)
                    continue
                p_ab, p_a, p_b = dab / N, da / N, db / N
                vals.append(1.0 if p_ab >= 1.0 else math.log(p_ab / (p_a * p_b)) / -math.log(p_ab))
        rows.append({"topic": k + 1, "npmi": float(np.mean(vals)) if vals else np.nan})
    return pd.DataFrame(rows)


def topic_diversity(top_terms: Sequence[Sequence[str]], top_n: int = 25) -> float:
    """Share of unique words among all topics' top-n terms (Dieng et al., 2020)."""
    allw = [w for t in top_terms for w in list(t)[:top_n]]
    return float(len(set(allw)) / len(allw)) if allw else np.nan


def topic_k_diagnostics(units: Sequence[str], k_values: Sequence[int], max_features: int = 5000,
                        random_state: int = 42) -> pd.DataFrame:
    """Fit LDA and NMF for several k and report coherence (NPMI) and diversity."""
    rows = []
    for k in k_values:
        res = topic_models(units, int(k), max_features, random_state)
        for m in ["lda", "nmf"]:
            tt = res.get(f"{m}_top_terms")
            if not tt:
                continue
            coh = topic_coherence_npmi(tt, units, 10)
            rows.append({"model": m.upper(), "k": int(k), "mean_npmi": float(coh["npmi"].mean()),
                         "min_npmi": float(coh["npmi"].min()), "topic_diversity": topic_diversity(tt, 25)})
    return pd.DataFrame(rows)


def parse_concept_anchors(raw: object) -> Dict[str, List[str]]:
    """Parse concept anchors from JSON ({"concept": ["anchor text", ...]}) or from lines
    ``concept: anchor text`` (several lines may share a concept)."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        data = raw
    else:
        txt = str(raw).strip()
        if not txt:
            return {}
        if txt.startswith("{"):
            data = json.loads(txt)
        else:
            data = {}
            for line in txt.splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                if k.strip() and v.strip():
                    data.setdefault(k.strip(), []).append(v.strip())
    out = {}
    for k, v in data.items():
        vals = [v] if isinstance(v, str) else list(v or [])
        vals = [str(x).strip() for x in vals if str(x).strip()]
        if str(k).strip() and vals:
            out[str(k).strip()] = vals
    return out


def concept_projection(chunk_emb: np.ndarray, chunks: pd.DataFrame, doc_ids: Sequence[str],
                       concept_emb: Dict[str, np.ndarray], threshold: float = 0.45,
                       top_passages: int = 8) -> Dict[str, pd.DataFrame]:
    """Semantic projection of every chunk onto researcher-defined concept anchors.

    Each concept is the normalised centroid of its anchor embeddings. For every document the
    function reports the word-weighted mean cosine similarity of its chunks to each concept,
    the same score centred on the corpus mean (useful for comparing concepts whose raw
    similarity levels differ), and the share of the document's words in chunks whose closest
    concept is that one with similarity >= ``threshold``.
    """
    E = normalize(np.asarray(chunk_emb, dtype=float))
    names = list(concept_emb.keys())
    if not names or len(E) == 0:
        return {"documents": pd.DataFrame(), "passages": pd.DataFrame(), "chunks": pd.DataFrame()}
    C = normalize(np.vstack([np.mean(np.atleast_2d(concept_emb[n]), axis=0) for n in names]))
    S = E @ C.T
    words = pd.to_numeric(chunks.get("chunk_words", pd.Series(np.ones(len(E)))), errors="coerce").fillna(1).to_numpy()
    words = np.clip(words, 1, None)
    nearest = S.argmax(axis=1)
    assigned = S.max(axis=1) >= float(threshold)
    corpus_mean = (S * words[:, None]).sum(axis=0) / words.sum()
    ids = chunks["doc_id"].astype(str).to_numpy()
    rows = []
    for d in map(str, doc_ids):
        m = ids == d
        rec = {"doc_id": d}
        for j, n in enumerate(names):
            if not m.any():
                rec[f"concept_{n}_mean"] = np.nan
                rec[f"concept_{n}_centred"] = np.nan
                rec[f"concept_{n}_share"] = np.nan
                continue
            w = words[m]
            mean = float((S[m, j] * w).sum() / w.sum())
            rec[f"concept_{n}_mean"] = mean
            rec[f"concept_{n}_centred"] = mean - float(corpus_mean[j])
            rec[f"concept_{n}_share"] = float(w[(nearest[m] == j) & assigned[m]].sum() / w.sum())
        rows.append(rec)
    chunk_tab = chunks[["chunk_id", "doc_id"]].copy().reset_index(drop=True)
    for j, n in enumerate(names):
        chunk_tab[f"sim_{n}"] = S[:, j]
    chunk_tab["nearest_concept"] = [names[i] if a else "" for i, a in zip(nearest, assigned)]
    pass_rows = []
    text = chunks["chunk_text"].astype(str).to_numpy()
    cids = chunks["chunk_id"].astype(str).to_numpy()
    for j, n in enumerate(names):
        for i in np.argsort(S[:, j])[::-1][:int(top_passages)]:
            pass_rows.append({"concept": n, "similarity": float(S[i, j]), "doc_id": ids[i],
                              "chunk_id": cids[i], "passage": text[i]})
    return {"documents": pd.DataFrame(rows), "passages": pd.DataFrame(pass_rows), "chunks": chunk_tab}
