# PerspectiveMapper v3.3 — Doctrine & Concepts Edition

PerspectiveMapper is a Streamlit application for multilingual, reproducible corpus, discourse, semantic and quantitative text analysis. Version 3.3 fixes several defects of v3.2, makes long-document corpora (such as papal encyclicals) faster and cleaner to process, and adds a concept-projection module for comparing a corpus with positions taken from another body of thought, for example economic theory.

## What is new in v3.3

### Text preparation
- **PDF/DOCX cleaning** (on by default): removes page numbers and running headers/footers (detected as lines that recur at the top or bottom of at least 40% of pages), rejoins words hyphenated at line ends and rebuilds paragraphs broken by hard wraps.
- **Optional removal of the final notes section** (`NOTAS`, `NOTES`, `NOTE`…) and of `[n]` note markers, designed for vatican.va PDFs.
- **Lemmatisation** with the `simplemma` dictionary lemmatiser for lexical analysis and topic models (`trabajadores` → `trabajador`). Embeddings, sentiment and dictionaries use the original text.
- **Researcher-defined stopwords** from the sidebar.
- Uploaded files are parsed once and cached, so changing a widget no longer re-reads every PDF.

### Topic models
- Topics can be estimated on **non-overlapping passages** and aggregated to documents as word-weighted means. `Automatic` uses passages when there are fewer than 30 documents or the median document exceeds 1,500 words.
- Each topic now reports **NPMI coherence** (top-10 terms, co-occurrence within modelling units) and each model reports **topic diversity** (share of unique words among the top-25 terms per topic).
- A **k-diagnostics** panel fits LDA and NMF over a range of k and plots coherence and diversity.
- The topic slider now goes up to 30.

### Dictionaries and concordance
- Entries support a **within-word wildcard** (`salari*`, `*capital*`) and multi-word expressions that tolerate line breaks.
- Matching is whole-word, case-insensitive and optionally accent-insensitive.
- A **dictionary audit** lists the exact forms each entry matched, so false positives can be spotted before the indicators are used.
- KWIC uses the same matching rules and keeps the original spelling in the output.

### Concept projection (new tab)
Define concepts with one or more anchor sentences, for example:

```text
free_market: free competition in the market allocates resources efficiently
just_wage: wages must suffice for the decent livelihood of the worker and his family
universal_destination: the goods of the earth are destined for all people
```

Each concept is the centroid of its anchor embeddings. For every passage the app computes cosine similarity to each concept and, per document, reports:

- `concept_<name>_mean`: word-weighted mean similarity;
- `concept_<name>_centred`: the same mean minus the corpus mean for that concept, so concepts with different baseline similarity can be compared;
- `concept_<name>_share`: share of the document's words in passages whose nearest concept is this one with similarity above the chosen threshold.

The closest passages to every concept are listed for validation. The three measures become outcomes in the grouped, temporal, two-group and regression tabs. Anchors can be written in a different language from the corpus because the embedding model is multilingual. This is a measure of semantic proximity, not of doctrinal agreement.

### Fixes
- Document similarity is now a true cosine similarity: v3.2 averaged unit-length passage embeddings without re-normalising them, which shrank similarities for documents with heterogeneous passages.
- Sentiment document scores are word-weighted, so a short trailing passage no longer counts as much as a full one.
- Silhouette scores are estimated on a fixed 3,000-unit subsample for large corpora (the exact computation is quadratic in the number of passages).
- Bootstrap resampling is vectorised.
- Deprecated `use_container_width` arguments replaced by `width="stretch"` (Streamlit ≥ 1.51).
- `.streamlit/runtime.txt` contained the literal text `3.10\n` and was removed; `package.json`, a duplicate `config.toml`, the v2 app and two obsolete documents were removed too.
- CSV/TXT files encoded in Windows-1252 are read correctly.
- The Excel workbook is built on request instead of on every rerun.

### Security
- `.streamlit/secrets.toml` is no longer part of the project and is listed in `.gitignore`.
- Passwords are compared in constant time and can be stored as `sha256:<hex>` hashes (see `.streamlit/secrets.toml.example`).

## Features carried over from v3.2

### 1. Long Excel documents are reconstructed automatically
Excel cells are limited to 32,767 characters, so long documents are often split across columns such as:

```text
texto, texto_2, texto_3, ... texto_10
```

PerspectiveMapper now detects these sequential text-column families and proposes all of them automatically. The selected columns are concatenated **in natural order before chunking or analysis**.

This preserves the correct unit of analysis:

> one spreadsheet row = one document

The sidebar shows the detected text columns and a preview of the number of rows and concatenated characters.

### 2. Global keyword filtering
Before analysis, the corpus can be filtered by one or more words or phrases.

- `ANY`: keep a document when at least one requested expression appears.
- `ALL`: keep a document only when all requested expressions appear.
- Search scope: full text, metadata, or both.
- Matching is case-insensitive and can ignore accents/diacritics.
- Phrases can be entered directly and terms are separated by commas, semicolons or line breaks.
- The original document remains the statistical unit; matching passages are **not** converted into separate observations.
- An audit table records which terms matched each document and the total number of matches.

Example:

```text
pace, guerra, giustizia sociale
```

### 3. Global result grouping
Results can be aggregated by any suitable metadata variable, for example:

```text
papa
country
source
year
group
```

When a `papa` column exists, it is proposed automatically. The new **Grouped results** tab provides:

- number of documents and words by group;
- means, medians and standard deviations of numeric outcomes;
- document-level percentile-bootstrap 95% confidence intervals;
- distributions by group;
- LDA/NMF topic prevalence by group;
- heatmaps of group-topic mixtures;
- distinctive vocabulary for one group versus the rest using informative-prior log-odds.

Topic and sentiment tabs also respect the selected grouping variable where relevant.

### 4. No preloaded “bias” dictionary
The custom dictionary field is now **blank by default**. PerspectiveMapper does not preload categories such as gender, political party, religion, migration, or other normative categories.

Dictionary analysis runs only when the researcher explicitly supplies a JSON dictionary, for example:

```json
{
  "peace": ["peace", "paz", "pace"],
  "war": ["war", "guerra"]
}
```

These results are described as lexicometric dictionary/framing indicators. They are not automatic proof of bias, discrimination, intention or causality.

## Core research capabilities

### Corpus and multilingual processing
- TXT, DOCX, text-PDF, CSV, XLSX/XLS.
- One row = one document for structured corpora.
- Multiple text columns can be concatenated before analysis.
- Researcher-selected IDs and metadata.
- Interface in Spanish, English and Italian.
- Automatic language detection.
- Configurable chunking with overlap for long documents.

### Lexical analysis
- TF-IDF terms.
- PMI bigram collocations.
- TTR, MATTR, hapax share, sentence length and document statistics.
- KWIC concordances.
- Informative-prior log-odds keyness for group comparisons.

### Topic modelling
- LDA.
- NMF.
- Document-topic mixtures.
- Perplexity / reconstruction diagnostics.
- Topic prevalence aggregated by metadata group.
- Passage-level estimation, NPMI coherence, topic diversity and k diagnostics (v3.3).

### Semantic analysis
- Multilingual embeddings with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- K-means in the full normalised embedding space.
- PCA only for visualisation.
- Automatic k selection by silhouette.
- Davies–Bouldin diagnostics.
- Average-linkage hierarchical clustering using cosine distance.
- Document similarity matrix.
- Semantic passage search.

### Sentiment
- Chunk-level multilingual sentiment.
- Document-level aggregation of P(positive), P(neutral), P(negative).
- Continuous score = P(positive) − P(negative).
- Grouped means and bootstrap confidence intervals when a result group is selected.

### Entities and actor-topic networks
Optional multilingual NER using `Davlan/xlm-roberta-base-ner-hrl`.

- Persons, organisations, locations and other entity types.
- Aggregation by document.
- Actor/entity ↔ topic weighted bipartite networks.
- Exportable network edge table.

NER is optional because it downloads an additional transformer model and may be memory-intensive on Streamlit Community Cloud.

### Supervised stance / framing
The app does **not** invent training labels. Supply a manually coded metadata column such as `stance_manual` or `frame_manual`.

PerspectiveMapper then:

- uses multilingual document embeddings as predictors;
- fits class-balanced logistic regression;
- evaluates with stratified cross-validation;
- reports accuracy, balanced accuracy, macro-F1 and weighted-F1;
- shows confusion matrices and per-class metrics;
- returns predicted class probabilities.

### Temporal analysis
With year/date/wave metadata:

- mean outcomes by period;
- percentile-bootstrap 95% confidence intervals;
- optional stratification by another metadata variable;
- HC3-robust linear time trends.

### Group inference
Two-group comparison includes:

- informative-prior log-odds keyness;
- Welch t-test;
- Mann–Whitney U;
- Hedges' g;
- percentile-bootstrap 95% CI for the difference in means.

### Econometric models
Document-level OLS with HC3 robust standard errors can use NLP-generated indicators as outcomes and metadata as predictors. Use parsimonious specifications for small corpora and design-appropriate multilevel/panel models when observations are nested or repeated.

### Publication-ready output
The app can create:

- 300-dpi PNG figures;
- vector SVG figures;
- CSV tables;
- methodology table;
- HTML research report;
- ZIP publication package;
- multi-sheet Excel workbook;
- JSON results.

The Excel export also records the keyword-filter audit, grouped summaries, custom dictionary results and available inferential outputs.

## Example: papal encyclicals
A spreadsheet may contain:

```text
papa
nombre_enciclica
anio
texto
texto_2
...
texto_10
archivo_fuente
```

Recommended setup:

- document ID: `nombre_enciclica` or `archivo_fuente`;
- text: `texto` + `texto_2` + ... + `texto_10` (auto-detected);
- metadata: `papa`, `anio`, `archivo_fuente`;
- result grouping: `papa`.

This allows topic, sentiment, lexical and temporal results to be analysed both by individual encyclical and by pontificate.

Suggested workflow for comparing encyclicals with economic theory:

1. Load the PDFs or the spreadsheet, keep text cleaning on and, for PDFs, remove the notes section if the notes are not part of the analysis.
2. Turn lemmatisation on and add corpus-specific stopwords (forms of address, citation abbreviations).
3. Leave the topic unit on `Automatic` (passages) and use the k diagnostics to choose a small set of candidate k values; then read the topics.
4. Write a dictionary with wildcards for the economic vocabulary and check the audit table.
5. Write anchor sentences for each theoretical position in the Concepts tab, read the closest passages, revise the anchors and only then use the concept measures by pope or year.
6. Report the methodology table exported with the results.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the tests:

```bash
pip install pytest
pytest -q
```

The first semantic, sentiment or NER run downloads the corresponding Hugging Face model and caches it locally.

## Streamlit Community Cloud

Use `app.py` as the entry point (`app_perspectivemapper.py` also works: it only launches `app.py`, for deployments created with that file). No secret is required. To enable optional login, create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`.

For limited-memory deployments, leave NER disabled unless actor/entity analysis is needed.

## Research cautions

1. **Unit of inference:** bootstrap and significance tests operate at document level.
2. **Filtering:** keyword filters define a substantive subsample and should be reported in the methods section when used.
3. **Grouping:** aggregated means can hide within-group heterogeneity; inspect document-level distributions as well.
4. **Cross-language lexical analysis:** lexical models use surface forms; multilingual embeddings are the main cross-language semantic representation.
5. **Supervised classification:** validity depends on researcher coding quality and representativeness.
6. **Sentiment/dictionaries:** model and dictionary outputs are measurements, not ground truth.
7. **Causal interpretation:** temporal trends, OLS coefficients and group differences are not automatically causal effects.

8. **Concept projection:** similarity to anchor sentences depends on how the anchors are worded; report them in full and show validation passages.
9. **Topic coherence:** NPMI is computed on the same units used to fit the model; it helps compare k values, it does not validate topic labels.

## Version

PerspectiveMapper v3.3 Doctrine & Concepts Edition.
