# PerspectiveMapper v3.2 — Corpus & Grouping Edition

PerspectiveMapper is a Streamlit application for multilingual, reproducible corpus, discourse, semantic and quantitative text analysis. Version 3.2 keeps the research modules from v3.1 and adds a substantially improved corpus-ingestion layer, global keyword filtering and result aggregation by metadata groups.

## What is new in v3.2

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

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first semantic, sentiment or NER run downloads the corresponding Hugging Face model and caches it locally.

## Streamlit Community Cloud

Use `app.py` as the entry point. No secret is required. To enable optional login, create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`.

For limited-memory deployments, leave NER disabled unless actor/entity analysis is needed.

## Research cautions

1. **Unit of inference:** bootstrap and significance tests operate at document level.
2. **Filtering:** keyword filters define a substantive subsample and should be reported in the methods section when used.
3. **Grouping:** aggregated means can hide within-group heterogeneity; inspect document-level distributions as well.
4. **Cross-language lexical analysis:** lexical models use surface forms; multilingual embeddings are the main cross-language semantic representation.
5. **Supervised classification:** validity depends on researcher coding quality and representativeness.
6. **Sentiment/dictionaries:** model and dictionary outputs are measurements, not ground truth.
7. **Causal interpretation:** temporal trends, OLS coefficients and group differences are not automatically causal effects.

## Version

PerspectiveMapper v3.2 Corpus & Grouping Edition.
