# PerspectiveMapper v3.1 — Research Edition

PerspectiveMapper is a Streamlit application for multilingual and reproducible corpus, discourse, framing and semantic analysis. Version 3.1 adds inferential, temporal, supervised and network-analysis modules intended for research workflows rather than demo-only NLP.

## Main capabilities

### Corpus and multilingual processing
- TXT, DOCX, text-PDF, CSV, XLSX/XLS.
- One row = one document for structured corpora, with ID and researcher-selected metadata.
- Interface in Spanish, English and Italian.
- Automatic document language detection.
- Configurable chunking with overlap for long documents.

### Lexical and topic analysis
- TF-IDF distinctive terms.
- PMI bigram collocations.
- TTR, MATTR, hapax share, sentence length and document statistics.
- LDA and NMF topic models with document-topic mixtures and diagnostics.
- KWIC concordances for contextual inspection.

### Semantic analysis
- Multilingual sentence embeddings with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- K-means in the full normalized embedding space; PCA is used only for display.
- Automatic k selection by silhouette, with Davies–Bouldin diagnostics.
- Average-linkage hierarchical clustering on cosine distance.
- Document similarity matrix and semantic passage search.

### Sentiment
- Chunk-level multilingual three-class sentiment.
- Document-level aggregation of P(positive), P(neutral), P(negative).
- Continuous score = P(positive) - P(negative).
- The app explicitly treats sentiment output as a model-based measurement, not ground truth.

### Framing
- Editable multilingual framing dictionaries.
- Counts and rates per 1,000 tokens.
- Default frames include security/threat, rights/humanitarian, economic contribution, integration/belonging, health/vulnerability and climate/environment.
- Dictionary framing is explicitly separated from claims about bias, discrimination or intention.

## New in v3.1

### 1. Entities and actors
Optional multilingual NER using `Davlan/xlm-roberta-base-ner-hrl`.

- Extracts and aggregates entities by document and type.
- Top persons/organisations/locations table.
- Actor/entity ↔ topic weighted bipartite networks.
- Network edge table available for export.
- NER is optional because the additional transformer model is relatively heavy and is downloaded on first use.

### 2. Supervised stance / supervised framing
The supervised module **does not generate its own training labels**.

Add a manually coded metadata column such as `stance_manual` or `frame_manual`. The app then:

- uses multilingual document embeddings as predictors;
- fits class-balanced multinomial/binary logistic regression;
- evaluates it using stratified cross-validation;
- reports accuracy, balanced accuracy, macro-F1 and weighted-F1;
- shows a confusion matrix and per-class precision/recall/F1;
- fits the final model on all labelled documents and returns predicted probabilities for all documents.

Cross-validation metrics are kept conceptually separate from final fitted predictions to reduce the risk of reporting in-sample fit as predictive performance.

### 3. Temporal analysis
If metadata contain `year`, `date`, `wave`, `month` or another usable temporal variable:

- mean outcomes by period;
- percentile bootstrap 95% confidence intervals;
- optional stratification by a grouping variable;
- HC3-robust linear time trends;
- support for sentiment, framing rates, topic weights and other numeric document-level outcomes.

Temporal association does not establish causality.

### 4. Better group inference
Two-group comparisons include:

- informative-prior log-odds keyness;
- Welch t-test;
- Mann–Whitney U;
- Hedges' g;
- percentile-bootstrap 95% CI for the mean difference.

The numerical outcome is selectable rather than being restricted to sentiment.

### 5. Econometric models
Document-level OLS with HC3 robust standard errors can use generated NLP variables as outcomes and uploaded metadata as predictors.

This is useful for exploratory or publication workflows, but it does not correct measurement error in NLP-generated variables and does not replace panel, multilevel or clustered models when the corpus design requires them.

### 6. Publication-ready output
The publication tab can create:

- 300-dpi PNG figures;
- vector SVG versions;
- CSV analysis tables;
- methodology table;
- HTML research report;
- a ZIP package containing the complete publication output.

The main Excel export also includes available temporal, bootstrap, entity, network and supervised-classification results.

## Recommended structured corpus

A research CSV/XLSX can contain columns such as:

```text
id,text,country,source,date,year,group,stance_manual,frame_manual
```

Only `text` is required. `id` and metadata are selected from the sidebar after upload.

`sample_research_corpus.csv` is included to demonstrate temporal, group and supervised workflows.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first semantic, sentiment or NER run downloads the corresponding Hugging Face model and caches it locally.

## Streamlit Community Cloud

Use `app.py` as the entry point. No secret is required. To enable the optional login, create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`.

For limited-memory deployments, leave NER disabled unless actor/entity analysis is needed.

## Research cautions

1. **Unit of inference:** significance tests and bootstrap resampling operate at document level. If documents are repeated, clustered, nested or longitudinal, use a design-appropriate model.
2. **Cross-language lexical analysis:** TF-IDF, LDA, NMF and dictionaries operate on surface forms. Multilingual embeddings provide the main cross-language semantic representation.
3. **Supervised classification:** predictive quality depends on the quality, consistency and representativeness of the researcher's coded labels.
4. **NER:** aliases, tokenisation errors, homonyms and entity linking are not automatically resolved; networks require manual validation.
5. **Sentiment/framing:** model or dictionary outputs are measurements and should be validated for the substantive domain.
6. **Causal interpretation:** temporal trends, associations and OLS coefficients are not automatically causal effects.

## Version

PerspectiveMapper v3.1 Research Edition.
