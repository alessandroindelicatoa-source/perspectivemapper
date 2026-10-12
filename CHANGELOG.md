# Changelog

## v3.3 Doctrine & Concepts Edition

- Added PDF/DOCX text cleaning: page numbers, running headers/footers, line-end hyphenation and hard line wraps.
- Added optional removal of the final notes section and `[n]` markers.
- Added `simplemma` lemmatisation for lexical analysis and topic models.
- Added researcher-defined stopwords.
- Added passage-level topic estimation with word-weighted aggregation to documents (`Automatic`, `Passages`, `Documents`).
- Added NPMI coherence per topic, topic diversity per model and a k-diagnostics panel.
- Raised the maximum number of topics from 12 to 30.
- Added wildcard (`*`), accent-insensitive and line-break-tolerant dictionary matching, plus a dictionary audit table.
- KWIC now uses whole-word, wildcard-aware matching and keeps the original spelling.
- Added the Concepts tab: semantic projection of passages onto researcher-defined anchor sentences, with mean, centred and share measures, validation passages, and integration into grouped, temporal, comparison and regression tabs.
- Fixed document similarity, which was not a cosine similarity because averaged embeddings were not re-normalised.
- Sentiment document scores are now word-weighted.
- Silhouette scores use a fixed subsample above 3,000 units; bootstrap resampling is vectorised.
- Uploaded files are parsed once and cached.
- Added a progress bar for the main analysis.
- Excel export is built on request.
- Replaced deprecated `use_container_width` with `width="stretch"`; requires Streamlit ≥ 1.51.
- CPU-only PyTorch index in `requirements.txt`.
- Removed `secrets.toml`, `runtime.txt`, `package.json`, the duplicate root `config.toml`, the v2 app and obsolete documents; added `.gitignore`.
- Constant-time password comparison and optional `sha256:` password hashes.
- Added unit tests for `analysis_core.py`.
- New outputs added to the Excel, JSON, HTML report and publication package.

## v3.2 Corpus & Grouping Edition

- Added automatic detection of split long-text families such as `texto`, `texto_2`, …, `texto_10`.
- Added natural-order concatenation of selected text columns before chunking and all NLP analyses.
- Added a structured-data preview showing selected text columns, rows with text and concatenated characters.
- Added global keyword/phrase filtering with ANY/ALL logic.
- Added text, metadata and combined filter scopes.
- Added case-insensitive and optional accent/diacritic-insensitive matching.
- Added keyword-filter audit data and export.
- Added global “Group results by” selector with automatic preference for `papa` when available.
- Added a dedicated Grouped results tab.
- Added grouped document/word counts.
- Added grouped numeric summaries with document-level bootstrap 95% confidence intervals.
- Added grouped outcome distributions.
- Added LDA/NMF topic prevalence by group and group-topic heatmaps.
- Added distinctive vocabulary for a selected group versus the remainder of the corpus.
- Added group-aware topic and sentiment visualisations.
- Removed every preloaded custom/bias/framing dictionary category; the dictionary input is blank by default.
- Custom dictionary analysis now runs only when the researcher explicitly provides JSON terms.
- Added grouped custom-dictionary indicators.
- Added keyword, grouped and custom-dictionary outputs to Excel/JSON exports.
- Integrated the new PerspectiveMapper logo into the app assets.

## v3.1 Research Edition

- Added optional multilingual named-entity recognition and actor/entity aggregation.
- Added actor/entity ↔ topic weighted network analysis.
- Added supervised stance/framing classification using researcher-coded labels, multilingual embeddings, class-balanced logistic regression and stratified cross-validation.
- Added confusion matrices, per-class metrics and document-level class probabilities.
- Added temporal summaries with percentile-bootstrap 95% confidence intervals.
- Added HC3-robust temporal trend estimation.
- Generalised two-group numerical comparisons beyond sentiment.
- Added bootstrap confidence intervals for mean differences.
- Added publication-ready ZIP export with 300-dpi PNG, SVG, CSV tables, methodology and HTML report.
- Expanded Excel and JSON exports with new analysis outputs.
- Added `sample_research_corpus.csv` with metadata for group, temporal and supervised workflows.
