# Changelog

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
