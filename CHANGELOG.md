# Changelog

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
- Updated Spanish, English and Italian interface strings.
- NER is optional and disabled by default to keep resource use manageable on Streamlit deployments.
