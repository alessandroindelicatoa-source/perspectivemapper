import numpy as np
import pandas as pd
import pytest

import analysis_core as ac


def docs_df(texts, **meta):
    df = pd.DataFrame({"doc_id": [f"d{i}" for i in range(len(texts))], "text": texts, "language": "es"})
    for k, v in meta.items():
        df[k] = v
    return df


# --- PDF cleaning -----------------------------------------------------------

def test_clean_extracted_text_dehyphenates_and_rejoins_lines():
    raw = "La doctrina social de la Igle-\nsia enseña que el\ntrabajo tiene dignidad.\n\n12\n\nNuevo párrafo."
    out = ac.clean_extracted_text(raw)
    assert "Iglesia" in out
    assert "que el trabajo tiene" in out
    assert "\n12\n" not in out
    assert out.endswith("Nuevo párrafo.")


def test_repeated_headers_are_removed():
    body = ["El trabajo humano es la clave.", "La propiedad tiene función social.", "El Estado debe tutelar a los débiles.",
            "El salario debe ser suficiente.", "La caridad no sustituye a la justicia.", "Los obreros pueden asociarse.",
            "El capital necesita del trabajo."]
    pages = [f"Carta Encíclica Rerum Novarum\n{b}\n{b.upper()}\n{i}"
             for i, b in enumerate(body, 1)]
    rep = ac.find_repeated_page_lines(pages)
    out = ac.clean_extracted_text("\n".join(pages), rep)
    assert "Rerum Novarum" not in out
    assert "El Estado debe tutelar" in out


def test_strip_notes_cuts_final_notes_section_only():
    body = "Primer párrafo con una cita [1]. " * 40
    raw = body + "\n\nNOTAS\n\n[1] León XIII, Rerum novarum."
    out = ac.clean_extracted_text(raw, strip_notes=True)
    assert "León XIII" not in out
    assert "[1]" not in out
    assert "Primer párrafo" in out


# --- Dictionaries, KWIC, filtering ------------------------------------------

def test_wildcard_and_accent_insensitive_dictionary():
    d = docs_df(["El salario justo y los salarios mínimos.", "La economía y la ECONOMIA política."])
    fr = ac.frame_indicators(d, {"wage": ["salari*"], "econ": ["economía"]})
    assert fr.loc[0, "frame_wage_count"] == 2
    assert fr.loc[1, "frame_econ_count"] == 2
    # whole-word: "economía" must not match inside "macroeconomía"
    fr2 = ac.frame_indicators(docs_df(["macroeconomía"]), {"econ": ["economía"]})
    assert fr2.loc[0, "frame_econ_count"] == 0


def test_multiword_dictionary_entry_tolerates_line_breaks():
    d = docs_df(["la propiedad\n privada es legítima"])
    fr = ac.frame_indicators(d, {"prop": ["propiedad privada"]})
    assert fr.loc[0, "frame_prop_count"] == 1


def test_dictionary_term_counts_audit():
    d = docs_df(["salario y salarios", "nada"])
    audit = ac.dictionary_term_counts(d, {"wage": ["salari*"]})
    assert len(audit) == 1 and audit.iloc[0]["hits"] == 2
    assert "salarios" in audit.iloc[0]["matched_forms"]


def test_kwic_keeps_original_spelling_and_whole_words():
    d = docs_df(["El Capital y el capitalismo; también el capital humano."])
    k = ac.kwic(d, "capital", 3)
    assert list(k["match"]) == ["Capital", "capital"]
    k2 = ac.kwic(d, "capital*", 3)
    assert len(k2) == 3


def test_keyword_filter_any_all():
    d = docs_df(["paz y guerra", "solo paz", "nada"])
    out, audit = ac.filter_documents_by_keywords(d, ["paz", "guerra"], mode="ALL")
    assert out["doc_id"].tolist() == ["d0"]
    out, _ = ac.filter_documents_by_keywords(d, ["paz", "guerra"], mode="ANY")
    assert out["doc_id"].tolist() == ["d0", "d1"]


# --- Lexical ----------------------------------------------------------------

def test_lemmatization_merges_inflections():
    pytest.importorskip("simplemma")
    out = ac.clean_for_lexical("Los trabajadores y el trabajador reclaman salarios", "es", lemmatize=True)
    toks = out.split()
    assert toks.count("trabajador") == 2
    assert "salario" in toks


def test_extra_stopwords_removed():
    out = ac.clean_for_lexical("Dios y la Iglesia enseñan", "es", extra_stop=["dios", "iglesia"])
    assert "dios" not in out and "iglesia" not in out


# --- Topics -----------------------------------------------------------------

def test_aggregate_unit_topics_weighted_and_normalised():
    ut = pd.DataFrame({"T1": [1.0, 0.0, 0.5], "T2": [0.0, 1.0, 0.5]})
    agg = ac.aggregate_unit_topics(ut, ["a", "a", "b"], ["a", "b", "c"], unit_weights=[3, 1, 1])
    assert np.allclose(agg.iloc[0].values, [0.75, 0.25])
    assert np.allclose(agg.iloc[1].values, [0.5, 0.5])
    assert agg.iloc[2].isna().all()


def test_npmi_coherence_orders_topics():
    units = ["mercado precio oferta"] * 10 + ["salario obrero trabajo"] * 10
    good = [["mercado", "precio", "oferta"]]
    bad = [["mercado", "salario", "obrero"]]
    g = ac.topic_coherence_npmi(good, units)["npmi"].iloc[0]
    b = ac.topic_coherence_npmi(bad, units)["npmi"].iloc[0]
    assert g > b


def test_topic_k_diagnostics_runs():
    rng = np.random.default_rng(0)
    vocab_a = "mercado precio oferta demanda competencia".split()
    vocab_b = "salario obrero trabajo justicia familia".split()
    units = [" ".join(rng.choice(vocab_a if i % 2 else vocab_b, 20)) for i in range(40)]
    diag = ac.topic_k_diagnostics(units, [2, 3], max_features=100)
    assert set(diag["model"]) == {"LDA", "NMF"}
    assert diag["topic_diversity"].between(0, 1).all()


# --- Semantic ---------------------------------------------------------------

def test_aggregate_embeddings_are_unit_length():
    rng = np.random.default_rng(1)
    emb = ac.normalize(rng.normal(size=(6, 8)))
    chunks = pd.DataFrame({"doc_id": ["a", "a", "a", "b", "b", "c"]})
    doc = ac.aggregate_embeddings(emb, chunks, ["a", "b", "c"])
    assert np.allclose(np.linalg.norm(doc, axis=1), 1.0)


def test_concept_projection_assigns_nearest_concept():
    chunk_emb = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    chunks = pd.DataFrame({"chunk_id": list("wxyz"), "doc_id": ["a", "a", "b", "b"],
                           "chunk_text": list("WXYZ"), "chunk_words": [100, 100, 100, 100]})
    concepts = {"market": np.array([[1, 0, 0]]), "labour": np.array([[0, 1, 0]])}
    out = ac.concept_projection(chunk_emb, chunks, ["a", "b"], concepts, threshold=0.5)
    d = out["documents"].set_index("doc_id")
    assert d.loc["a", "concept_market_share"] == 1.0
    assert d.loc["b", "concept_labour_share"] == 0.5
    assert d.loc["a", "concept_market_mean"] > d.loc["b", "concept_market_mean"]
    assert out["passages"].query("concept == 'labour'").iloc[0]["chunk_id"] == "y"


def test_parse_concept_anchors_formats():
    assert ac.parse_concept_anchors('{"a": "x", "b": ["y", "z"]}') == {"a": ["x"], "b": ["y", "z"]}
    assert ac.parse_concept_anchors("mercado: libre competencia\nmercado: precios\nmal") == {"mercado": ["libre competencia", "precios"]}


# --- Inference --------------------------------------------------------------

def test_bootstrap_ci_contains_mean_and_is_reproducible():
    x = np.arange(30, dtype=float)
    a = ac.bootstrap_mean_ci(x, n_boot=1000, random_state=3)
    b = ac.bootstrap_mean_ci(x, n_boot=1000, random_state=3)
    assert a == b
    assert a["ci_low"] < a["mean"] < a["ci_high"]


def test_bootstrap_difference_sign():
    out = ac.bootstrap_difference_ci(np.ones(20) + np.arange(20) * 0.01, np.zeros(20) + np.arange(20) * 0.01)
    assert out["bootstrap_ci_low_95"].iloc[0] > 0.9


def test_hedges_g_known_value():
    g = ac.hedges_g([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert g == pytest.approx(-0.5711, abs=1e-3)


def test_semantic_clustering_large_uses_sampling():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc=c, size=(1600, 5)) for c in (-5, 5)])
    res = ac.semantic_clustering(X, auto_k=True, random_state=0)
    assert res["k"] >= 2 and np.isfinite(res["silhouette"])
