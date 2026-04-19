from src.rag import build_grounded_explanation, infer_profile_from_text, retrieve_docs, load_corpus


def test_infer_profile_from_text_extracts_genre_and_defaults():
    songs = [{"genre": "pop", "mood": "happy"}]
    prefs, docs = infer_profile_from_text("upbeat pop workout music", songs)

    assert prefs["genre"] == "pop"
    assert prefs["energy"] >= 0.8
    assert len(docs) > 0


def test_retrieve_docs_returns_ranked_documents():
    docs = load_corpus()
    results = retrieve_docs("focus lofi study", docs, k=2)

    assert len(results) == 2
    assert any("Focus" in doc.title or "Late Night" in doc.title for doc in results)


def test_build_grounded_explanation_mentions_sources():
    docs = load_corpus()[:2]
    explanation = build_grounded_explanation(
        song={"title": "Any"},
        score=4.2,
        reasons=["genre match", "mood match"],
        retrieved_docs=docs,
    )

    assert "Grounded by" in explanation
