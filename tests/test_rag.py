from src.rag import (
    build_grounded_explanation,
    detect_cluster_warning,
    infer_profile_from_text,
    load_corpus,
    retrieve_docs,
    select_bridge_recommendation,
)


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


def test_select_bridge_recommendation_returns_outside_cluster_song():
    songs = [
        {"id": 1, "title": "One", "genre": "pop", "mood": "happy", "energy": 0.9, "tempo_bpm": 120, "valence": 0.8, "danceability": 0.8, "acousticness": 0.2},
        {"id": 2, "title": "Two", "genre": "pop", "mood": "happy", "energy": 0.85, "tempo_bpm": 118, "valence": 0.78, "danceability": 0.82, "acousticness": 0.18},
        {"id": 3, "title": "Bridge", "genre": "lofi", "mood": "chill", "energy": 0.55, "tempo_bpm": 84, "valence": 0.6, "danceability": 0.58, "acousticness": 0.75},
    ]
    recommendations = [
        (songs[0], 7.5, ["genre match"]),
        (songs[1], 7.1, ["genre match"]),
    ]

    cluster_warning = detect_cluster_warning(recommendations)
    bridge_pick = select_bridge_recommendation(
        {"genre": "pop", "mood": "happy", "energy": 0.9},
        songs,
        recommendations,
    )

    assert cluster_warning is not None
    assert bridge_pick is not None
    bridge_song, _, bridge_reasons = bridge_pick
    assert bridge_song["genre"] != "pop"
    assert any("bridge from" in reason for reason in bridge_reasons)
