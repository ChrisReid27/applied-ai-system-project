from src.rag import (
    build_grounded_explanation,
    detect_cluster_warning,
    discovery_mode_requested,
    infer_profile_from_text,
    load_corpus,
    retrieve_docs,
    retrieve_song_grounding_docs,
    select_bridge_recommendation,
)
from src.recommender import load_songs, recommend_songs


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


def test_hybrid_retriever_matches_workout_synonyms():
    docs = load_corpus()
    results = retrieve_docs("gym run hype mix", docs, k=2)

    assert len(results) == 2
    assert any("Workout" in doc.title for doc in results)


def test_hybrid_retriever_matches_reggaeton_party_context():
    docs = load_corpus()
    results = retrieve_docs("latin club reggaeton party", docs, k=2)

    assert len(results) == 2
    assert any("Reggaeton" in doc.title for doc in results)


def test_build_grounded_explanation_mentions_sources():
    docs = load_corpus()[:2]
    explanation = build_grounded_explanation(
        song={"title": "Any"},
        score=4.2,
        reasons=["genre match", "mood match"],
        retrieved_docs=docs,
    )

    assert "Grounded by" in explanation


def test_retrieve_song_grounding_docs_uses_song_specific_tags():
    docs = load_corpus()
    edm_docs = retrieve_song_grounding_docs(
        "i want some edm",
        {"title": "Voltage Bloom", "genre": "edm", "mood": "euphoric"},
        docs,
        k=2,
    )
    lofi_docs = retrieve_song_grounding_docs(
        "i want some edm",
        {"title": "Focus Flow", "genre": "lofi", "mood": "focused"},
        docs,
        k=2,
    )

    assert edm_docs[0].title != lofi_docs[0].title
    assert any("EDM" in doc.title or "Workout" in doc.title for doc in edm_docs)
    assert any("Focus" in doc.title or "Lofi" in doc.title or "Calm" in doc.title for doc in lofi_docs)


def test_discovery_mode_requested_recognizes_opt_in_language():
    assert discovery_mode_requested("surprise me with discovery mode")
    assert discovery_mode_requested("take me somewhere new")
    assert not discovery_mode_requested("i want some edm")


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


def test_infer_profile_from_text_handles_new_genre_and_mood_keywords():
    songs = [{"genre": "pop", "mood": "happy"}]
    prefs, _ = infer_profile_from_text("Need a reggaeton party mix for tonight", songs)

    assert prefs["genre"] == "reggaeton"
    assert prefs["mood"] == "party"
    assert float(prefs["danceability"]) >= 0.85


def test_infer_profile_from_text_treats_edm_as_high_energy_dance_music():
    songs = load_songs("data/songs.csv")
    prefs, _ = infer_profile_from_text("i want some edm", songs)
    recs = recommend_songs(prefs, songs, k=3)

    assert prefs["genre"] == "edm"
    assert prefs["mood"] in {"euphoric", "energetic"}
    assert float(prefs["energy"]) >= 0.9
    assert float(prefs["danceability"]) >= 0.85
    assert recs[0][0]["genre"] == "edm"
    assert recs[0][0]["title"] in {"Voltage Bloom", "365", "Blessings", "Golden"}
