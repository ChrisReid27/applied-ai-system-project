from src.recommender import load_songs, recommend_songs


def test_recommend_returns_songs_sorted_by_score():
    """Verify that recommendations are ranked by score."""
    songs = load_songs("data/songs.csv")
    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "acousticness": 0.2,
    }
    results = recommend_songs(user_prefs, songs, k=2)

    assert len(results) == 2
    # Verify tuple structure: (song_dict, score, reasons_list)
    song_1, score_1, reasons_1 = results[0]
    assert isinstance(song_1, dict)
    assert isinstance(score_1, float)
    assert isinstance(reasons_1, list)
    # First result should score higher than second
    assert score_1 >= results[1][1]


def test_explain_recommendation_returns_non_empty_string():
    """Verify that recommendations have valid scoring data."""
    songs = load_songs("data/songs.csv")
    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "acousticness": 0.2,
    }
    results = recommend_songs(user_prefs, songs, k=1)
    
    song, score, reasons = results[0]
    # Verify scoring structure (score and reasons are used by RAG system)
    assert score > 0.0
    assert isinstance(reasons, list)
    assert len(reasons) > 0

