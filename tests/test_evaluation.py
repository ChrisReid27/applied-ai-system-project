from src.evaluation import compare_baseline_vs_rag, evaluate_baseline_stability, evaluate_rag_explanations
from src.recommender import load_songs


def test_baseline_stability_keeps_deterministic_rankings():
    songs = load_songs("data/songs.csv")
    first_run = evaluate_baseline_stability(songs)
    second_run = evaluate_baseline_stability(songs)

    assert first_run == second_run
    assert set(first_run.keys()) == {"Workout Pop Session", "Late Night Focus", "Moody Drive"}


def test_rag_explanations_are_grounded_for_all_prompts():
    songs = load_songs("data/songs.csv")
    explanations = evaluate_rag_explanations(songs)

    assert set(explanations.keys()) == {"Workout Pop Session", "Late Night Focus", "Moody Drive"}
    for prompt_explanations in explanations.values():
        assert len(prompt_explanations) == 5
        assert all("Grounded by" in explanation for explanation in prompt_explanations)


def test_baseline_vs_rag_report_contains_expected_fields():
    songs = load_songs("data/songs.csv")
    report = compare_baseline_vs_rag(songs)

    assert set(report.keys()) == {"Workout Pop Session", "Late Night Focus", "Moody Drive"}
    for prompt_name, payload in report.items():
        assert "user_text" in payload
        assert "profile" in payload
        assert "baseline_titles" in payload
        assert "rag_explanations" in payload
        assert len(payload["baseline_titles"]) == 5
        assert len(payload["rag_explanations"]) == 5
        assert prompt_name in {"Workout Pop Session", "Late Night Focus", "Moody Drive"}
