"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from .rag import (
    build_grounded_explanation,
    detect_cluster_warning,
    discovery_mode_requested,
    infer_profile_from_text,
    load_corpus,
    retrieve_song_grounding_docs,
    select_bridge_recommendation,
)
from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    corpus = load_corpus()

    print("Music recommender ready.")
    print("Type what you want to hear and press Enter. Type 'quit' to exit.\n")

    while True:
        user_text = input("What do you want to hear? ").strip()
        if user_text.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        if not user_text:
            print("Please enter a request or type 'quit' to exit.\n")
            continue

        user_prefs, profile_docs = infer_profile_from_text(user_text, songs)
        discovery_requested = discovery_mode_requested(user_text)

        if user_prefs.get("needs_clarification"):
            print("\nI need a little more to work with. Try naming a genre, a mood, or ask for discovery mode.")
            print()
            continue

        print(f"\n{'='*50}")
        genre = str(user_prefs.get("genre", "music"))
        mood = str(user_prefs.get("mood", ""))
        if user_prefs.get("mode") == "discovery" and not user_prefs.get("genre"):
            print("I heard you want to explore a few options.")
        elif mood:
            print(f"I heard you want {genre} with a {mood} feel.")
        else:
            print(f"I heard you want {genre}.")
        if discovery_requested:
            print("Discovery mode is on. I’ll include one exploratory pick if it helps.")
        print("Here are the best matches I found:\n")
        print(f"{'='*50}\n")

        recommendations = recommend_songs(user_prefs, songs, k=5)
        cluster_warning = detect_cluster_warning(recommendations)
        bridge_pick = (
            select_bridge_recommendation(user_prefs, songs, recommendations)
            if discovery_requested
            else None
        )

        for index, (song, score, reasons) in enumerate(recommendations, start=1):
            docs = retrieve_song_grounding_docs(user_text, song, corpus, k=2)
            explanation = build_grounded_explanation(song, score, reasons, docs)

            print(f"{index}. {song['title']} by {song['artist']}")
            print(f"   {explanation}")
            print()

        if cluster_warning:
            print(cluster_warning)

        if bridge_pick:
            bridge_song, bridge_score, bridge_reasons = bridge_pick
            bridge_docs = retrieve_song_grounding_docs(user_text, bridge_song, corpus, k=2)
            bridge_explanation = build_grounded_explanation(
                bridge_song,
                bridge_score,
                bridge_reasons,
                bridge_docs,
            )
            print("\nDiscovery mode pick:")
            print(f"{bridge_song['title']} by {bridge_song['artist']}")
            print(f"   {bridge_explanation}")

        print()


if __name__ == "__main__":
    main()
