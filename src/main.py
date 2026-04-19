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
    infer_profile_from_text,
    select_bridge_recommendation,
    retrieve_docs,
)
from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    print("Music recommender ready.")
    print("Type any request and press Enter. Type 'quit' to exit.\n")

    while True:
        user_text = input("What do you want to hear? ").strip()
        if user_text.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        if not user_text:
            print("Please enter a request or type 'quit' to exit.\n")
            continue

        user_prefs, profile_docs = infer_profile_from_text(user_text, songs)

        print(f"\n{'='*50}")
        print(f"User text: {user_text}")
        print(f"Inferred profile: {user_prefs}")
        print(f"{'='*50}\n")

        recommendations = recommend_songs(user_prefs, songs, k=5)
        cluster_warning = detect_cluster_warning(recommendations)
        bridge_pick = select_bridge_recommendation(user_prefs, songs, recommendations) if cluster_warning else None

        print("Top recommendations:\n")
        for song, score, reasons in recommendations:
            query = f"{song['genre']} {song['mood']} {user_text}"
            docs = retrieve_docs(query, profile_docs, k=2)
            explanation = build_grounded_explanation(song, score, reasons, docs)

            print(f"  {song['title']} - Score: {score:.2f}")
            print(f"  Because: {explanation}")
            print()

        if cluster_warning:
            print(cluster_warning)

        if bridge_pick:
            bridge_song, bridge_score, bridge_reasons = bridge_pick
            bridge_docs = retrieve_docs(
                f"{bridge_song['genre']} {bridge_song['mood']} {user_text}",
                profile_docs,
                k=2,
            )
            bridge_explanation = build_grounded_explanation(
                bridge_song,
                bridge_score,
                bridge_reasons,
                bridge_docs,
            )
            print("\nDiscovery mode pick:")
            print(f"  {bridge_song['title']} - Score: {bridge_score:.2f}")
            print(f"  Because: {bridge_explanation}")

        print()


if __name__ == "__main__":
    main()
