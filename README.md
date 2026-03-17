# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Explain your design in plain language.

Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
- What information does your `UserProfile` store
- How does your `Recommender` compute a score for each song
- How do you choose which songs to recommend

You can include a simple diagram or bullet list if helpful.

Songs:
Songs in my system uses genre and mood as categories and numeric values will include energy, tempo/bpm, valence, danceability, and acousticness. Other secondary features will include the song title and song artist which won't be used in scoring. Mood and genre will be higher weighted than the numeric value categories. Every song in the catalog stores the dat like this:

| Feature | Type | Description |
|---|---|---|
| `genre` | categorical | Broad style label (pop, lofi, rock, etc.) |
| `mood` | categorical | Emotional tone (chill, intense, happy, etc.) |
| `energy` | numeric 0–1 | How energetic or calm the track feels |
| `tempo_bpm` | numeric | Beats per minute |
| `valence` | numeric 0–1 | Musical positivity (high = upbeat, low = somber) |
| `danceability` | numeric 0–1 | How suitable the track is for dancing |
| `acousticness` | numeric 0–1 | How acoustic vs. electronic the track sounds |

---

User profile:
The UserProfile will store the listener's target preferences for every scored feature:

- `preferred_genre` — the genre they want to hear
- `preferred_mood` — the mood they are looking for right now
- `preferred_energy` — their ideal energy level (0.0 to 1.0)
- `preferred_tempo_bpm` — their ideal tempo in beats per minute
- `preferred_valence` — how positive or somber they want the music
- `preferred_danceability` — how danceable they want the track
- `preferred_acousticness` — how acoustic vs. produced they prefer

---

Scoring Rule for Recommender:
The scoring rule for my recommender will compute a similarity score between 0 and 1 for each song.

**Categorical features** (genre, mood):
- Score = `1.0` if the song matches the user preference, `0.0` otherwise.

**Numeric features** (energy, valence, danceability, acousticness):
```
similarity = 1 - |song_value - user_value|
```

**Tempo** (scaled by catalog range):
```
tempo_score = 1 - (|song_tempo - user_tempo| / (max_tempo - min_tempo))
```

All scores are clamped to the range `[0, 1]`.

**Final weighted score:**
```
score = 0.20 × genre
      + 0.25 × mood
      + 0.15 × energy
      + 0.15 × tempo
      + 0.10 × valence
      + 0.10 × danceability
      + 0.05 × acousticness
```

Mood receives the highest weight because it best reflects the listener's current intent (focus, chill, intense). Genre receives the second highest weight as a strong taste signal. Numeric features fine-tune the ranking within matching mood and genre groups.

---

Ranking rule (the list):
After scoring all songs, the recommender:
1. Sorts songs from highest score to lowest score
2. Returns the top N results (default: 5)
3. Excludes songs already in the user's history (if provided)

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

