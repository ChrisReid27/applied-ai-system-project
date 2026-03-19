# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**Music Taste Discerner (MTD) Version 1.0**  

---

## 2. Intended Use    

MTD is a class project meant to match songs to user music taste via their user profiles and explore how recommenders work. It assumes the user has knowledge on their tastes for genre, preferred mood, whether they like acoutics or not, etc.

---

## 3. How the Model Works  

Avoid code here. Pretend you are explaining the idea to a friend who does not program. Features included for scoring is genre, mood, energy, tempo/bpm, valence, and if you can dance to the song or not. User's preference for acoustics is considered but is also a number value for songs, too. Each feature has a weight from 0-1 that gets put in an math algorithm to calculate the best songs to recommend users with the highest getting listed first. From starter logic to now I changed a lot, mostly changing how songs get scored based on the features listed before. I also added in explanations that show how your top song got recommended to you and why others were ranked lower.

---

## 4. Data  

There's 18 songs in MTD's catalog. Genre's include pop, lofi, ambient, rock, jazz, synthwave, metal, reggae, classical, hip hop, country, edm, folk, r&b, and indie pop. I added in 8 songs to the original 10. Acoustic music is barely in my catalog.

---

## 5. Strengths  

MTD works reliably when the genre and mood match the user's tastes. It is good at distinguishing between energized music and more calm music. Users who know what they like will be fine using MTD. Mood weighting captures what listeners want to listen to in the moment since vibes can change every day.

---

## 6. Limitations and Bias 

One weakness I found when the energy weight got doubled from 1.5 to 3.0 in the experiment was that the system became significantly more likely to group users into energy-based "bubbles." Users with extreme energy preferences (like preferring 0.2 or 0.9 on a 0-1 scale) will find it nearly impossible to discover songs outside their range, since a mismatch of 0.7 energy points now costs them 2.0 scoring points which is difficult to overcome even with a perfect genre and mood match. This reveals that the system over-prioritizes energy similarity in a way that locks users into narrow recommendation bands (which happened with genre before the shift), particularly disadvantaging low-energy and acoustic music listeners (only 2 songs with energy <=0.3 vs. 11 songs with energy >=0.7).

---

## 7. Evaluation  

I checked if the recommender was behaving as expected by running three different listener profiles and checking whether the top songs felt reasonable to me. One profile was a high-energy pop listener who wanted very energetic music and didn't like acoustic songs. In that case, Night Drive Loop was their top song before and after weight shifting even though the scores were different (Before: 4.26, After: 5.54) because it has very high energy, and after my weight-shift experiment energy became one of the strongest parts of the score. MTD started mainly focusing on energy similarity more than before, so songs with matching energy were pushed up even when other traits were not perfect matches. This helped me confirm the model was working as coded, but also that the ranking can become repetitive for users with extreme energy preferences.

---

## 8. Future Work  

I would add more songs and genres and add more acoustic songs specifically since it's so absent from the catalog right now. I could also generate more formatted sentences for why a song got recommended or not instead of just showing scoring process values. Overweighting could be fine tuned out of MTD so that too. Maybe relax the genre-mood matching so their's more specificity to the song recs for users.

---

## 9. Personal Reflection  
  
Something major I learned during this process is that the valuing of features is the most important part and the balncing of them. Overweighting a certain stat was skewing the system throughout the project, basically the main issue to be honest. A.I. helped greatly, it helped to plan how song score would be calculated and helped me clean up my functions during the part where you had to make them pythonic and organized. This project made me understand Spotify a lot more and also how they would get the data for things like Wrapped, which every app like this does now. Most of these systems change even more if the user actually listens to a song through their system, while this project just does the recs part.