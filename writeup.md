# First draft of genre classifier 

Angela Chen, Pragati Toppo, Samiksha Karimbil

---

## Strengths and weaknesses of the baseline
Our baseline classifier currently uses dummy classification to classify every song to be the most popular genre in our dataset, which is currently Indian music.

### Strengths:
- It gives a valid base performance for us to make sure our model always performs better. If our model barely beats the baseline, we know to rethink our approach.
- It trains basically instantly as there is no computational logic.

### Weaknesses:
- It ignores all the features and simply looks at the most frequent genre.
- The model provides no insight and does not explain which features matter.
- Since there is more Indian music, it predicts that 100% of the time, therefore giving us 100% recall for India and 0% for the others.
- If the dataset is evenly split between genres, then the accuracy would be equal for all of them.

---

## Possible reasons for errors or bias

We are using a very small sample set of 38 songs, with 13 being Indian. We 

---

## Ideas for final report (e.g., new features, different algorithms, hyperparameter tuning).

![Feature importance](data/feature%20importance.png)