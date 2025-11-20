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

In our results, we got the exact same accuracy result. This seems very suspicious -- however, we are using a very small test set of 38 songs, with 13 being Indian so it seems plausible that this result could happen, especially since per model our confusion matrix is different. Additionally, with such a small set we can't be sure that the actual classifcation is trained on the features we intend or rather just some noise specific to this small sample group.

---

## Ideas for final report (e.g., new features, different algorithms, hyperparameter tuning).
![Feature importance](data/feature%20importance.png)
We extracted the most important features for the random forest model. We could try using these fields only to train some models. In addition, we could try to tune some hyperparameters such as limiting the depth of the trees more to ensure we don't overfit our models. However, it is kind of hard to see if are actually classifying based on any identifiable feature well. A good initial step we will take before hyperparameter tuning is trying our classifer on English songs/American pop and see if we can visually identify any trends and see if our classifier is interesting at all before going into finetuning. We could also try implementing cross fold validation because this model may only be training on this limited dataset and learn the noise, as mentioned above.
