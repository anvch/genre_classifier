# **International Pop Genre Classification + Extension to American Pop**

**Angela Chen, Pragati Toppo, Samiksha Karimbil**
---

## **1. Introduction**

Automated music genre classification has become increasingly relevant as streaming platforms, recommendation systems, and digital archives handle millions of audio tracks. Traditional approaches rely on human tagging or metadata, which is often inconsistent and subjective. Machine learning, paired with modern audio feature extraction, provides a scalable and data-driven pathway for robust genre prediction.

This project aims to build and evaluate a machine-learning system capable of classifying audio tracks into multiple genres. We compare classical machine-learning models, gradient-boosting methods, and ensemble methods to understand their strengths and limitations when trained on a small, imbalanced dataset.

Our analysis highlights the challenges of genre classification when working with limited and skewed data, and demonstrates that—while some models show promising learning behavior—overfitting and dataset bias significantly impact generalization performance. We propose several improvements for future iterations, including targeted feature selection, cross-validation, and dataset expansion.

---

## **2. Dataset and Preprocessing**

### **2.1 Dataset Overview + Feature Extraction**

Our dataset consists of music across 4 languages. After preprocessing (combining lyrics and metadata from our Genius lyrics and Spotify metadata datasets), we obtained 62 Indian, 50 Korean, 50 Chinese, and 24 Chinese songs. This is a lot less data than we initially expected (with thousands of songs in the original datasets), but it makes sense considering the difficulty of joining on title/artist that may have slight differences between the datasets. We attempted to alleviate this difference by normalizing the text and making a join key, yet this is our result. This imbalance is a central challenge throughout the project.

At this point, the dataset includes title, tag (genre), artist, year, views, features, lyrics, id, language, region_group, join_key, popularity, duration_ms, acousticness, explicit, energy, danceability, key, loudnes, mode, speechiness, instrumentalness, liveness, valence, tempo.

After creating this combined dataset, we cleaned the lyrics to remove text such as '[CHORUS]' and then translated all of the lyrics to English to avoid data leakage. We then extracted features from the lyrics such as word_count, unique_words, repetition_ratio, lexical_diversity, sentiment_polarity, and embedding (of the lyrics). We chose to extract these features because we were curious whether there was a common sentiment or theme among songs of certain genre, as well as patterns in the range of vocabulary used.


### **2.2 Train/Test Split**

To evaluate model generalization, we used an 80/20 split, with stratification to preserve genre proportions as well as possible given the imbalance. Our true label of the genre was region_group. We decided not to include columns such as title, artist, year, views, features, lyrics, id, language, join_key, popularity, explicit, and clean lyrics because we either thought they were irrelevant or had little relevance, or because we thought that having those would be too obvious as too what genre it is (i.e. a Chinese song having a Chinese title, or BTS having a K-pop song). In most of the models, we also chose to just use the numerical fields as it was easier to develop compared to including categorical fields as well.

---

## **3. Models and Methods**

We trained several models to compare classical, ensemble, and gradient-boosting approaches:

* **Baseline Dummy Classifier**
  Predicts the most frequent genre (Indian).
* **Support Vector Machine**
* **Random Forest**
* **XGBoost Classifier**

For each model, we computed:

* Accuracy
* Precision, Recall, F1-score
* Confusion matrix
* ROC and PR curves 

Hyperparameters were initially left at defaults to establish a comparison baseline.

---

## **4. Baseline Model Performance**

Our baseline predicts the majority genre, Indian, for every input. Despite its simplicity, it provides an important lower bound for evaluating real models.

### **Strengths**

* Establishes a minimum viable benchmark
* Trains instantly
* Useful for sanity-checking model performance

### **Weaknesses**

* Uses no meaningful features
* Offers no interpretability
* Provides 100% recall for Indian and 0% recall for all other genres
* Performance collapses if classes are balanced

Because Indian songs comprise a large portion of the dataset, the baseline achieves deceptively high accuracy.

---

## **5. Results and Observed Behavior**

### **5.1 Identical Accuracy Concern**

Across several trained models we observed identical accuracy values, which initially raised concerns. However, this is explainable given:

* A small test set of 38 songs
* 13/38 songs (34%) are of an Indian language
* Models sometimes overfitting to the majority genre
* Different confusion matrices but identical overall accuracy

Small datasets create high variance in accuracy metrics, making it difficult to distinguish model quality.

### **5.2 Random Forest and XGBoost Behavior**

We observed several noteworthy patterns:

* **Random Forest achieved perfect AUC (1.0)**, strongly suggesting overfitting.
  With so few training examples, deep trees may memorize noise or playlist-specific patterns.

* **XGBoost achieved high but not perfect AUC**, indicating it is actually learning discriminative structure rather than memorizing.

* Both models performed inconsistently across genres due to sample count differences.

### **5.3 Feature Importance Analysis**

Random Forest feature importance (Figure 1) revealed:

### **5.4 ROC and Precision-Recall Curves**

Both ROC and PR curves reveal clear differences in model behavior:

* XGBoost maintains reasonable performance across threshold values
* Random Forest saturates at AUC=1 early, a hallmark of memorization
* PR curves show instability for minority genres, confirming class imbalance issues

---

## **6. Limitations and Sources of Bias**

Several factors significantly impact model reliability:

### **6.1 Dataset Imbalance**

The dominance of Indian songs causes models to:

* Favor predicting the Indian class
* Struggle to learn minority genres
* Exhibit inflated accuracy despite poor real-world performance

### **6.2 Small Training and Test Sets**

With only 38 test samples:

* Accuracy fluctuates unpredictably
* Minor changes in predictions produce large swings
* Overfitting becomes extremely easy

---

## **7. Future Work**

Based on our findings, we propose several improvements for the next iteration:

- Dataset Expansion: A larger, more diverse dataset would drastically improve model stability. Adding English/American pop songs is a planned next step.
- Feature Selection - Using only the top N features from Random Forest importance values may reduce noise and improve generalization.
- Hyperparameter Tuning:
Promising directions include limiting Random Forest depth, reducing the number of estimators, adjusting XGBoost learning rate and max_depth, grid search for SVM C and gamma values
---

## **8. Conclusion**

This project explores the challenges and possibilities of machine-learning-based music genre classification using a small, imbalanced dataset. Our analysis shows:

* Gradient-boosting models such as XGBoost learn meaningful structure but are constrained by data limitations.
* Ensemble methods like Random Forest risk severe overfitting without depth constraints.
* Dataset imbalance dominates evaluation metrics, making accuracy a misleading measure.
* Visualization through ROC, PR curves, and feature importance provides essential evidence of model behavior.

Despite these limitations, our system demonstrates early signs of effective genre classification and establishes a foundation for more sophisticated models. Future work in dataset expansion, regularization, and cross-validation will allow us to evaluate model performance more reliably and build a classifier that generalizes to all languages.

---
