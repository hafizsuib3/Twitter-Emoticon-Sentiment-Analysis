![Twitter Sentiment Analysis](TwitterBanner.png)

## Table of Contents
- [About Project](#about-project)  
- [Background Overview](#background-overview)  
- [Problem Statement](#problem-statement)  
- [Objective](#objective)  
- [Built With](#built-with)  
- [Data Source](#data-source)  
- [Methodology](#methodology)  
- [Result and Impact](#result-and-impact)  
- [Challenges and Solutions](#challenges-and-solutions)  
- [Sneak Peek of the App](#sneak-peek-of-the-app)

---

## About Project  

This project trains and evaluates multiple **machine learning models** for sentiment classification of tweets, using the **Sentiment140 dataset** where labels are automatically derived from emoticons (positive `:)`, negative `:(`).  

It includes:  
- A training pipeline that preprocesses tweets, extracts features with **TF-IDF**, and benchmarks multiple models.  
- Automatic selection and saving of the **best-performing model** (`sentiment_model.pkl`) alongside the **TF-IDF vectorizer** (`tfidf.pkl`).  
- A framework ready to be extended into a **Streamlit web app** for real-time sentiment prediction.  

---

## Background Overview  

Social media platforms like Twitter produce massive amounts of opinionated text daily. Analyzing this sentiment helps businesses, researchers, and policymakers understand trends in public mood.  

Traditional manual annotation of tweets is slow and expensive. The **Sentiment140 dataset** provides a scalable alternative by labeling tweets based on emoticons. Although noisy, it allows training robust baseline sentiment classifiers.  

This project builds a fast and interpretable machine-learning pipeline to automatically classify tweets as **positive** or **negative**.  

---

## Problem Statement  

While sentiment analysis is powerful, practical challenges remain:  
- **Data Volume**: Millions of tweets cannot be manually labeled or reviewed.  
- **Noise**: Tweets contain slang, hashtags, mentions, and URLs that complicate text analysis.  
- **Consistency**: Manual labeling may differ across annotators, making emoticon-based labeling a useful but imperfect proxy.  
- **Accessibility**: Many available models are not user-friendly for non-technical stakeholders.  

---

## Objective  

The project aims to:  
1. **Preprocess and clean raw tweets** into a usable text representation.  
2. **Compare multiple ML models** on sentiment classification using TF-IDF features.  
3. **Identify the best-performing model** using weighted F1-score.  
4. **Save artifacts** for deployment (model + TF-IDF vectorizer).  

Ultimately, the goal is to provide a **reproducible and deployable sentiment analysis baseline** that can later be extended into an application.  

---

## Built With  

- **Python** (pandas, numpy, scikit-learn, joblib, nltk)  
- **TF-IDF Vectorizer** (for feature extraction)  
- **Candidate Models**:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
- *(Optional to extend: Linear SVM, MLP, XGBoost, LightGBM)*  

---

## Data Source  

- **Train Dataset**: `training.1600000.processed.noemoticon.csv`  
  - 1.6M tweets labeled via emoticons.  
- **Test Dataset**: `testdata.manual.2009.06.14.csv`  
  - A smaller manually annotated test set.  

Columns included in the dataset:

| Column | Description |
|--------|-------------|
| 0 = target | Sentiment label (**0 = negative**, **4 = positive → remapped to 1**) |
| 1 = id     | Tweet ID |
| 2 = date   | Date of tweet |
| 3 = flag   | Query/flag (unused) |
| 4 = user   | Username |
| 5 = text   | Tweet content |


---

## Methodology  

**1. Data Preparation**  
- Map target labels: `4 → 1` (binary classes).  
- Clean tweets:  
  - Lowercasing  
  - Remove URLs, mentions, hashtags, non-letters  
  - Tokenize and remove stopwords  

**2. Sampling for Efficiency**  
- Use a **200k tweet sample** from the training set for faster experiments.  
- Keep all positive/negative samples from the test set.  

**3. Feature Extraction**  
- Use **TF-IDF** (`max_features=5000`) to represent tweets numerically.  

**4. Candidate Models**  
- Logistic Regression (baseline linear model)  
- Decision Tree (non-linear classifier)  
- Random Forest (ensemble of trees)  

**5. Model Evaluation**  
- Train each model on training data.  
- Predict on test data.  
- Report **classification metrics** (precision, recall, F1).  
- Select **best model** based on weighted F1-score.  

**6. Save Artifacts**  
- Save the best-performing model (`sentiment_model.pkl`).  
- Save the fitted TF-IDF vectorizer (`tfidf.pkl`).  

---

## Result and Impact  

- The pipeline successfully identifies the **best sentiment model** among candidates.  
- **Weighted F1-score** ensures fair evaluation under class imbalance.  
- The trained pipeline can be reused for inference on unseen tweets.  
- Artifacts are ready for deployment into a real-time system (e.g., Streamlit).  

> *(You can update this section with actual F1-scores and confusion matrices once you run experiments.)*  

---

## Challenges and Solutions  

**Noise in Tweets**  
- Challenge: Tweets contain hashtags, links, and mentions.  
- Solution: Regex-based cleaning and stopword removal.  

**Large Dataset**  
- Challenge: Training on 1.6M tweets can be time-consuming.  
- Solution: Use random sampling of 200k rows for fast experimentation.  

**Class Imbalance**  
- Challenge: Dataset may not be perfectly balanced.  
- Solution: Use **weighted F1-score** to account for imbalance.  

**Reproducibility**  
- Challenge: Random sampling leads to variability.  
- Solution: Fix `random_state=42` in sampling and models.  

---

## Sneak Peek of the App  

<p align="center">
  <img src="Preview_Sentiment.gif" alt="Twitter Sentiment App Preview" style="width:70%;">
</p>

<p align="center"><em>Interactive Streamlit app (future extension) for live tweet sentiment predictions.</em></p>
