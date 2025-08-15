# My Data Science Portfolio

Welcome to my portfolio of data science projects, showcasing my skills in data analysis, machine learning, and scalable solutions.

---

## 1. GPS Trajectory Anomaly Detection and "Normal Route" Generation

**Description:** This project analyzes the GoLife GPS Trajectories dataset (18,000+ trajectories) to detect route anomalies and generate "normal" paths, simulating fleet monitoring.

**Key Technologies:** Python (Pandas, Shapely, scikit-learn), Folium for visualization. Discusses scalability with PySpark.

**Technical Details:**
* **Data Acquisition & Preprocessing:** Parsing raw `.plt` files, extracting `latitude`, `longitude`, `timestamp`. Implemented speed-based filtering to isolate driving segments.
* **"Normal Route" Generation:** Employed DBSCAN clustering on trip start/end points to define common O-D pairs. A "normal route" for an O-D pair is derived from the most common/densest trajectory in that cluster (or a representative average).
* **Anomaly Detection:** Utilizes Haversine distance calculations to measure deviations of test trips from the defined "normal route." Trips with points exceeding a predefined distance threshold are flagged as anomalous.
* **Visualization:** Interactive maps generated with Folium to display historical trips, the "normal" route, and highlighted anomalous segments.
* **Scalability Discussion:** Conceptual outline for distributed processing using PySpark for large-scale trajectory analysis, including window functions and UDFs.

**[Go to Project Repository](https://github.com/your-username/gps-anomaly-project)**

---

## 2. Big Five Personality Traits EDA & Hypothesis Testing

**Description:** This project performed extensive Exploratory Data Analysis (EDA) on a 1M+ record Big Five personality dataset, analyzing trait distributions, response times, and inter-trait correlations.

**Key Technologies:** Python (Pandas, Matplotlib, Seaborn), Statistical methods.

**Technical Details:**
* **Data Cleaning:** Handled missing values (row-wise removal), performed reverse scoring for negatively-worded items (`new_value = 6 - original_value`).
* **Feature Engineering:** Calculated Big Five trait scores as the mean of 10 respective items. Applied outlier capping and Box-Cox transformation to `testelapse` for normalization.
* **Statistical Analysis:** Generated descriptive statistics, visualized trait distributions (histograms), and explored inter-trait relationships using Pearson correlation (e.g., Extraversion and Openness: `r = 0.149, p < 0.001`). Conducted hypothesis tests on trait relationships with completion time.

**[Go to Project Repository](https://github.com/ManikPandey/GPS-Anomaly-Detection-Route-Generation-Trajectories)**

---

## 3. Multi-Dataset Sentiment Analysis with ML/DL

**Description:** This project implements sentiment analysis using LSTM, Logistic Regression, and Bernoulli Naive Bayes on four diverse datasets (Sentiment140, IMDB, Amazon Reviews, Yelp), leveraging pre-trained GloVe embeddings.

**Key Technologies:** Python (TensorFlow/Keras, Scikit-learn), Natural Language Processing (NLP), GloVe word embeddings.

**Technical Details:**
* **Data Preparation:** Handled large text datasets, tokenization, and conversion to numerical representations for model input.
* **Feature Extraction:** Utilized pre-trained 50d to 300d GloVe word embeddings for dense vector representations of text, capturing semantic relationships.
* **Machine Learning Models:**
    * **Logistic Regression & Bernoulli Naive Bayes:** Implemented for baseline binary classification.
    * **LSTM (Long Short-Term Memory):** A deep learning recurrent neural network applied for its effectiveness in processing sequential text data, capturing long-range dependencies for sentiment classification.
* **Model Evaluation:** Assessed performance across different datasets and algorithms using metrics like accuracy, precision, recall, and F1-score.

**[Go to Project Repository](https://github.com/ManikPandey/Sentimental_analysis_4dataset)**

---

## 3.  Predictive Modeling of Electricity Costs Regression

**Description:** This project develops and compares several regularized linear regression models to accurately predict monthly electricity costs. Using a dataset of modern infrastructure sites, the analysis identifies the key drivers of energy expenditure through feature importance analysis.

**Key Technologies:**Python (Pandas, NumPy, Scikit-learn), Machine Learning (Regression), Data Preprocessing, Hyperparameter Tuning.

**Technical Details:**
* **Data Preparation:**Loaded a clean dataset of 10,000 records, standardized column names for consistency, and verified data integrity, confirming no missing values.
* **Feature Engineering & Preprocessing:**
         * **Standard Scaling:** Applied `StandardScaler` to all numerical features to normalize their distributions, a crucial step for regularized regression.
         * **One-Hot Encoding:** Converted the categorical `structure type` feature into a numerical format using `OneHotEncoder` to make it compatible with the machine learning models.
* **Machine Learning Models:**
    * **Ridge, Lasso, & ElasticNet Regression:**  Implemented these three advanced linear models to handle potential multicollinearity and prevent overfitting.
    * **Cross-Validated Hyperparameter Tuning:** Utilized Scikit-learn's `RidgeCV`, `LassoCV`, and `ElasticNetCV` to automatically and efficiently find the optimal regularization strength `(alpha)` for each model, ensuring robustness.
* **Model Evaluation** Evaluated the final model on a held-out test set. The model demonstrated strong predictive power, achieving an R-squared (R2) of 0.9199, indicating it explains 92% of the variance in electricity costs. The Root Mean Squared Error (RMSE) was $313.99 and the Mean Absolute Error (MAE) was $247.62, quantifying the model's high accuracy in predicting monthly expenditures.

**[Go to Project Repository](https://github.com/ManikPandey/data-science-projects/blob/main/FinalAssignment_v1.ipynb)**


---



