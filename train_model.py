import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import joblib
from scipy.stats import randint, uniform

# Preprocess text data
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    return text

# Load the dataset
data = pd.read_csv(os.path.join('data', 'dataset.csv'))  # Adjust the path as necessary

# Preprocess the text
data['text'] = data['text'].apply(preprocess_text)

# Handle class imbalance if necessary
max_samples = data['label'].value_counts().max()
balanced_data = pd.concat([
    resample(data[data['label'] == label], 
             replace=True, 
             n_samples=max_samples, 
             random_state=42)
    for label in data['label'].unique()
])

# Split the data into features and labels
X = balanced_data['text']
y = balanced_data['label']

# Create a pipeline with TfidfVectorizer and RandomForestClassifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier(random_state=42))
])

# Define a randomized parameter grid for RandomizedSearchCV
param_distributions = {
    'tfidf__max_df': [0.6, 0.75, 0.9],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__max_features': [None, 5000, 10000],
    'clf__n_estimators': randint(100, 1000),  # Wider range
    'clf__max_depth': [None, 10, 20, 30, 50],
    'clf__min_samples_split': randint(2, 20),
    'clf__min_samples_leaf': randint(1, 10),
    'clf__max_features': ['sqrt', 'log2', None],
    'clf__bootstrap': [True, False],
    'clf__class_weight': [None, 'balanced']
}

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(
    pipeline, 
    param_distributions, 
    n_iter=100,  # Increase iterations for better parameter search
    cv=5,        # Stratified k-fold
    scoring='accuracy', 
    n_jobs=-1, 
    verbose=2,
    random_state=42
)

random_search.fit(X, y)

# Print the best parameters and the corresponding score
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best cross-validation accuracy: {random_search.best_score_}")

# Get the best model from random search
best_model = random_search.best_estimator_

# Make predictions on the training set
y_pred = best_model.predict(X)

# Print the classification report
print("Classification Report:")
print(classification_report(y, y_pred))

# Print the accuracy score
print(f"Accuracy Score: {accuracy_score(y, y_pred)}")

# Save the trained model to a file
model_path = 'models/skill_model_rf_tuned.pkl'
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")
