# model_training.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline

# Load the dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Data Cleaning
df.drop_duplicates(inplace=True)
df = df[df['gender'] != 'Other']  # Remove rare category in 'gender'

# Fill missing values for numeric columns
numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

# Define features and target
X = df[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
        'hypertension', 'heart_disease', 'gender_Male', 'smoking_history_former',
        'smoking_history_never', 'smoking_history_current']]
y = df['diabetes']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']),
        ('passthrough', 'passthrough', ['hypertension', 'heart_disease', 'gender_Male',
                                         'smoking_history_former', 'smoking_history_never',
                                         'smoking_history_current']),
    ]
)

# Define resampling
smote = SMOTE(random_state=42)

def create_pipeline(classifier):
    return imbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', classifier)
    ])

def train_and_save_model(classifier, filename, X_train, y_train):
    pipeline = create_pipeline(classifier)
    pipeline.fit(X_train, y_train)
    with open(filename, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"Model saved to {filename}")

# Train and save models
train_and_save_model(RandomForestClassifier(random_state=42), 'random_forest_model.pkl', X_train, y_train)
train_and_save_model(DecisionTreeClassifier(random_state=42), 'decision_tree_model.pkl', X_train, y_train)
train_and_save_model(LogisticRegression(max_iter=1000, random_state=42), 'logistic_regression_model.pkl', X_train, y_train)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ],
    voting='hard'
)
train_and_save_model(voting_clf, 'voting_classifier_model.pkl', X_train, y_train)

# Evaluate models
for model_name, model_file in [('Random Forest', 'random_forest_model.pkl'), 
                               ('Decision Tree', 'decision_tree_model.pkl'), 
                               ('Logistic Regression', 'logistic_regression_model.pkl'), 
                               ('Voting Classifier', 'voting_classifier_model.pkl')]:
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {acc:.2f}")

print("Training complete.")