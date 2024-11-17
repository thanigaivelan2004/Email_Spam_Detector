import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1', header=None)
df = df.iloc[:, [0, 1]]  # Select the first two columns (label, message)
df.columns = ['label', 'message']

# Check for missing values and display counts
print("Missing values before cleaning:")
print(df.isnull().sum())

# Remove rows with missing values
df.dropna(subset=['label', 'message'], inplace=True)

# Remove any leading or trailing whitespace in the label column
df['label'] = df['label'].str.strip()

# Convert labels to binary (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop any remaining rows with NaN values in the label column
df = df.dropna(subset=['label'])

# Check again for missing values after cleaning
print("Missing values after cleaning:")
print(df.isnull().sum())

# Ensure there are no NaN values in the label column
if df['label'].isnull().any():
    raise ValueError("The label column contains NaN values even after cleaning.")

# Split the dataset into features and labels
X = df['message']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer and MultinomialNB
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

print('Model saved as spam_model.pkl')
