import pickle
import sys

# Load the trained model
model = pickle.load(open('spam_model.pkl', 'rb'))

# Get the email text from command line arguments
email_text = sys.argv[1]

# Make a prediction
prediction = model.predict([email_text])[0]
result = "Spam" if prediction == 1 else "Not Spam"

print(result)
