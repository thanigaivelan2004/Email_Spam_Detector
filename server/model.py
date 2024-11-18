import pickle
import sys

def preprocess_text(text):
    # Add preprocessing steps here (e.g., lowercasing, removing punctuation, etc.)
    return text.lower().strip()

def main():
    try:
        # Load the trained model
        model = pickle.load(open('spam_model.pkl', 'rb'))
    except FileNotFoundError:
        print("Error: spam_model.pkl not found.")
        sys.exit(1)

    # Check for input text
    if len(sys.argv) < 2:
        print("Usage: python model.py <email_text>")
        sys.exit(1)

    # Get and preprocess the input text
    email_text = preprocess_text(sys.argv[1])

    # Make a prediction
    prediction = model.predict([email_text])[0]
    result = "Spam" if prediction == 1 else "Not Spam"

    print(result)

if __name__ == "__main__":
    main()
