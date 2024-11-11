import nltk
nltk.download('punkt')  # This should download the basic punkt tokenizer data
nltk.download('punkt_tab')  # This will ensure the 'punkt_tab' data is also available
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import string

# Step 1: Ensure that 'punkt' is available for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' for sentence tokenization...")
    nltk.download('punkt')

# Function for extractive summarization
def extractive_summary(text, num_sentences):
    # Tokenize sentences
    sentences = nltk.sent_tokenize(text)
    # Compute TF-IDF matrix and cosine similarity
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    similarity_matrix = cosine_similarity(vectorizer)
    # Rank sentences based on similarity scores
    ranked_sentences = sorted(((similarity_matrix[i].sum(), s) for i, s in enumerate(sentences)), reverse=True)
    # Return the top-ranked sentences as summary
    summary = " ".join([sentence for _, sentence in ranked_sentences[:num_sentences]])
    return summary

# Function for abstractive summarization
def abstractive_summary(text, num_sentences):
    summarizer = pipeline("summarization")
    # Use the summarizer to generate an abstractive summary
    return summarizer(text, max_length=num_sentences * 10, min_length=num_sentences * 5, do_sample=False)[0]['summary_text']

# Main function to generate the summary based on user's choice
def generate_summary(text, num_sentences, use_abstractive):
    if use_abstractive:
        return abstractive_summary(text, num_sentences)
    else:
        return extractive_summary(text, num_sentences)

def main():
    document = input("Enter the document text: ")

    # Loop until a valid number is entered
    while True:
        num_sentences_input = input("Enter the number of sentences for the summary (integer only): ")
        try:
            num_sentences = int(num_sentences_input)
            if num_sentences <= 0:
                print("Please enter a positive integer for the number of sentences.")
                continue
            break  # Exit the loop if input is valid
        except ValueError:
            print(f"Invalid input '{num_sentences_input}'. Please enter a valid integer.")

    use_abstractive = input("Do you want an abstractive summary? (yes/no): ").strip().lower() == "yes"
    
    # Generate and display summary
    summary = generate_summary(document, num_sentences, use_abstractive)
    print("\nSummary:\n", summary)

# Corrected entry point
if __name__ == "__main__":
    main()
