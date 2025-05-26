# Install the transformers library
!pip install transformers

# Import the summarization pipeline from Hugging Face
from transformers import pipeline

# Load a more accurate pre-trained summarization model
print("Loading Summarization Model (BART)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize text with improved accuracy
def summarize_text(text, max_length=None, min_length=None):
    """
    Summarizes a given long text using a pre-trained BART summarization model.

    Args:
        text (str): The input passage to summarize.
        max_length (int): Maximum length of the summary (default: auto-calculated).
        min_length (int): Minimum length of the summary (default: auto-calculated).

    Returns:
        str: The summarized text.
    """
    # Remove unnecessary line breaks
    text = " ".join(text.split())

    # Auto-adjust summary length based on text size
    if not max_length:
        max_length = min(len(text) // 3, 150)  # Summary should be ~1/3rd of input
    if not min_length:
        min_length = max(30, max_length // 3)  # Minimum length should be at least 30

    # Generate the summaries with different configurations
    summary_1 = summarizer(text, max_length=150, min_length=30, do_sample=False)
    summary_2 = summarizer(text, max_length=150, min_length=30, do_sample=True, temperature=0.9)
    summary_3 = summarizer(text, max_length=150, min_length=30, do_sample=False, num_beams=5)
    summary_4 = summarizer(text, max_length=150, min_length=30, do_sample=True, top_k=50, top_p=0.95)

    # Print original and summarized text
    print("\nOriginal Text:")
    print(text)

    print("\nSummarized Text:")
    print("Default:", summary_1[0]['summary_text'])
    print("High randomness:", summary_2[0]['summary_text'])
    print("Conservative:", summary_3[0]['summary_text'])
    print("Diverse sampling:", summary_4[0]['summary_text'])

# Example long text passage
long_text = """
Artificial Intelligence (AI) is a rapidly evolving field of computer science focused on creating intelligent machines
capable of mimicking human cognitive functions such as learning, problem-solving, and decision-making. In recent years,
AI has significantly impacted various industries, including healthcare, finance, education, and entertainment.
AI-powered applications, such as chatbots, self-driving cars, and recommendation systems, have transformed the way we
interact with technology. Machine learning and deep learning, subsets of AI, enable systems to learn from data and improve
over time without explicit programming. However, AI also poses ethical challenges, such as bias in decision-making and
concerns over job displacement. As AI technology continues to advance, it is crucial to balance innovation with ethical
considerations to ensure its responsible development and deployment.
"""

# Summarize the passage
summarize_text(long_text)
