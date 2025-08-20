import requests
from bs4 import BeautifulSoup
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai_secret_manager
import openai

# Set up the OpenAI API client
secrets = openai_secret_manager.get_secret("openai")
openai.api_key = secrets["api_key"]

# Download the stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove stop words and punctuations
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text

def get_vectors(texts):
    # Preprocess the texts
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # Create the CountVectorizer
    vectorizer = CountVectorizer().fit_transform(preprocessed_texts)
    
    # Get the vectors
    vectors = vectorizer.toarray()
    
    return vectors

def remove_similar_text(texts):
    # Get the vectors
    vectors = get_vectors(texts)
    
    # Get the cosine similarities
    similarities = cosine_similarity(vectors)
    
    # Get the indices to remove
    indices_to_remove = []
    for i in range(similarities.shape[0]):
        for j in range(i+1, similarities.shape[1]):
            if similarities[i,j] > 0.8:
                indices_to_remove.append(j)
                
    # Remove the similar texts
    new_texts = []
    for i in range(len(texts)):
        if i not in indices_to_remove:
            new_texts.append(texts[i])
            
    return new_texts

def generate_article(prompt, max_tokens=2048, temperature=0.7):
    # Set up the prompt for the OpenAI API
    prompt = (f"{prompt}\n\nThe following article was generated using AI.\n\n")
    completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        stop=None,
        timeout=20,
    )
    message = completions.choices[0].text
    
    # Print the generated article
    print(message)

def web_scrape(topic):
    # Quizlet
    quizlet_url = 'https://quizlet.com/subject/' + topic
    quizlet_res = requests.get(quizlet_url)
    quizlet_soup = BeautifulSoup(quizlet_res.content, 'html.parser')
    quizlet_text = quizlet_soup.get_text()
    
    # Britannica
    britannica_url = 'https://www.britannica.com/search?query=' + topic
    britannica_res = requests.get(britannica_url)
    britannica_soup = BeautifulSoup(britannica_res.content, 'html.parser')
    britannica_text = britannica_soup.get_text()
    
    # Brainly
    brainly_url = 'https://brainly.com/question/?q=' + topic
    brainly_res = requests.get(brainly_url)
    brainly_soup = BeautifulSoup(brainly_res.content, 'html.parser')
    brainly_text = brainly_soup.get_text()

    # Wikipedia
    wikipedia_url = 'https://en.wikipedia.org/wiki/' + topic
    wikipedia_res = requests.get(wikipedia_url)
    wikipedia_soup = BeautifulSoup(wikipedia_res.content, 'html.parser')
    wikipedia_text = wikipedia_soup.get_text()

    # Remove similar text
    texts = [quizlet_text, britannica_text, brainly_text, wikipedia_text]
    texts = remove_similar_text(texts)

    # Generate the article
    prompt = '\n\n'.join(texts)
    generate_article(prompt)

