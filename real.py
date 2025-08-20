#for judges: The code first loads pre-trained GloVe embeddings for word vectors and creates a deep learning model 
# for part-of-speech tagging. It then loads the pre-trained weights for this model. The extract_definitions function uses 
# this model to perform part-of-speech tagging on the input text and extract definitions for specific nouns. Finally, the Flask 
# application is set up to serve a web page that takes in user input, calls the extract_definitions function, and displays the results.


#future goals: Increasing the depth and complexity of the neural network model used for part-of-speech tagging, such as by adding more LSTM layers or incorporating attention mechanisms.
#Incorporating additional NLP tasks, such as named entity recognition or sentiment analysis, and integrating these tasks with the 
#existing model. Incorporating more advanced pre-processing techniques, such as lemmatization or stemming, to improve the accuracy of the part-of-speech tagging and definition extraction steps.
#Using more advanced techniques for embedding words, such as using contextualized word embeddings like BERT or ELMo instead 
#of pre-trained GloVe embeddings. Adding a more complex user interface with more features, such as the ability to save and load 
#previously entered text, or the ability to visualize the output of the part-of-speech tagging and definition extraction steps in 
#a more interactive way.

import numpy as np
import nltk
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request

app = Flask(__name__)

def load_glove_embeddings(embeddings_file_path):
    """Load pre-trained GloVe embeddings from a file.

    Args:
        embeddings_file_path (str): Path to the GloVe embeddings file.

    Returns:
        tuple: A tuple containing the embedding matrix and a dictionary mapping words to their indices in the matrix.
    """
    embeddings_index = {}
    with open(embeddings_file_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_dim = len(coefs)
    embedding_matrix = np.zeros((len(embeddings_index) + 1, embedding_dim))
    word_to_index = {}
    index_to_word = {}
    for i, word in enumerate(embeddings_index.keys()):
        embedding_matrix[i+1] = embeddings_index[word]
        word_to_index[word] = i+1
        index_to_word[i+1] = word
    return embedding_matrix, word_to_index, index_to_word

embedding_matrix, word_to_index, index_to_word = load_glove_embeddings('glove.6B.100d.txt')

def create_model(input_shape, output_shape):
    """Create a deep learning model for part-of-speech tagging.

    Args:
        input_shape (tuple): Shape of the input tensor.
        output_shape (int): Number of possible output tags.

    Returns:
        tensorflow.keras.Model: The compiled model.
    """
    input_layer = keras.layers.Input(shape=input_shape)
    embedding_layer = keras.layers.Embedding(len(word_to_index) + 1, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(input_layer)
    lstm_layer = keras.layers.LSTM(128, return_sequences=True)(embedding_layer)
    dropout_layer = keras.layers.Dropout(0.5)(lstm_layer)
    output_layer = keras.layers.Dense(output_shape, activation='softmax')(dropout_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 
            'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO',
            'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
model = create_model((None,), len(pos_tags))

model.load_weights('pos_tagger_weights.h5')

def extract_definitions(text):
    definitions = []
    tagged_sentence = pos_tagging(text)
    i = 0
    while i < len(tagged_sentence):
        term = ""
        definition = ""
        if tagged_sentence[i][1] in ["NN", "NNS", "NNP", "NNPS"]:
            term += tagged_sentence[i][0]
            i += 1
            while i < len(tagged_sentence) and tagged_sentence[i][1] in ["NN", "NNS", "NNP", "NNPS", "JJ"]:
                term += " " + tagged_sentence[i][0]
                i += 1
            if i < len(tagged_sentence) and tagged_sentence[i][1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                j = i - 1
                while j >= 0 and tagged_sentence[j][2] != "O":
                    j -= 1
                k = i
                while k < len(tagged_sentence) and tagged_sentence[k][2] != "O":
                    k += 1
                for l in range(j+1, k):
                    definition += tagged_sentence[l][0] + " "
                definition = definition[0].upper() + definition[1:].rstrip() + "."
                definitions.append((term, definition))
                i = k
        else:
            i += 1
    return definitions

@app.route('/', methods=['GET', 'POST'])
def flashcards():
    if request.method == 'POST':
        text = request.form['inputText']
    else:
        text = "Enter your text here"
    
    definitions = extract_definitions(text)
    
    return render_template('real.html', definitions=definitions)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
