import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()
# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

data = pd.read_csv('Mental_Health_FAQ.csv')
data.drop('Question_ID', axis = 1, inplace = True)



# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    global tokens
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


data['tokenized Questions'] = data['Questions'].apply(preprocess_text)

corpus = data['tokenized Questions'].to_list()

#Vectorization process
tfidf_vector = TfidfVectorizer()
v_corpus = tfidf_vector.fit_transform(corpus) 

# -------------------------------- STEAMLT DESIGN ---------------------------
st.markdown("<h1 style = 'color: #39A7FF; text-align: center; font-family:montserrat'>Chat Bot Project</h1>",unsafe_allow_html=True)
st.markdown("<h3 style = 'margin: -15px; color: #39A7FF; text-align: center; font-family:montserrat'>Chat Bot build by Datapsalm </h3>",unsafe_allow_html=True)


st.markdown("<br></br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.image('my robot.png', caption = 'Mental Health Related Chats')


# Putting it all in a function
def response(user_input):
    user_input_processed = preprocess_text(user_input)
    v_input = tfidf_vector.transform([user_input_processed])
    most_similar = cosine_similarity(v_input, v_corpus)
    most_similar_index = most_similar.argmax()
    
    return data['Answers'].iloc[most_similar_index]



import random

chatbot_greeting = [
    "Hello there, welcome to Orpheus Bot. pls ejoy your usage",
    "Hi user, This bot is created by oprheus, enjoy your usage",
    "Hi hi, How you dey my nigga",
    "Alaye mi, Abeg enjoy your usage",
    "Hey Hey, pls enjoy your usage"    
]


user_greeting = ["hi", "hello there", "hey", "hi there"]
exit_word = ['bye', 'thanks bye', 'exit', 'goodbye']


# print(f'\t\t\t\t\tWelcome To Orpheus ChatBot\n\n')
# while True:
#     user_q = input('Pls ask your mental illness related question: ')
#     if user_q in user_greeting:
#         print(random.choice(chatbot_greeting))
#     elif user_q in exit_word:
#         print('Thank you for your usage. Bye')
#         break
#     else:
#         responses = response(user_q)
#         print(f'ChatBot:  {responses}')


# st.write(f'\t\t\t\t\tWelcome To Orpheus ChatBot\n\n')
# while True:
user_q = col2.text_input('Please ask your mental illness related question: ')
if user_q in user_greeting:
        col2.write(random.choice(chatbot_greeting))
elif user_q in exit_word:
        col2.write('Thank you for your usage. Bye')
elif user_q == '':
        st.write('')
else:
        responses = response(user_q)
        col2.write(f'ChatBot:  {responses}')
