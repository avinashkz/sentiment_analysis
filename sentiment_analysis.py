
# Importing libraries
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.ensemble import VotingClassifier


classifier_f = open("pickled/Voting_classifier.pickle", "rb")
Voting_classifier = pickle.load(classifier_f)
classifier_f.close()
print("Sentiment Analysis Module Loaded!")

features_f = open("pickled/features.pickle", "rb")
word_features = pickle.load(features_f)
features_f.close()
print("Word Features Loaded!")



def find_features(document):
    '''
    document = list of all the words in a review
    '''
    # To extract only the unique words in a document
    words = word_tokenize(document)
    features = {}
    
    #Set true or false based on the if the word is 
    #present in the top 3000 words
    for w in word_features:
        features[w] = (w in words)
        
    return features

    
def sentiment_module(text):
    feats = find_features(text)
    feats = [items for (keys, items) in feats.items()]
    feats = np.array(feats).reshape(1, -1)
    prediction = Voting_classifier.predict(feats)
    confidence = Voting_classifier.predict_proba(feats)
    return prediction, np.max(confidence)


