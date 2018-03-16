import nltk
import string
from nltk.corpus import stopwords
import spacy
import re
from nltk.stem.porter import *

nlp = spacy.load('en_core_web_lg')

#词性还原
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#注意啦，这边接受的就是一个字符串
def ie_process(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


def get_tokens(document):
    document = document.lower()
    document = ''.join(c for c in document if c not in string.punctuation)
    document = nltk.word_tokenize(document)
    document = [c for c in document if c not in stopwords.words('english')]
    #stemmer = PorterStemmer()
    #document = stem_tokens(document, stemmer)
    return document

def get_lemma(document):
    document = nlp(document)
    document = ' '.join(token.lemma_ for token in document)
    return document

def noun_chunk(document):
    doc = nlp(document)
    document = [item.text for item in doc.noun_chunks]
    return document

def clean_title(document):
    document = re.split(r'[_-]',  document)
    return document

def word_level_sim(sentence_a,sentence_b):
    #input two sentences
    #output each word of sentence_a sim_score
    a = get_lemma(sentence_a)
    a = get_tokens(a)
    b = get_lemma(sentence_b)
    b = get_tokens(b)
    score = [max([nlp(item_01).similarity(nlp(item_02)) for item_02 in b]) for item_01 in a]
    return score