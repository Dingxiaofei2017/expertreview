#分词
#获取名词短语
#去除停用词
import json
import string
import nltk
from my_method import noun_chunk
from nltk.corpus import stopwords
from my_method import get_lemma
from my_method import get_tokens

#read data
with open('data/cristic_consensus.json','r') as f:
    consensus = json.load(f)
with open('data/cristic.json','r') as f:
    cristic = json.load(f)

#tokenize(rmove stopwords)
cristic_token = []
for item in cristic:
    temp = []
    for item_01 in item:
        temp_01 = get_lemma(item_01)
        temp_01 = get_tokens(item_01)
        temp.append(temp_01)
    cristic_token.append(temp)
with open('data/cristic_token.json','w') as f:
    json.dump(cristic_token, f)

consensus_token = []
for item in consensus:
    temp = get_lemma(item)
    temp = get_tokens()
    consensus_token.append(temp)

with open('data/consensus_token.json','w') as f:
    json.dump(consensus_token, f)


#remove stopword
critics_temp = []
critics = [[x.lower() for x in c] for c in cristic]
for item in critics:
    temp = [''.join(c for c in item_01 if c not in string.punctuation) for item_01 in item]
    critics_temp.append(temp)

critics = critics_temp
critics = [[nltk.word_tokenize(x) for x in item] for item in critics]

critics = [[' '.join(c for c in item if c not in stopwords.words('english')) for item in item_01] for item_01 in critics]

with open('data/cristic_no_stop.json', 'w') as f:
    json.dump(critics,f)


#get noun
consensus_noun_chunk = [noun_chunk(item) for item in consensus]
with open('data/consensus_noun_chunk.json','w') as f:
    json.dump(consensus_noun_chunk, f)

cristic_noun_chunk = []
for _ in cristic:
    temp_01 = [noun_chunk(item) for item in _]
    cristic_noun_chunk.append(temp_01)
with open('data/cristic_noun_chunk.json','w') as f:
    json.dump(cristic_noun_chunk, f)
