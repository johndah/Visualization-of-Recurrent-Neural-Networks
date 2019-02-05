'''
import spacy

nlp = spacy.load('en_core_web_lg')  # make sure to use larger model!
tokens = nlp(u'dog cat banana')

#for token1 in tokens:
#    for token2 in tokens:
#        print(token1.text, token2.text, token1.similarity(token2))
print(tokens[0])
a = tokens[0].vector
2

import subprocess
print('\033[0;31mTEST\033[0m')
subprocess.call('', shell=True)
print('\033[0;31mTEST\033[0m')


import platform
if platform.system().lower() == 'windows':
    print('YES')
    from ctypes import windll, c_int, byref
    stdout_handle = windll.kernel32.GetStdHandle(c_int(-11))
    mode = c_int(0)
    windll.kernel32.GetConsoleMode(c_int(stdout_handle), byref(mode))
    mode = c_int(mode.value | 4)
    windll.kernel32.SetConsoleMode(c_int(stdout_handle), mode)



import spacy

nlp = spacy.load('de')

vector1 = nlp("Queen")[0].vector
vector2 = nlp("King")[0].vector
vector3 = vector1 - vector2

result = nlp.vocab.vectors.most_similar(vector1)
result = nlp.vocab.vectors.most_similar(vector2)
result = nlp.vocab.vectors.most_similar(vector3)
'''

'''
from numpy import *
import re
import zipfile
import lxml.etree
#download the data
#urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
# extract subtitle
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = '\n'.join(doc.xpath('//content/text()'))


# remove parenthesis
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
'''

'''
# store as list of sentences
sentences_strings_ted = []
for line in input_text.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
# store as list of lists of words

sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)

from gensim.models import Word2Vec

model = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=4, sg=0)


print(model.wv.vocab['here'].index)
print(model.wv.index2word[69])
a = array([model['here']])
b = array([model['there']])
c = array([model['where']])
d = array([model['great']])

print(model.wv.index2word[0])

vectors =  Word2Vec.load_word2vec_format(file)
# vectors.word_from_vec(vectors['cat']) == 'cat'

vectors.most_similar(positive=[vectors['cat']],topn=1)

f = linalg.norm(a*b)
g = linalg.norm(a *c)
h = linalg.norm(a*d)

e = 2
'''

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
            ['this', 'is', 'the', 'second', 'sentence'],
            ['yet', 'another', 'sentence'],
            ['one', 'more', 'sentence'],
            ['and', 'the', 'final', 'sentence']]
# train model
model_1 = Word2Vec(sentences, size=300, min_count=1)

# fit a 2d PCA model to the vectors
X = model_1[model_1.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = list(model_1.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

# plt.show()
from numpy import *
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

file = 'Data/glove_short.txt'
# 'Data/glove.840B.300d.txt'
model_2 = Word2Vec(size=300, min_count=1)
model_2.build_vocab(sentences)
total_examples = model_2.corpus_count
model = KeyedVectors.load_word2vec_format(file, binary=False)

model_2.build_vocab([list(model.vocab.keys())], update=True)
model_2.intersect_word2vec_format(file, binary=False, lockf=1.0)
model_2.train(sentences, total_examples=total_examples, epochs=model_2.iter)

some_output = random.randn(300)
print('\nSome output:', some_output[:10])
list_most_similar = model.most_similar(positive=[some_output],topn=5)
print('\nList of most similar', list_most_similar)
most_similar = list_most_similar[1][0]
print('\nMost similar word', most_similar)
most_similar_vec = model[most_similar]
second_most_similar = list_most_similar[2][0]
second_most_similar_vec = model[list_most_similar[2][0]]
print('\nIts vector', most_similar_vec[:10])
print(linalg.norm(most_similar_vec[:10]))
distance = model.distance(second_most_similar, second_most_similar)
print('Distance', distance)

def computeLoss(output, y):
    # Cross entropy loss
    loss = -sum(log(dot(y.T, p)))

    dotprod = dot(y.T, p)
    logdot = log(dotprod)

    return loss

def softmax(s):
    exP = exp(s)
    p = exP/exP.sum()

    return p


c = zeros((300, 1))
c[3, 0] = 1
p = softmax(c)
loss = computeLoss(p, p)
print(loss)

# print(model.wv.index2word[0])


'''
# fit a 2d PCA model to the vectors
X = model_2[model_1.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = list(model_1.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()

'''
