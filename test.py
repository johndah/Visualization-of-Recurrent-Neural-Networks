
from __future__ import print_function

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

a = model_1['this']

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
distance = model.distance(most_similar, second_most_similar)
aa1 = array([most_similar_vec])
aa2 = array([second_most_similar_vec])
distance2 = linalg.norm((aa1-aa2))
print('Distance', distance)
print('Distance2', distance2)

'''
'''
a = model_1.wv.vocab['and']
print(a)
for word in vocab:
    if word not in word_vecs and vocab[word] >= min_df:
        word_vecs[word] = np.random.uniform(-0.25,0.25,k)


import re

words = []
textFile = 'LordOfTheRings2.txt'
with open(textFile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        words.extend(re.findall(r"\w+|[^\w\s]", line))
        words.append('\n')
# print(model.wv.index2word[0])

'''

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


counter = 0

# model_file = 'Data/glove_840B_300d.txt'
# model_file = 'Data/glove_test.txt'


with open(model_file, 'r+', encoding="utf8") as f:
    lines = f.readlines()

    first_line = lines[0]
    print(first_line)

    for line in lines:
        counter += 1

    #line = line.replace(line, str(counter) + ' 300\n' + line)

    first_line = lines[0]
    print(first_line)

    lines.insert(0, str(counter) + ' 300\n')

    f.seek(0)  # readlines consumes the iterator, so we need to start over
    f.writelines(lines)


print(counter)
'''


'''
from textwrap import wrap

from terminaltables import SingleTable

LONG_STRING = ('Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore '
               'et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut '
               'aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum '
               'dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui '
               'officia deserunt mollit anim id est laborum.')


def main():
    """Main function."""
    table_data = [
        ['Long String', '']  # One row. Two columns. Long string will replace this empty string.
    ]
    table = SingleTable(table_data)

    # Calculate newlines.
    max_width = table.column_max_width(1)
    wrapped_string = '\n'.join(LONG_STRING)
    wrapped_string = '\n'.join(wrap(LONG_STRING, max_width))
    wrapped_string = '\n'.join(wrap(LONG_STRING, 100))
    table.table_data[0][1] = wrapped_string

    print(table.table)

    print(LONG_STRING)
if __name__ == '__main__':
    main()
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(-2.0, 2.0, 0.001)
s = t ** 2
initial_text = "t ** 2"
l, = plt.plot(t, s, lw=2)


def submit(text):
    ydata = eval(text)
    l.set_ydata(ydata)
    ax.set_ylim(np.min(ydata), np.max(ydata))
    plt.draw()

axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
text_box = TextBox(axbox, 'Evaluate', initial=initial_text)
text_box.on_submit(submit)

plt.show()

