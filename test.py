
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

'''

'''
from time import time

#import warc
from bs4 import BeautifulSoup
from selectolax.parser import HTMLParser


def get_text_bs(html):
    tree = BeautifulSoup(html, 'lxml')

    body = tree.body
    if body is None:
        return None

    for tag in body.select('script'):
        tag.decompose()
    for tag in body.select('style'):
        tag.decompose()

    text = body.get_text(separator='\n')
    return text


def get_text_selectolax(html):
    tree = HTMLParser(html)

    if tree.body is None:
        return None

    for tag in tree.css('script'):
        tag.decompose()
    for tag in tree.css('style'):
        tag.decompose()

    text = tree.body.text(separator='\n')
    return text


def read_doc(record, parser=get_text_selectolax):
    url = record.url
    text = None

    if url:
        payload = record.payload.read()
        header, html = payload.split(b'\r\n\r\n', maxsplit=1)
        html = html.strip()

        if len(html) > 0:
            text = parser(html)

    return url, text


def process_warc(file_name, parser, limit=10000):
    warc_file = warc.open(file_name, 'rb')
    t0 = time()
    n_documents = 0
    for i, record in enumerate(warc_file):
        url, doc = read_doc(record, parser)

        if not doc or not url:
            continue

        n_documents += 1

        if i > limit:
            break

    warc_file.close()
    print('Parser: %s' % parser.__name__)
    print('Parsing took %s seconds and produced %s documents\n' % (time() - t0, n_documents))

file_name = "CC-MAIN-20180116070444-20180116090444-00000.warc.gz"
process_warc(file_name, get_text_selectolax, 10000)
process_warc(file_name, get_text_bs, 10000)

import numpy as np
import os
from random import shuffle
import re
import urllib.request
#download the data
#urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
# extract subtitle
import zipfile
import lxml.etree
with zipfile.ZipFile('Data/ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = '\n'.join(doc.xpath('//content/text()'))
'''
'''
#from __future__ import print_function

__author__ = 'maxim'

import numpy as np
import gensim
import string

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file

#
print('\nFetching the text...')
url = 'https://raw.githubusercontent.com/maxim5/stanford-tensorflow-tutorials/master/data/arxiv_abstracts.txt'
path = get_file('arxiv_abstracts.txt', origin=url)

print('\nPreparing the sentences...')
max_sentence_len = 40
with open(path) as file_:
  docs = file_.readlines()
sentences = [[word for word in doc.lower().translate(string.punctuation).split()[:max_sentence_len]] for doc in docs]
print('Num sentences:', len(sentences))

print('\nTraining word2vec...')
word_model = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=5, iter=100)
from gensim.models import KeyedVectors
import zipfile
import lxml.etree
import re
from numpy import *


textFile = 'Data/ted_en.zip'  # Name of book text file, needs to be longer than lengthSynthesizedTextBest
# model_file = 'Data/glove_short.txt'  # 'Data/glove_840B_300d.txt',  #

#is_binary = model_file[-4:] == '.bin'
# print('Loading model "' + model_file + '"...')
# word_model = KeyedVectors.load_word2vec_format(model_file, binary=is_binary)
# print('Loading text file "' + textFile + '"...')

words = []
if textFile[-4:] == '.zip':
    with zipfile.ZipFile(textFile, 'r') as z:
        doc = lxml.etree.parse(z.open(z.filelist[0].filename, 'r'))

    print('Extracting words...')
    # input_text = '\n'.join(doc.xpath('//content/text()'))
    lines = doc.xpath('//content/text()')
    max_sentence_len = 40
    sentences = [[word for word in line.lower().translate(string.punctuation).split()[:max_sentence_len]] for line in
                 lines]

    # words.extend(re.findall(r"\w+|[^\w]", input_text))
else:
    with open(textFile, 'r') as f:
        lines = f.readlines()
        print('Extracting words...')
        for line in lines:
            words.extend(re.findall(r"\w+|[^\w]", line))
            words.append('\n')


word_model = gensim.models.Word2Vec(sentences, size=300, min_count=1, window=5, iter=10) # 100
pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)
print('Checking similar words:')
for word in ['this', 'is', 'a', 'why']:
  most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.most_similar(word)[:8])
  print('  %s -> %s' % (word, most_similar))

def word2idx(word):
  return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]

print('\nPreparing the data for LSTM...')
train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
    try:
        train_x[i, t] = word2idx(word)
    except KeyError:
        word_model[word] = random.uniform(-0.25, 0.25, emdedding_size)
        print("Word '" + word + "'" + ' added to model.')
  try:
    train_y[i] = word2idx(sentence[-1])
  except KeyError:
    word_model[word] = random.uniform(-0.25, 0.25, emdedding_size)
    print("Word '" + word + "'" + ' added to model.')
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)

print('\nTraining LSTM...')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word) for word in text.lower().split()]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature=0.7)
    word_idxs.append(idx)
  return ' '.join(idx2word(idx) for idx in word_idxs)

def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  texts = [
    'This is a',
    'so why is',
    'a',
  ]
  for text in texts:
    sample = generate_next(text)
    print('%s... -> %s' % (text, sample))

model.fit(train_x, train_y,
          batch_size=128,
          epochs=1,
          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])

x = array([word2idx('a')])

inference_model = Sequential()
inference_model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=model.layers[0].get_weights()))
actication = inference_model.predict(x)

inference_model.add(LSTM(units=emdedding_size, weights=model.layers[1].get_weights()))
actication = inference_model.predict(x)

inference_model.add(Dense(units=vocab_size, weights=model.layers[2].get_weights()))
actication = inference_model.predict(x)

hey = 2
# model2.add(Dense(20, 64, weights=model.layers[0].get_weights()))
#model2.add(Activation('tanh'))

# activations = model2._predict(train_x)


from keras import backend as K

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

# Testing
test = random.random(300) # [newaxis,:]
layer_outs = [func([input_word, 1.]) for func in functors]
a = model.summary()

from keras import backend as K

# with a Sequential model
aa = []
for i in range(3):
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[i].output])
    input_word = word2idx('a')
    input_word = random.rand(32)
    layer_output = get_3rd_layer_output([input_word])[0]
    aa.append(layer_output)

hey = 2
import spacy

# spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')

#input = 'I we are Apple is looking at buying U.K. startup for $1 billion'
input = ' '
doc = nlp(u''.join(input))

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)
####
import platform
import subprocess
from ctypes import windll, c_int, byref
from sty import bg, RgbBg

neuron_activation = 0.7
bg.set_style('activationColor', RgbBg(0, 0, int(abs(neuron_activation) * 255)))

colored_word = bg.activationColor + 'hey' + bg.rs
from colorama import init as colorama_init
colored_word_succ ='\x1b[6;30;42m' + 'Success!' + '\x1b[0m' # bg.activationColor + 'hey' + bg.rs

print(colored_word)
# Allowing ANSI Escape Sequences for colors
if platform.system().lower() == 'windows':
    stdout_handle = windll.kernel32.GetStdHandle(c_int(-11))
    mode = c_int(0)
    windll.kernel32.GetConsoleMode(c_int(stdout_handle), byref(mode))
    mode = c_int(mode.value | 4)
    windll.kernel32.SetConsoleMode(c_int(stdout_handle), mode)
'''
'''


print(colored_word)
from colr import color

print(color('Hello there.', fore=(255, 0, 0), back=(255, 0, 0)))
import spacy

print(colored_word)

subprocess.call('', shell=True)
print(colored_word)
print(color('Hello world.', back=(50,0,0)))
'''
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

a = 3.5
X = np.arange(-a, a, 0.25)
Y = np.arange(-a, a, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
shift=0 # 0.5
b = 100*(sigmoid(X+shift) * sigmoid(shift-X) * sigmoid(Y+shift) * sigmoid(shift-Y))
c = 4#np.amax(b)
Z = b/c - np.mean(b/c)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

x_vec = [-3, -3, 3, 3]
y_vec = [0, 0, 0, 0]
z = np.atleast_2d([0, 1, 0, 1])
xx, yy = np.meshgrid(x_vec, y_vec)
point = np.array([0, 0, 0])
normal = np.array([1e-4, 1, 1e-4])
d = -point.dot(normal)

# calculate corresponding z
zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

xs = np.linspace(-3, 3, 100)
zs = np.linspace(0, 1, 100)

X, Z = np.meshgrid(xs, zs)
Y = 0*X

ax.plot_surface(X, Y, Z, alpha=.2, color='black')
cset = ax.contour(X, Y, Z, zdir='x', offset=3, colors='black', linestyles='dashed', linewidths=2)
cset = ax.contour(X, Y, Z, zdir='x', offset=-3, colors='black', linestyles='dashed', linewidths=2)
cset = ax.contour(X, Y, Z, zdir='z', offset=-1, colors='black', linestyles='dashed', linewidths=2)
cset = ax.contour(X, Y, Z, zdir='z', offset=1, colors='black', linestyles='dashed', linewidths=2)

#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.title('Sigmod')
#ax.grid(False)
# Hide axes ticks
#ax.set_xticks([])
#ax.set_zticks([])

#intervals = range(-3, 3)
#plt.yticks(intervals, (range(-6, 6, 2)))


#for y in range(3, (3 - 1) * 3 + 1, 3):
#    x_vec = [1, np.size(X, 1) - 2]
#    y_vec = np.array([y, y]) - .5
#    z_vec = [0, 0]
#    plt.plot(x_vec, y_vec, z_vec, color='black', LineStyle='dashed', LineWidth=2)

# ax.set_yticks([])
#plt.axis('off')
#plt.grid(b=None)
#fig, axes = plt.subplots(nrows=2, sharex=True)

#ax = axes[0]
'''
divider = make_axes_locatable(ax.yaxis)
ax2 = divider.new_vertical(size="100%", pad=0.1)
fig.add_axes(ax2)

ax.scatter(X, Y)
ax.set_ylim(0, 1)
ax.spines['top'].set_visible(False)
ax2.scatter(X, Y)
ax2.set_ylim(10, 100)
ax2.tick_params(bottom="off", labelbottom='off')
ax2.spines['bottom'].set_visible(False)


# From https://matplotlib.org/examples/pylab_examples/broken_axis.html
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


'''
'''
'''
#ax.view_init(17, -126)
plt.show()

'''
print('saving')
plt.savefig('f5.png', transparent=True)

'''
'''

import numpy as np
from viznet import connecta2a, node_sequence, NodeBrush, EdgeBrush, DynamicShow


def draw_feed_forward(ax, num_node_list):

    num_hidden_layer = len(num_node_list) - 2
    token_list = ['\sigma^z'] + \
        ['y^{(%s)}' % (i + 1) for i in range(num_hidden_layer)] + ['\psi']
    kind_list = ['nn.input'] + ['nn.input'] + ['nn.input'] #+ ['nn.hidden'] * num_hidden_layer + ['nn.recurrent']
    radius_list = [.5] + [.5] + [.5]#[1.2] + [0.035] * num_hidden_layer + [0.7]
    y_list = 1 * np.arange(len(num_node_list))

    seq_list = []
    for n, kind, radius, y in zip(num_node_list, kind_list, radius_list, y_list):
        b = NodeBrush(kind, ax)
        seq_list.append(node_sequence(b, n, center=(0, y)))

    eb = EdgeBrush('-->', ax)
    for st, et in zip(seq_list[:-1], seq_list[1:]):
        connecta2a(st, et, eb)


def real_bp():
    with DynamicShow((6, 6), '_feed_forward.png') as d:
        draw_feed_forward(d.ax, num_node_list=[3, 1, 1])


if __name__ == '__main__':
    real_bp()

'''
'''
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

Fs = 1
T = 1/Fs
L = 1500
t = arange(0, L-1)*T
S = zeros((10, len(t)))
for i in range(10):
    S[i, :] = sin(2*pi*.05*i*t)
# S = array([sin(2*pi*.5*t), random.randn(2*pi*.5*t)]) # + sin(2*pi*120*t);
X = S  # + 2*random.randn(size(t));

f = fft.fft(X)
P2 = abs(f/L)
# P1 = P2[1:1000/2+1]
P1 = P2[:, 0:int(L/2)]
P1[:, 2:-2] = 2*P1[:, 2:-2]

#freq = Fs*(0:(L/2))/L;
freq = Fs*arange(0,L/2)/L
neurons = arange(0, 10)
neurons, freq = meshgrid(neurons, freq)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(40, 70)
surf = ax.plot_surface(neurons, freq, P1.T, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# max_activation = amax(neuron_activation_map)
# min_activation = amin(neuron_activation_map)
fig.colorbar(surf, shrink=0.5, aspect='auto')#, vmin=min_activation, vmax=max_activation

plt.title('Fourier Amplitude Spectrum of Neuron Activation')
plt.xlabel('Frequency (/sequence time step)')
plt.ylabel('Neurons of interest')
plt.show()
'''
