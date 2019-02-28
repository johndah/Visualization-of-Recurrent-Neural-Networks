'''
@author: John Henry Dahlberg

2019-02-22
'''

from __future__ import print_function
import sklearn.preprocessing
from numpy import *
from copy import *
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import platform
from sty import bg, RgbBg
from gensim.models import KeyedVectors
from ctypes import windll, c_int, byref
import re
import zipfile
import lxml.etree
from terminaltables import SingleTable
import gensim
import string
from keras.models import Sequential
from keras.models import load_model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import LambdaCallback

class VisualizeLSTM(object):

    def __init__(self, attributes=None):
        if not attributes:
            raise Exception('Dictionary argument "attributes" is required.')

        self.__dict__ = attributes

        # Allowing ANSI Escape Sequences for colors
        if platform.system().lower() == 'windows':
            stdout_handle = windll.kernel32.GetStdHandle(c_int(-11))
            mode = c_int(0)
            windll.kernel32.GetConsoleMode(c_int(stdout_handle), byref(mode))
            mode = c_int(mode.value | 4)
            windll.kernel32.SetConsoleMode(c_int(stdout_handle), mode)

        if self.word_domain:
            self.word2vec_model, self.input_sequence, self.K = self.loadVocabulary()
            weights_word_embedding = self.word2vec_model.wv.syn0
            self.M = size(weights_word_embedding, 0)
        else:
            self.input_sequence, self.charToInd, self.indToChar = self.loadCharacters()
            self.K = len(self.indToChar)

        if self.nHiddenNeurons == 'Auto':
            self.nHiddenNeurons = self.K

    def loadVocabulary(self):
        words = []

        print('Loading text file "' + self.textFile + '"...')
        if self.textFile[-4:] == '.zip':
            with zipfile.ZipFile(self.textFile, 'r') as z:
                doc = lxml.etree.parse(z.open(z.filelist[0].filename, 'r'))
            print('Extracting words...')
            input_text = '\n'.join(doc.xpath('//content/text()'))
            words.extend(re.findall(r"\w+|[^\w]", input_text))
            sentences = list(self.evenlySplit(words, self.seqLength))
        else:
            with open(self.textFile, 'r') as f:
                lines = f.readlines()
                print('Extracting words...')
                for line in lines:
                    words.extend(re.findall(r"\w+|[^\w]", line))
                    words.append('\n')

        if '.' in self.model_file:
            is_binary = self.model_file[-4:] == '.bin'
            print('Loading model "' + self.model_file + '"...')
            word2vec_model = KeyedVectors.load_word2vec_format(self.model_file, binary=is_binary)
            K = size(word2vec_model.vectors, 1)
        else:
            K = 300
            sentences = [words]  # [''.join(words).split()]
            word2vec_model = gensim.models.Word2Vec(sentences, size=K, min_count=1, window=5, iter=10, sg=0)

        return word2vec_model, sentences, K

    def evenlySplit(self, items, lengths):
        for i in range(0, len(items)-lengths, lengths):
            yield items[i:i + lengths]

    def preProcessData(self):
        print('Preprocessing data...')
        x = zeros([self.seqLength, len(self.input_sequence)], dtype=int32)
        y = zeros([len(self.input_sequence)], dtype=int32)

        '''
        input_sequence_indices = zeros([len(self.input_sequence)], dtype=int32)
        for i, entity in enumerate(len(self.input_sequence)):
            if i%int(len(self.input_sequence)/100):
                print(str(int(i/len(input_sequence_indices))) + ' %')
            try:
                input_sequence_indices[i] = self.wordToIndex(entity)
            except KeyError:
                self.word2vec_model[entity] = random.uniform(-0.25, 0.25, self.K)
                input_sequence_indices[i] = self.wordToIndex(entity)
                print("Entity '" + entity + "'" + ' added to model.')
        '''

        for i, sentence in enumerate(self.input_sequence):
            if i % int(len(self.input_sequence) / 100):
                print(str(int(i / len(self.input_sequence))) + ' %')

            for t, entity in enumerate(sentence[:-1]):
                try:
                    x[t, i] = self.wordToIndex(entity)
                except KeyError:
                    self.word2vec_model[entity] = random.uniform(-0.25, 0.25, self.K)
                    x[t, i] = self.wordToIndex(entity)
                    print("Entity '" + entity + "'" + ' added to model.')
            label_entity = sentence[-1]
            try:
                y[i] = self.wordToIndex(label_entity)
            except KeyError:
                self.word2vec_model[label_entity] = random.uniform(-0.25, 0.25, self.K)
                y[i] = self.wordToIndex(label_entity)
                print("Entity '" + entity + "'" + ' added to model.')

        return x, y

    def on_epoch_end(self, _, logs={}):
        self.losses.append(logs.get('loss'))

    def trainLSTM(self):
        print('\nTraining LSTM...')
        if not self.model_init == 'Load':
            self.lstm_model = Sequential()
            self.lstm_model.add(Embedding(input_dim=self.M, output_dim=self.K, weights=[self.word2vec_model.wv.syn0]))
            self.lstm_model.add(LSTM(units=self.K))
            self.lstm_model.add(Dense(units=self.M))
            self.lstm_model.add(Activation('softmax'))
            self.lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        else:
            self.lstm_model = load_model('LSTM/lstm_model.h5')

        if self.word_domain:
            self.domain_specification = 'Words'
        else:
            self.domain_specification = 'Characters'

        self.lstm_model.summary()
        self.losses = []
        for e in range(0, len(self.input_sequence) - self.seqLength*self.batch_size - 1, self.seqLength*self.batch_size):
            x, y = self.getWords(e)
            self.lstm_model.fit(x, y, batch_size=128, epochs=1, callbacks=[LambdaCallback(on_epoch_end=self.on_epoch_end)])

            if self.saveParameters:
                self.lstm_model.save('LSTM/lstm_model.h5')

            table, neuron_activation_map, inputs = self.synthesizeText()

            with open('config.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split('#')[0]
                    if 'plotColorMap:' in line:
                        self.plotColorMap = ''.join(line.split()).split(':')[1] == 'True'
                        break
                    else:
                        self.plotColorMap = False

            if self.plotColorMap:
                self.plotNeuralActivity(inputs, neuron_activation_map)

            a = e%(self.seqLength*self.batch_size)
            b = (self.seqLength*self.batch_size)
            print('\nEpoch: ' + str(int(e/self.seqLength*self.batch_size)) + ', Epoch process: ' + str('{0:.2f}'.format(a/b*100) + '%'
                  + ', Loss: ' + str('{0:.2f}'.format(self.losses[-1])) + ', Neuron of interest: ' +
                  str(self.neuronsOfInterest) + '(/' + str(self.nHiddenNeurons) + ')'))

            print(table)

    def getWords(self, e):

        sentences = [self.input_sequence[e + i*self.seqLength:e + (i+1)*self.seqLength] for i in range(self.batch_size)]

        x = zeros((self.batch_size, self.seqLength), dtype=int32)
        y = zeros((self.batch_size), dtype=int32)

        for i, sentence in enumerate(sentences):
            for t, entity in enumerate(sentence[:-1]):
                try:
                    x[i, t] = self.wordToIndex(entity)
                except KeyError:
                    self.word2vec_model[entity] = random.uniform(-0.25, 0.25, self.K)
                    x[i, t] = self.wordToIndex(entity)
                    print("Entity '" + entity + "'" + ' added to model.')

            label_entity = sentence[-1]

            try:
                y[i] = array([self.wordToIndex(label_entity)])
            except KeyError:
                self.word2vec_model[label_entity] = random.uniform(-0.25, 0.25, self.K)
                y[i] = array([self.wordToIndex(label_entity)])
                print("Entity '" + entity + "'" + ' added to model.')

        return x, y

    def generateWords(self):
        
        #sentences = [self.input_sequence[e + i*self.seqLength:e + (i+1)*self.seqLength] for i in range(self.batch_size)]

        x = zeros((self.batch_size, self.seqLength), dtype=int32)
        y = zeros((self.batch_size), dtype=int32)

        # for i, sentence in enumerate(self.sentences[self.e:self.e+self.batch_size]):
        for i, sentence in enumerate(self.sentences[self.e:self.e+self.batch_size]):
            for t, entity in enumerate(sentence[:-1]):
                try:
                    x[i, t] = self.wordToIndex(entity)
                except KeyError:
                    self.word2vec_model[entity] = random.uniform(-0.25, 0.25, self.K)
                    x[i, t] = self.wordToIndex(entity)
                    print("Entity '" + entity + "'" + ' added to model.')

            label_entity = sentence[-1]

            try:
                y[i] = array([self.wordToIndex(label_entity)])
            except KeyError:
                self.word2vec_model[label_entity] = random.uniform(-0.25, 0.25, self.K)
                y[i] = array([self.wordToIndex(label_entity)])
                print("Entity '" + entity + "'" + ' added to model.')

        return x, y
    
    def synthesizeText(self):

        seqLength = self.lengthSynthesizedText

        self.neuronsOfInterest = []
        self.neuronsOfInterestPlot = []
        self.neuronsOfInterestPlotIntervals = []

        self.loadNeuronIntervals()

        table_data = [['Neuron ' + str(self.neuronsOfInterest[int(i/2)]), ''] if i % 2 == 0 else ['\n', '\n'] for i in range(2*len(self.neuronsOfInterest))]
        table = SingleTable(table_data)
        table.table_data.insert(0, ['Neuron ', 'Predicted sentence '])

        max_width = table.column_max_width(1)

        y_n = [[] for _ in range(len(self.neuronsOfInterest))]
        y = [[] for _ in range(len(self.neuronsOfInterest))]

        neuron_activation_map = zeros((self.nHiddenNeurons, seqLength))

        entity_indices = [self.wordToIndex(' ')]

        for t in range(seqLength):

            import keras.backend as K

            output = self.lstm_model.predict(x=array(entity_indices))

            aaoutput1 = self.lstm_model.predict(x=[3])
            aaoutput2 = self.lstm_model.predict(x=array([range(4)]))
            aaoutput22 = self.lstm_model.predict(x=array([0, 1, 2, 3]))
            aaoutput3 = self.lstm_model.predict(x=array([3]))

            # get_all_layer_outputs = K.function([self.lstm_model.layers[0].input],
             #   [l.output for l in self.lstm_model.layers])
            # all_layers = get_all_layer_outputs([atleast_2d(entity) for entity in entity_indices])

            lstm_layer = K.function([self.lstm_model.layers[0].input], [self.lstm_model.layers[1].output])
            # activations = lstm_layer([atleast_2d(entity) for entity in entity_indices])[0].T
            activations = lstm_layer([atleast_2d(entity_indices[-1])])[0].T

            #inference_model = Sequential()
            #inference_model.add(Embedding(input_dim=self.M, output_dim=self.K, weights=self.lstm_model.layers[0].get_weights()))
            #inference_model.add(LSTM(units=self.K, weights=self.lstm_model.layers[1].get_weights()))
            #activations = inference_model.predict(x=entity_indices).T

            neuron_activation_map[:, t] = activations[:, 0]
            neuronActivations = activations[self.neuronsOfInterest]

            p = self.softmax(output[-1])
            cp = cumsum(p)
            rand = random.uniform()
            diff = cp - rand
            sample_index = [i for i in range(len(diff)) if diff[i] > 0][0]
            entity_indices.append(sample_index)

            sample = self.indexToWord(sample_index)

            for i in range(len(self.neuronsOfInterest)):

                neuronActivation = neuronActivations[i, 0]

                if neuronActivation > 0:
                    bg.set_style('activationColor', RgbBg(int(neuronActivation * 255), 0, 0))
                else:
                    bg.set_style('activationColor', RgbBg(0, 0, int(abs(neuronActivation) * 255)))

                coloredWord = bg.activationColor + sample + bg.rs

                y_n[i].append(sample)
                y[i].append(coloredWord)

        for i in range(len(self.neuronsOfInterest)):
            # sequences.append(''.join(y[i]))

            wrapped_string = ''
            line_width = 0

            for j in range(len(y[i])):
                table_row = y[i][j]
                if '\n' in table_row:
                    wrapped_string += table_row.split('\n')[0] + '\\n' + table_row.split('\n')[1] + '\n'
                    line_width = 0
                    wrapped_string += ' ' * (max_width - line_width) * 0 + '\n'
                else:
                   wrapped_string += ''.join(y[i][j])

                line_width += len(y_n[i][j])
                if line_width > max_width - 10:
                    line_width = 0
                    wrapped_string += ' '*(max_width - line_width)*0 + '\n'

            table.table_data[2*i+1][1] = wrapped_string

        max_activation = amax(neuron_activation_map[self.neuronsOfInterest, :])
        min_activation = amin(neuron_activation_map[self.neuronsOfInterest, :])
        margin = 8
        color_range_width = max_width - len(table.table_data[0][1]) - (len(str(max_activation)) + len(str(min_activation)) + margin)
        color_range = arange(min_activation, max_activation,
                             (max_activation - min_activation) / color_range_width)

        color_range_str = ' '*margin + str(round(min_activation, 1))

        for i in range(color_range_width):

            color_range_value = color_range[i]

            if color_range_value > 0:
                bg.set_style('activationColor', RgbBg(int(color_range_value * 255), 0, 0))
            else:
                bg.set_style('activationColor', RgbBg(0, 0, int(abs(color_range_value) * 255)))

            colored_indicator = bg.activationColor + ' ' + bg.rs

            color_range_str += colored_indicator

        color_range_str += str(round(max_activation, 1))
        table.table_data[0][1] += color_range_str

        return table.table, neuron_activation_map, y_n[0]


    def softmax(self, s):
        exP = exp(s)
        p = exP / exP.sum()

        return p

    def plotNeuralActivity(self, inputs, neuron_activation_map):
        with open('FeaturesOfInterest.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('#')[0]
                if 'Prediction features:' in line:
                    feature = line.split(':')[1].split("'")[1]
                    break
        try:
            input_indices_of_interets = []
            inputs_of_interets = []
            for i in range(len(inputs)):
                if bool(re.fullmatch(r''.join(feature), inputs[i])):
                    input_indices_of_interets.append(i)
                    if inputs[i] == '\n':
                        inputs[i] = '\\n'
                    inputs_of_interets.append('"' + inputs[i] + '"')
        except Exception as ex:
            print(ex)

        f, axarr = plt.subplots(1, 2, num=1, gridspec_kw={'width_ratios': [5, 1]}, clear=True)
        axarr[0].set_title('Colormap of hidden neuron activations')

        feature_label = 'Feature: "' + feature + '"'
        if not self.word_domain and feature == '.':
            feature_label = 'Feature: ' + '$\it{Any}$'
        x = range(len(inputs_of_interets))
        axarr[0].set_xticks(x)
        axarr[0].set_xlabel('Predicted sequence (' + feature_label + ')')
        axarr[0].set_xticklabels(inputs_of_interets, fontsize=7, rotation=90 * self.word_domain)
        axarr[1].set_xticks([])

        y = range(len(self.neuronsOfInterestPlot))
        intervals = [
            self.intervals_to_plot[where(self.interval_limits == i)[0][0]] if i in self.interval_limits else ' ' for i
            in self.neuronsOfInterestPlot]

        for i in range(len(axarr)):
            axarr[i].set_yticks(y)
            axarr[i].set_yticklabels(intervals, fontsize=7)
            axarr[0].set_ylabel('Neuron')

        neuron_activation_rows = neuron_activation_map[self.neuronsOfInterestPlot, :]
        # f = plt.subplot(1, 2)
        # f, (ax1) = plt.subplot(1, 2, 1)
        max_activation = amax(neuron_activation_map)
        min_activation = amin(neuron_activation_map)
        neuron_feature_extracted_map = neuron_activation_rows[:, input_indices_of_interets]
        colmap = axarr[0].imshow(neuron_feature_extracted_map, cmap='coolwarm', interpolation='nearest', aspect='auto',
                                 vmin=min_activation, vmax=max_activation)
        colmap = axarr[1].imshow(
            array([mean(neuron_feature_extracted_map, axis=1)]).T / array([mean(neuron_activation_rows, axis=1)]).T,
            cmap='coolwarm', interpolation='nearest', aspect='auto', vmin=min_activation, vmax=max_activation)
        axarr[1].set_title('Relevance')

        interval = 0
        for i in range(len(self.neuronsOfInterestPlotIntervals) + 1):
            if i > 0:
                limit = self.neuronsOfInterestPlotIntervals[i - 1]
                interval += 1 + limit[-1] - limit[0]
            axarr[0].plot(arange(-.5, len(input_indices_of_interets) + .5),
                          (len(input_indices_of_interets) + 1) * [interval - 0.5], 'k--', LineWidth=1)

        f.colorbar(colmap, ax=axarr.ravel().tolist())

        plt.pause(.1)

    def loadNeuronIntervals(self):
        with open('FeaturesOfInterest.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '#' not in line:
                    if 'Neurons to print:' in line:
                        line = line.replace('Neurons to print:', '')
                        intervals = ''.join(line.split()).split(',')
                        for interval in intervals:
                            if ':' in interval:
                                interval = interval.split(':')
                                interval[0] = str(max(int(interval[0]), 0))
                                interval[-1] = str(min(int(interval[-1]), self.K - 1))
                                self.neuronsOfInterest.extend(range(int(interval[0]), int(interval[-1]) + 1))
                            else:
                                interval = str(max(int(interval), 0))
                                interval = str(min(int(interval), self.K - 1))
                                self.neuronsOfInterest.append(int(interval))
                    if 'Neurons to plot:' in line:
                        line = line.replace('Neurons to plot:', '')
                        intervals = ''.join(line.split()).split(',')
                        self.intervals_to_plot = []
                        self.interval_limits = []
                        self.interval_label_shift = '      '

                        for interval in intervals:
                            if ':' in interval:
                                interval = interval.split(':')
                                interval[0] = str(max(int(interval[0]), 0))
                                interval[-1] = str(min(int(interval[-1]), self.K - 1))
                                self.neuronsOfInterestPlot.extend(range(int(interval[0]), int(interval[-1]) + 1))
                                self.neuronsOfInterestPlotIntervals.append(
                                    range(int(interval[0]), int(interval[-1]) + 1))
                                intermediate_range = [i for i in range(int(interval[0]) + 1, int(interval[-1])) if
                                                      i % 5 == 0]
                                intermediate_range.insert(0, int(interval[0]))
                                intermediate_range.append(int(interval[-1]))
                                intermediate_range_str = [str(i) for i in intermediate_range]
                                intermediate_range_str[-1] += self.interval_label_shift
                                self.intervals_to_plot.extend(intermediate_range_str)
                                self.interval_limits.extend(intermediate_range)
                            else:
                                interval = str(max(int(interval), 0))
                                interval = str(min(int(interval), self.K - 1))
                                self.neuronsOfInterestPlot.append(int(interval))
                                self.neuronsOfInterestPlotIntervals.append([int(interval)])
                                self.intervals_to_plot.append(interval)
                                self.interval_limits.append(int(interval))
                        self.interval_limits = array(self.interval_limits)

    def wordToIndex(self, word):
        return self.word2vec_model.wv.vocab[word].index

    def indexToWord(self, index):
        return self.word2vec_model.wv.index2word[index]

def main():

    attributes = {
        'textFile': 'LordOfTheRings2.txt', # "'Data/ted_en.zip',  # Name of book text file, needs to be longer than lengthSynthesizedTextBest
        'model_file': 'None', #''Data/glove_840B_300d.txt',  # 'Data/glove_short.txt',  #
        'word_domain': True,  # True for words, False for characters
        'model_init': 'Load',  # 'Load' or 'Random'
        'nHiddenNeurons': 'Auto',  # Number of hidden neurons
        'batch_size': 2,
        'nEpochs': 100,  # Total number of epochs, each corresponds to (n book characters)/(seqLength) seq iterations
        'seqLength': 5,  # Sequence length of each sequence iteration
        'lengthSynthesizedText': 10,  # Sequence length of each print of text evolution
        'lengthSynthesizedTextBest': 100,  # Sequence length of final best sequence, requires saveParameters
        'saveParameters': False  # Save best weights with corresponding arrays iterations and smooth loss
    }

    lstm_vis = VisualizeLSTM(attributes)
    lstm_vis.trainLSTM()


if __name__ == '__main__':
    random.seed(1)
    main()
    plt.show()
