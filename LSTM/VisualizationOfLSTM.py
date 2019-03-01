'''
@author: John Henry Dahlberg

2019-02-22
'''

from __future__ import print_function
import os
import platform
from sty import bg, RgbBg

from numpy import *
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import gensim
from gensim.models import KeyedVectors

from ctypes import windll, c_int, byref
import re
import zipfile
import lxml.etree
from terminaltables import SingleTable

import tensorflow as tf
from keras import optimizers
import keras.backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint, RemoteMonitor


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
            self.word2vec_model, self.words, self.input_sequence, self.K = self.loadVocabulary()
            weights_word_embedding = self.word2vec_model.wv.syn0
            self.M = size(weights_word_embedding, 0)
        else:
            self.input_sequence, self.charToInd, self.indToChar = self.loadCharacters()
            self.K = len(self.indToChar)

        if self.n_hiddenNeurons == 'Auto':
            self.n_hiddenNeurons = self.K

        self.e = 0
        self.input = tf.Variable(0., validate_shape=False)

        if self.word_domain:
            self.domain_specification = 'Words'
        else:
            self.domain_specification = 'Characters'

        self.constants = '# Hidden neurons: ' + str(self.n_hiddenNeurons) \
                    + '\nVocabulary size: ' + str(self.M) \
                    + '\nOptimizer: Adams' \
                    + '\n' + r'$\eta$ = ' + "{:.2e}".format(self.eta) \
                    + '\n' + 'Training sequence length: ' + str(self.seq_length) \
                    + '\n' + 'Batch size: ' + str(self.batch_size) \
                    + '\n#' + self.domain_specification + ' in training text:' + '\n' + str(len(self.input_sequence))
        
        if self.load_lstm_model:
            self.seq_iterations = [i for i in loadtxt('Parameters/seqIterations.txt', delimiter=",", unpack=False)]
            self.losses = [i for i in loadtxt('Parameters/losses.txt', delimiter=",", unpack=False)]
        else:
            self.seq_iterations = []
            self.losses = []

    def loadVocabulary(self):
        words = []

        print('Loading text file "' + self.textFile + '"...')
        if self.textFile[-4:] == '.zip':
            with zipfile.ZipFile(self.textFile, 'r') as z:
                doc = lxml.etree.parse(z.open(z.filelist[0].filename, 'r'))
            print('Extracting words...')
            input_text = '\n'.join(doc.xpath('//content/text()'))
            words.extend(re.findall(r"\w+|[^\w]", input_text))
            sentences = list(self.evenlySplit(words, self.seq_length))
        else:
            with open(self.textFile, 'r') as f:
                lines = f.readlines()
                print('Extracting words...')
                for line in lines:
                    words.extend(re.findall(r"\w+|[^\w]", line))
                    words.append('\n')
                sentences = list(self.evenlySplit(words, self.seq_length))  # [''.join(words).split()]

        if '.' in self.embedding_model_file:
            is_binary = self.embedding_model_file[-4:] == '.bin'
            print('Loading model "' + self.embedding_model_file + '"...')
            word2vec_model = KeyedVectors.load_word2vec_format(self.embedding_model_file, binary=is_binary)
            K = size(word2vec_model.vectors, 1)

            print('Searching for corpus words not in model...')
            sigma = std(word2vec_model.wv.syn0)
            for word in words:
                try:
                    word2vec_model.wv.vocab[word]
                except KeyError:
                    word2vec_model[word] = random.uniform(-sigma, sigma, K)
                    print("Entity '" + word + "'" + ' added to model.')
        else:
            K = 300
            word2vec_model = gensim.models.Word2Vec(sentences, size=K, min_count=1, window=5, iter=10, sg=0)

        return word2vec_model, words, sentences, K

    def evenlySplit(self, items, lengths):
        for i in range(0, len(items)-lengths, lengths):
            yield items[i:i + lengths]

    def preProcessData(self):
        print('Preprocessing data...')
        x = zeros([self.seq_length, len(self.input_sequence)], dtype=int32)
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
                    self.M += 1
            label_entity = sentence[-1]
            try:
                y[i] = self.wordToIndex(label_entity)
            except KeyError:
                self.word2vec_model[label_entity] = random.uniform(-0.25, 0.25, self.K)
                y[i] = self.wordToIndex(label_entity)
                print("Entity '" + entity + "'" + ' added to model.')
                self.M += 1

        return x, y

    def trainLSTM(self):
        print('\nInitiate LSTM training...')

        if not self.load_lstm_model:
            print('Initiate new LSTM model...')
            self.lstm_model = Sequential()
            self.lstm_model.add(Embedding(input_dim=self.M, output_dim=self.K, weights=[self.word2vec_model.wv.syn0]))
            self.lstm_model.add(LSTM(units=self.K))
            self.lstm_model.add(Dense(units=self.M))
            self.lstm_model.add(Activation('softmax'))
            adam_optimizer = optimizers.Adam(lr=self.eta)
            self.lstm_model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            model_directory = './LSTM Saved Models/Checkpoints/'
            model = model_directory + os.listdir(model_directory)[-1]
            print('Loading LSTM model ' + model + '...')
            self.lstm_model = load_model(model)

        if self.word_domain:
            self.domain_specification = 'Words'
        else:
            self.domain_specification = 'Characters'

        self.lstm_model.summary()

        synthesizeText = LambdaCallback(on_epoch_end=self.synthesizeText)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        remote = RemoteMonitor(root='http://localhost:9000')
        callbacks = [synthesizeText, early_stopping, remote]

        if self.save_checkpoints:
            file_path = "./LSTM Saved Models/Checkpoints/epoch{epoch:03d}-sequence%d-loss{loss:.4f}-val_loss{val_loss:.4f}" % (self.seq_length)
            checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True)
            callbacks.append(checkpoint)

        fetch = [tf.assign(self.input, self.lstm_model.input, validate_shape=False)]
        self.lstm_model._function_kwargs = {'fetches': fetch}


        self.lstm_model.fit_generator(self.generateWords(),
                            steps_per_epoch=int(len(self.input_sequence) / self.batch_size) + 1,
                            epochs=100,
                            callbacks=callbacks,
                            validation_data=self.generateWords(),
                            validation_steps=int(len(self.input_sequence) / self.batch_size) + 1)

        #table, neuron_activation_map, inputs = self.synthesizeText()

    def generateWords(self):

        while True:
            x = zeros((self.batch_size, self.seq_length), dtype=int32)
            y = zeros((self.batch_size), dtype=int32)

            # for i, sentence in enumerate(self.sentences[self.e:self.e+self.batch_size]):
            for i, sentence in enumerate(self.input_sequence[self.e:self.e+self.batch_size]):
                for t, entity in enumerate(sentence[:-1]):
                    try:
                        x[i, t] = self.wordToIndex(entity)
                    except KeyError:
                        #self.word2vec_model[entity] = random.uniform(-0.25, 0.25, self.K)
                        x[i, t] = self.wordToIndex(entity)
                        print("Entity '" + entity + "'" + ' added to model.')

                label_entity = sentence[-1]

                try:
                    y[i] = array([self.wordToIndex(label_entity)])
                except KeyError:
                    self.word2vec_model[label_entity] = random.uniform(-0.25, 0.25, self.K)
                    y[i] = array([self.wordToIndex(label_entity)])
                    print("Entity '" + entity + "'" + ' added to model.')

            self.e += self.batch_size

            if self.e == len(self.input_sequence):
                self.e = 0

            yield x, y
    
    def synthesizeText(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.seq_iterations.append(epoch)

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

        neuron_activation_map = zeros((self.n_hiddenNeurons, self.length_synthesized_text))

        input = atleast_2d(K.eval(self.input))
        entity_indices = zeros(self.length_synthesized_text + self.seq_length)
        entity_indices[:self.seq_length] = atleast_2d(input[-1])

        for t in range(self.length_synthesized_text):

            x = entity_indices[t:t+self.seq_length]
            output = self.lstm_model.predict(x=array(x))
            lstm_layer = K.function([self.lstm_model.layers[0].input], [self.lstm_model.layers[1].output])
            activations = lstm_layer([atleast_2d(x)])[0].T

            neuron_activation_map[:, t] = activations[:, 0]
            neuronActivations = activations[self.neuronsOfInterest]

            cp = cumsum(output)
            rand = random.uniform()
            diff = cp - rand
            sample_index = [i for i in range(len(diff)) if diff[i] > 0][0]
            sample = self.indexToWord(sample_index)
            entity_indices[t + self.seq_length] = sample_index

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

        inputs = y_n[0]

        with open('config.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('#')[0]
                if 'plot_color_map:' in line:
                    self.plot_color_map = ''.join(line.split()).split(':')[1] == 'True'
                    break
                else:
                    self.plot_color_map = False

                if 'plot_process:' in line:
                    self.plot_process = ''.join(line.split()).split(':')[1] == 'True'
                    break
                else:
                    self.plot_process = False

        if self.plot_color_map:
            self.plotNeuralActivity(inputs, neuron_activation_map)

        if self.plot_process:
            self.plotLearningCurve()

        a = e%(self.seq_length*self.batch_size)
        b = (self.seq_length*self.batch_size)
        print('\nEpoch: ' + str(int(e/self.seq_length*self.batch_size)) + ', Epoch process: ' + str('{0:.2f}'.format(a/b*100) + '%'
              + ', Loss: ' + str('{0:.2f}'.format(self.losses[-1])) + ', Neuron of interest: ' +
              str(self.neuronsOfInterest) + '(/' + str(self.n_hiddenNeurons) + ')'))

        print(table.table)

        if self.save_checkpoints:
            savetxt('Parameters/seqIterations.txt', self.seq_iterations, delimiter=',')
            savetxt('Parameters/losses.txt', self.losses, delimiter=',')

    def sequence_contains_sequence(self, haystack_seq, needle_seq):
        for i in range(0, len(haystack_seq) - len(needle_seq) + 1):
            if needle_seq == haystack_seq[i:i + len(needle_seq)]:
                return True, i
        return False, 0

    def softmax(self, s):
        exP = exp(s)
        p = exP / exP.sum()

        return p

    def plotLearningCurve(self):
        fig = plt.figure(2)
        plt.clf()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        anchored_text = AnchoredText(self.constants, loc=1)
        ax.add_artist(anchored_text)

        plt.title(self.domain_specification[:-1] + ' prediction learning curve of LSTM')
        plt.ylabel('Cross-entropy loss')
        plt.xlabel('Epoch')
        plt.plot(self.seq_iterations, self.losses, LineWidth=2)
        plt.grid()

        plt.pause(0.1)

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
        'textFile': 'Data/LordOfTheRings2.txt',  # 'Data/ted_en.zip',  #  Name of book text file, needs to be longer than length_synthesized_textBest
        'embedding_model_file': 'None',  #'Data/glove_short.txt',  # ''Data/glove_840B_300d.txt',  # 'Data/glove_short.txt',  #
        'word_domain': True,  # True for words, False for characters
        'load_lstm_model': True,  # True to load lstm checkpoint model
        'n_hiddenNeurons': 'Auto',  # Number of hidden neurons
        'eta': 1e-3,  # Learning rate
        'batch_size': 3,
        'nEpochs': 100,  # Total number of epochs, each corresponds to (n book characters)/(seq_length) seq iterations
        'seq_length': 5,  # Sequence length of each sequence iteration
        'length_synthesized_text': 10,  # Sequence length of each print of text evolution
        'remote_monitoring': False,  # Remote monitoring of learning curve at http://localhost:9000/
        'save_checkpoints': True  # Save best weights with corresponding arrays iterations and smooth loss
    }

    lstm_vis = VisualizeLSTM(attributes)
    lstm_vis.trainLSTM()


if __name__ == '__main__':
    random.seed(1)
    main()
    plt.show()
