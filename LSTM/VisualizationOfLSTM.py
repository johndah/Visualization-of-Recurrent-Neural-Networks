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
            self.vocabulary, sentences, self.K = self.load_vocabulary()

            n_validation = int(len(sentences) * self.validation_proportion)
            n_training = len(sentences) - n_validation

            self.input_sequence = sentences[:n_training]
            self.input_sequence_validation = sentences[n_training:]

            weights_word_embedding = self.vocabulary.syn0
            self.M = size(weights_word_embedding, 0)
        else:
            self.input_sequence, self.charToInd, self.indToChar = self.loadCharacters()
            self.K = len(self.indToChar)

        if self.n_hiddenNeurons == 'Auto':
            self.n_hiddenNeurons = self.K

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
                         + '\n#' + self.domain_specification + ' in training text:' + '\n' + str(
            len(self.input_sequence))

        if self.load_lstm_model:
            self.seq_iterations = [i for i in loadtxt('Parameters/seqIterations.txt', delimiter=",", unpack=False)]
            self.losses = [i for i in loadtxt('Parameters/losses.txt', delimiter=",", unpack=False)]
        else:
            self.seq_iterations = []
            self.losses = []

        self.neurons_of_interest = []
        self.neurons_of_interest_plot = []
        self.neurons_of_interest_plot_intervals = []

    def load_vocabulary(self):
        words = []

        print('Loading text file "' + self.text_file + '"...')
        if self.text_file[-4:] == '.zip':
            with zipfile.ZipFile(self.text_file, 'r') as z:
                doc = lxml.etree.parse(z.open(z.filelist[0].filename, 'r'))
            print('Extracting words...')
            input_text = '\n'.join(doc.xpath('//content/text()'))
            words.extend(re.findall(r"\w+|[^\w]", input_text))
            # words = input_text.split()
            sentences = list(self.evenly_split(words, self.seq_length))
        else:
            with open(self.text_file, 'r') as f:
                lines = f.readlines()
                print('Extracting words...')
                # input_text = '\n'.join(lines)
                # words = input_text.split() # Space edit
                for line in lines:
                    # words.extend(line.split())
                    # words.append('\n')
                    words.extend(re.findall(r"\w+|[^\w]", line))
                    words.append('\n')
                sentences = list(self.evenly_split(words, self.seq_length))  # [''.join(words).split()]

        if '.model' in self.embedding_model_file:
            word2vec_model = gensim.models.Word2Vec.load(self.embedding_model_file)

            if self.train_embedding_model:
                word2vec_model.train(sentences, total_examples=len(sentences), epochs=10)
            if self.save_embedding_model:
                word2vec_model.save('Word_Embedding_Model/' + self.embedding_model_file)

            K = size(word2vec_model.wv.syn0, 1)

        elif '.txt' in self.embedding_model_file or '.bin' in self.embedding_model_file:
            is_binary = self.embedding_model_file[-4:] == '.bin'
            print('Loading model "' + self.embedding_model_file + '"...')
            word2vec_model_loaded = KeyedVectors.load_word2vec_format(self.embedding_model_file, binary=is_binary)
            K = size(word2vec_model_loaded.vectors, 1)
            words_to_add = [' ', '\n']

            word2vec_model = gensim.models.Word2Vec(sentences, size=K, min_count=1, window=5, iter=0, sg=0)
            word2vec_model.wv = word2vec_model_loaded.wv

            print('Training word embedding model with new words ' + str(words_to_add) + '...')
            word2vec_model.build_vocab([words_to_add], update=True)
            word2vec_model.train([words_to_add], total_examples=1, epochs=1)

            '''
            print('Searching for corpus words not in model...')
            sigma = std(word2vec_model.wv.syn0)
            for word in words_to_add:
                try:
                    word2vec_model.wv.vocab[word]
                except KeyError:
                    word2vec_model[word] = random.uniform(-sigma, sigma, K)
                    print("Entity '" + word + "'" + ' added to model.')
                    # word2vec_model.save("Data/glove_840B_300d_extended.txt")
            '''
        else:
            K = 300
            print('Training word embedding model...')
            word2vec_model = gensim.models.Word2Vec(sentences, size=K, min_count=1, window=5, iter=100, sg=1)

            if self.save_embedding_model:
                word2vec_model.save('Word_Embedding_Model/' + self.text_file.split('.')[0].split('/')[-1] + ".model")

        vocabulary = word2vec_model.wv
        del word2vec_model

        return vocabulary, sentences, K

    def evenly_split(self, items, lengths):
        for i in range(0, len(items) - lengths, lengths):
            yield items[i:i + lengths]

    def train_lstm(self):
        print('\nInitiate LSTM training...')

        if not self.load_lstm_model:
            print('Initiate new LSTM model...')
            self.lstm_model = Sequential()
            self.lstm_model.add(Embedding(input_dim=self.M, output_dim=self.K, weights=[self.vocabulary.syn0]))
            self.lstm_model.add(LSTM(units=self.K))
            self.lstm_model.add(Dense(units=self.M))
            self.lstm_model.add(Activation('softmax'))
            adam_optimizer = optimizers.Adam(lr=self.eta)
            self.lstm_model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy'])
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

        synthesize_text = LambdaCallback(on_epoch_end=self.synthesize_text)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        callbacks = [synthesize_text, early_stopping]

        if self.save_checkpoints:
            file_path = "./LSTM Saved Models/Checkpoints/epoch{epoch:03d}-sequence%d-loss{loss:.4f}-val_loss{val_loss:.4f}" % (
                self.seq_length)
            checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True)
            callbacks.append(checkpoint)

        if self.remote_monitoring_ip:
            remote = RemoteMonitor(root='http://' + self.remote_monitoring_ip + ':9000')
            callbacks.append(remote)

        fetch = [tf.assign(self.input, self.lstm_model.input, validate_shape=False)]
        self.lstm_model._function_kwargs = {'fetches': fetch}

        x = zeros((1, self.seq_length - 1))
        for i, entity in enumerate(self.input_sequence_validation[0]):
            if i < len(self.input_sequence_validation[0]) - 1:
                x[0, i] = self.word_to_index(entity)
            else:
                y = atleast_2d(self.word_to_index(entity))

        loss = self.lstm_model.evaluate(x, y)[0]

        class InitLog:
            def get(self, attribute):
                if attribute == 'loss':
                    return loss

        logs = InitLog()
        self.synthesize_text(0, logs)

        self.lstm_model.fit_generator(self.generate_words(self.input_sequence),
                                      steps_per_epoch=int(len(self.input_sequence) / self.batch_size) + 1,
                                      epochs=self.n_epochs,
                                      callbacks=callbacks,
                                      validation_data=self.generate_words(self.input_sequence_validation),
                                      validation_steps=int(len(self.input_sequence_validation) / self.batch_size) + 1)

    def generate_words(self, input_sequence):
        batch_index = 0
        random.shuffle(input_sequence)
        while True:
            x = zeros((self.batch_size, self.seq_length - 1), dtype=int32)
            y = zeros((self.batch_size), dtype=int32)

            for i, sentence in enumerate(input_sequence[batch_index:batch_index + self.batch_size]):
                for t, entity in enumerate(sentence[:-1]):
                    try:
                        x[i, t] = self.word_to_index(entity)
                    except KeyError:
                        entity = '-'
                        x[i, t] = self.word_to_index(entity)

                label_entity = sentence[-1]

                try:
                    y[i] = array([self.word_to_index(label_entity)])
                except KeyError:
                    label_entity = '-'
                    y[i] = array([self.word_to_index(label_entity)])

            batch_index += self.batch_size

            if batch_index >= len(input_sequence) - 1:
                batch_index = 0

            yield x, y

    def synthesize_text(self, epoch, logs={}):
        print('Generating text sequence...')
        self.losses.append(logs.get('loss'))
        self.seq_iterations.append(epoch)

        self.load_neuron_intervals()

        table_data = [['Neuron ' + str(self.neurons_of_interest[int(i / 2)]), ''] if i % 2 == 0 else ['\n', '\n'] for i
                      in range(2 * len(self.neurons_of_interest))]
        table = SingleTable(table_data)
        table.table_data.insert(0, ['Neuron ', 'Predicted sentence '])

        max_width = table.column_max_width(1)

        y_n = [[] for _ in range(len(self.neurons_of_interest))]
        y = [[] for _ in range(len(self.neurons_of_interest))]

        neuron_activation_map = zeros((self.n_hiddenNeurons, self.length_synthesized_text))

        input = atleast_2d(K.eval(self.input)[0, :])

        entity_indices = zeros(self.seq_length - 1 + self.length_synthesized_text)
        entity_indices[:self.seq_length - 1] = atleast_2d(input[0, :])

        input_sentence = ''
        for i in range(len(input[0, :])):
            input_sentence += self.index_to_word(int(input[0, i]))

        print('Input sentence: ' + input_sentence)

        for t in range(self.length_synthesized_text):

            x = entity_indices[t:t + self.seq_length]
            output = self.lstm_model.predict(x=atleast_2d(x))
            lstm_layer = K.function([self.lstm_model.layers[0].input], [self.lstm_model.layers[1].output])
            activations = lstm_layer([atleast_2d(x)])[0].T

            neuron_activation_map[:, t] = activations[:, 0]
            neuron_activations = activations[self.neurons_of_interest]

            cp = cumsum(output)
            rand = random.uniform()
            diff = cp - rand
            sample_index = [i for i in range(len(diff)) if diff[i] > 0]

            if sample_index:
                sample_index = sample_index[0]
            else:
                sample_index = len(diff) - 1

            sample = self.index_to_word(sample_index)
            entity_indices[t + self.seq_length - 1] = sample_index

            for i in range(len(self.neurons_of_interest)):

                neuron_activation = neuron_activations[i, 0]

                if neuron_activation > 0:
                    bg.set_style('activationColor', RgbBg(int(neuron_activation * 255), 0, 0))
                else:
                    bg.set_style('activationColor', RgbBg(0, 0, int(abs(neuron_activation) * 255)))

                colored_word = bg.activationColor + sample + bg.rs

                y_n[i].append(sample)
                y[i].append(colored_word)

        for i in range(len(self.neurons_of_interest)):

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
                    wrapped_string += ' ' * (max_width - line_width) * 0 + '\n'

            table.table_data[2 * i + 1][1] = wrapped_string

        max_activation = amax(neuron_activation_map[self.neurons_of_interest, :])
        min_activation = amin(neuron_activation_map[self.neurons_of_interest, :])
        margin = 8
        color_range_width = max_width - len(table.table_data[0][1]) - (
                    len(str(max_activation)) + len(str(min_activation)) + margin)
        color_range = arange(min_activation, max_activation,
                             (max_activation - min_activation) / color_range_width)

        color_range_str = ' ' * margin + str(round(min_activation, 1))

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
                    # break
                else:
                    self.plot_process = False

        hey = []
        for word in self.input_sequence[0]:
            hey.append(self.word_to_index(word))
        if self.plot_color_map:
            self.plot_neural_activity(inputs, neuron_activation_map)

        if self.plot_process:
            self.plot_learning_curve()

        # a = e%(self.seq_length*self.batch_size)
        # b = (self.seq_length*self.batch_size)
        # ', Epoch process: ' + str('{0:.2f}'.format(a/b*100) + '%'
        print('\nEpoch: ' + str(int(e / self.seq_length * self.batch_size)) + ', Loss: ' + str(
            '{0:.2f}'.format(self.losses[-1])) + ', Neuron of interest: ' + str(self.neurons_of_interest) + '(/' + str(
            self.n_hiddenNeurons) + ')')

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
        ex_p = exp(s)
        p = ex_p / ex_p.sum()

        return p

    def plot_learning_curve(self):
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

    def plot_neural_activity(self, inputs, neuron_activation_map):
        with open('FeaturesOfInterest.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('#')[0]
                if 'Prediction features:' in line:
                    feature = line.split(':')[1].split("'")[1]
                    break
        try:
            input_indices_of_interest = []
            inputs_of_interest = []
            for i in range(len(inputs)):
                if bool(re.fullmatch(r''.join(feature), inputs[i])):
                    input_indices_of_interest.append(i)
                    if inputs[i] == '\n':
                        inputs[i] = '\\n'
                    inputs_of_interest.append('"' + inputs[i] + '"')
        except Exception as ex:
            print(ex)

        f, axarr = plt.subplots(1, 2, num=1, gridspec_kw={'width_ratios': [5, 1]}, clear=True)
        axarr[0].set_title('Colormap of hidden neuron activations')

        feature_label = 'Feature: "' + feature + '"'
        if not self.word_domain and (feature == '.' or feature == '\w+|[^\w]'):
            feature_label = 'Feature: ' + '$\it{Any}$'
        x = range(len(inputs_of_interest))
        axarr[0].set_xticks(x)
        axarr[0].set_xlabel('Predicted sequence (' + feature_label + ')')
        axarr[0].set_xticklabels(inputs_of_interest, fontsize=7, rotation=90 * self.word_domain * (len(feature) > 3))
        axarr[1].set_xticks([])

        y = range(len(self.neurons_of_interest_plot))
        intervals = [
            self.intervals_to_plot[where(self.interval_limits == i)[0][0]] if i in self.interval_limits else ' ' for i
            in self.neurons_of_interest_plot]

        for i in range(len(axarr)):
            axarr[i].set_yticks(y)
            axarr[i].set_yticklabels(intervals, fontsize=7)
            axarr[0].set_ylabel('Neuron')

        neuron_activation_rows = neuron_activation_map[self.neurons_of_interest_plot, :]
        # f = plt.subplot(1, 2)
        # f, (ax1) = plt.subplot(1, 2, 1)
        max_activation = amax(neuron_activation_map)
        min_activation = amin(neuron_activation_map)
        neuron_feature_extracted_map = neuron_activation_rows[:, input_indices_of_interest]
        colmap = axarr[0].imshow(neuron_feature_extracted_map, cmap='coolwarm', interpolation='nearest', aspect='auto',
                                 vmin=min_activation, vmax=max_activation)
        colmap = axarr[1].imshow(
            array([mean(neuron_feature_extracted_map, axis=1)]).T / array([mean(neuron_activation_rows, axis=1)]).T - 1,
            cmap='coolwarm', interpolation='nearest', aspect='auto', vmin=min_activation, vmax=max_activation)
        axarr[1].set_title('Relevance')

        interval = 0
        for i in range(len(self.neurons_of_interest_plot_intervals) + 1):
            if i > 0:
                limit = self.neurons_of_interest_plot_intervals[i - 1]
                interval += 1 + limit[-1] - limit[0]
            axarr[0].plot(arange(-.5, len(input_indices_of_interest) + .5),
                          (len(input_indices_of_interest) + 1) * [interval - 0.5], 'k--', LineWidth=1)

        f.colorbar(colmap, ax=axarr.ravel().tolist())

        plt.pause(.1)

    def load_neuron_intervals(self):
        self.neurons_of_interest = []
        self.neurons_of_interest_plot = []
        self.neurons_of_interest_plot_intervals = []

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
                                self.neurons_of_interest.extend(range(int(interval[0]), int(interval[-1]) + 1))
                            else:
                                interval = str(max(int(interval), 0))
                                interval = str(min(int(interval), self.K - 1))
                                self.neurons_of_interest.append(int(interval))
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
                                self.neurons_of_interest_plot.extend(range(int(interval[0]), int(interval[-1]) + 1))
                                self.neurons_of_interest_plot_intervals.append(
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
                                self.neurons_of_interest_plot.append(int(interval))
                                self.neurons_of_interest_plot_intervals.append([int(interval)])
                                self.intervals_to_plot.append(interval)
                                self.interval_limits.append(int(interval))
                        self.interval_limits = array(self.interval_limits)

    def word_to_index(self, word):
        return self.vocabulary.vocab[word].index

    def index_to_word(self, index):
        return self.vocabulary.index2word[index]


def main():
    attributes = {
        'text_file': 'Data/ted_en.zip',  # 'Data/LordOfTheRings2.txt',  #
        'load_lstm_model': False,  # True to load lstm checkpoint model
        'embedding_model_file': 'Word_Embedding_Model/ted_talks_word2vec.model',  # 'Data/glove_840B_300d.txt'
        'train_embedding_model': False,  # Further train the embedding model
        'save_embedding_model': False,  # Save trained embedding model
        'word_domain': True,  # True for words, False for characters
        'validation_proportion': .1,  # The proportion of data set used for validation
        'n_hiddenNeurons': 'Auto',  # Number of hidden neurons, 'Auto' equals to word embedding size
        'eta': 1e-3,  # Learning rate
        'batch_size': 10,  # Number of sentences for training for each epoch
        'n_epochs': 100,  # Total number of epochs, each corresponds to (n book characters)/(seq_length) seq iterations
        'seq_length': 10,  # Sequence length of each sequence iteration
        'length_synthesized_text': 50,  # Sequence length of each print of text evolution
        'remote_monitoring_ip': '',  # Ip for remote monitoring at http://localhost:9000/
        'save_checkpoints': False  # Save best weights with corresponding arrays iterations and smooth loss
    }

    lstm_vis = VisualizeLSTM(attributes)
    lstm_vis.train_lstm()


if __name__ == '__main__':
    random.seed(1)
    main()
    plt.show()
