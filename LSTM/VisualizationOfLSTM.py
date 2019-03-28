'''
@author: John Henry Dahlberg

2019-02-22
'''

from __future__ import print_function
import os
import platform
from sty import bg, RgbBg

from numpy import *
from copy import copy, deepcopy
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

import keras.backend as K
from keras import optimizers, regularizers
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding
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
            self.vocabulary, self.sentences, self.K = self.load_vocabulary()

            n_validation = int(len(self.sentences) * self.validation_proportion)
            n_training = len(self.sentences) - n_validation

            random.shuffle(self.sentences)
            self.input_sequence = self.sentences[:n_training]
            self.input_sequence_validation = self.sentences[n_training:]

            weights_word_embedding = self.vocabulary.syn0
            self.M = size(weights_word_embedding, 0)
        else:
            self.input_sequence, self.charToInd, self.indToChar = self.loadCharacters()
            self.K = len(self.indToChar)

        if self.n_hidden_neurons == 'Auto':
            self.n_hidden_neurons = self.K

        self.input = tf.Variable(0., validate_shape=False)

        if self.word_domain:
            self.domain_specification = 'Words'
        else:
            self.domain_specification = 'Characters'

        self.constants = '# Hidden neurons: ' + str(self.n_hidden_neurons) \
                         + '\nVocabulary size: ' + str(self.M) \
                         + '\nOptimizer: ' + self.optimizer \
                         + '\n' + r'$\eta$ = ' + "{:.2e}".format(self.eta) \
                         + '\n' + 'Training sequence length: ' + str(self.seq_length) \
                         + '\n' + 'Batch size: ' + str(self.batch_size) \
                         + '\n#' + 'Sample sequences in corpus:' + '\n' + str(
            len(self.sentences)) \
                         + '\n' + 'Proportion validation set: ' + str(self.validation_proportion)

        if self.load_lstm_model:
            self.seq_iterations = [i for i in loadtxt('Parameters/seqIterations.txt', delimiter=",", unpack=False)]
            self.losses = [i for i in loadtxt('Parameters/losses.txt', delimiter=",", unpack=False)]
            self.validation_losses = [i for i in loadtxt('Parameters/validation_losses.txt', delimiter=",", unpack=False)]
            self.seq_iteration = self.seq_iterations[-1]
        else:
            self.seq_iterations = []
            self.losses = []
            self.validation_losses = []
            self.seq_iteration = 0

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
        else:
            with open(self.text_file, 'r') as f:
                lines = f.readlines()
                print('Extracting words...')
                for line in lines:
                    words.extend(re.findall(r"\w+|[^\w]", line))
                    words.append('\n')

        print('Words in corpus: ' + str(len(words)))
        print('Calculating word frequency and ignoring if fewer than ' + str(self.word_frequency_threshold) + '...')
        # word_frequencies = dict(zip(words, [words.count(word) for word in words]))

        word_frequencies = {}
        for word in words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

        words_to_ignore = set()
        for word, count in word_frequencies.items():
            if count < self.word_frequency_threshold:
                words_to_ignore.add(word)

        print('Unique words in corpus: ' + str(len(set(words))))
        words_common = set(words) - words_to_ignore
        print('Unique words in corpus after ignoring rare ones: ' + str(len(words_common)))

        self.words_to_indices = dict((word, index) for index, word in enumerate(words_common))
        self.indices_to_words = dict((index, word) for index, word in enumerate(words_common))

        sentences = []

        ignored_sentences = 0
        for i in range(len(words) - self.seq_length):
            if not len(set(words[i:i+self.seq_length]).intersection(words_to_ignore)):
                sentences.append(words[i:i+self.seq_length])
            else:
                ignored_sentences += 1

        print('Sample sequences used in corpus: ' + str(len(sentences)) + ' where ' + str(ignored_sentences) + ' were ignored')

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

            if self.merge_embedding_model_corpus and '.txt' in self.embedding_model_file:
                print('Merging embeding model ' + self.embedding_model_file + ' with corpus ' + self.text_file)
                with open(self.embedding_model_file, 'r', encoding="utf8") as f:
                    lines = f.readlines()

                    counter = 0
                    extracted_words = []
                    for line in lines:
                        extracted_words.append(line.split()[0])

                intersection = set(words_common).intersection(set(extracted_words))
                extracted_lines = [str(len(intersection)) + ' 300\n']
                print('Done with reading')
                with open(self.embedding_model_file, 'r', encoding="utf8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.split()[0] in intersection:
                            extracted_lines.append(line)
                print('Size model: ' + str(len(extracted_words)))
                print('Size merge model: ' + str(len(intersection)))
                self.embedding_model_file = self.embedding_model_file.split('/')[-1][:-4] + '_extracted.txt'
                with open(self.embedding_model_file, 'r+', encoding="utf8") as f:
                    f.writelines(extracted_lines)
            elif self.merge_embedding_model_corpus and '.txt' not in self.embedding_model_file:
                print('Merge only allows .txt files! ')

            word2vec_model_loaded = KeyedVectors.load_word2vec_format(self.embedding_model_file, binary=is_binary)
            K = size(word2vec_model_loaded.vectors, 1)
            words_to_add = [' ', '\n']

            word2vec_model = gensim.models.Word2Vec(sentences, size=K, min_count=1, window=5, iter=0, sg=0)
            word2vec_model.wv = word2vec_model_loaded.wv

            print('Training word embedding model with new words ' + str(words_to_add) + '...')
            word2vec_model.build_vocab([words_to_add], update=True)
            word2vec_model.train([words_to_add], total_examples=1, epochs=1)

            print('Training word embedding model with corpus ' + self.text_file + '...')
            word2vec_model.build_vocab(sentences, update=True)
            word2vec_model.train(sentences, total_examples=1, epochs=100)

            if self.save_embedding_model:
                word2vec_model.save('Word_Embedding_Model/' + self.text_file.split('.')[0].split('/')[-1] + '_' + self.embedding_model_file.split('/')[-1][:-4] + "_extracted.model")
        else:
            K = 300
            print('Training word embedding model...')
            word2vec_model = gensim.models.Word2Vec(sentences, size=K, min_count=1, window=5, iter=100, sg=1)

            if self.save_embedding_model:
                word2vec_model.save('Word_Embedding_Model/' + self.text_file.split('.')[0].split('/')[-1] + ".model")

        vocabulary = word2vec_model.wv
        del word2vec_model

        return vocabulary, sentences[:int(self.corpus_proportion*len(sentences))], K

    def evenly_split(self, items, lengths):
        for i in range(0, len(items) - lengths, lengths):
            yield items[i:i + lengths]

    def train_lstm(self):
        print('\nInitiate LSTM training...')

        if not self.load_lstm_model:
            print('Initialize new LSTM model...')
            self.lstm_model = Sequential()
            self.lstm_model.add(Embedding(input_dim=self.M, output_dim=self.K, weights=[self.vocabulary.syn0]))
            for i in range(self.n_hidden_layers-1):
                self.lstm_model.add(LSTM(units=self.n_hidden_neurons, return_sequences=True))
            self.lstm_model.add(LSTM(units=self.n_hidden_neurons))
            if self.dropout > 0:
                self.lstm_model.add(Dropout(self.dropout))
            self.lstm_model.add(Dense(units=self.M))
            self.lstm_model.add(Activation('softmax'))
            if 'RMS' in self.optimizer:
                optimizer = optimizers.RMSprop(lr=self.eta, rho=0.9)
            else:
                optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

            self.lstm_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
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
            file_path = "./LSTM Saved Models/Checkpoints/val_loss{val_loss:.4f}-loss{loss:.4f}-epoch{epoch:03d}-neurons%d-layers%d-batch_size-%d-drop%f-eta-%f-flip%d"%(
                self.n_hidden_neurons, self.n_hidden_layers, self.batch_size, self.dropout, self.eta, self.flip_input)
            checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True)
            callbacks.append(checkpoint)

        if self.remote_monitoring_ip:
            remote = RemoteMonitor(root='http://' + self.remote_monitoring_ip + ':9000')
            callbacks.append(remote)

        fetch = [tf.assign(self.input, self.lstm_model.input, validate_shape=False)]
        self.lstm_model._function_kwargs = {'fetches': fetch}

        x = zeros((1, self.seq_length - 1))
        for i, entity in enumerate(self.input_sequence[0]):
            if i < len(self.input_sequence_validation[0]) - 1:
                x[0, i] = self.word_to_index(entity)
            else:
                y = atleast_2d(self.word_to_index(entity))

        loss = self.lstm_model.evaluate(x, y)[0]

        class InitLog:
            def get(self, attribute):
                if attribute == 'loss' or attribute == 'val_loss':
                    return loss

        logs = InitLog()
        self.synthesize_text(-1, logs, evaluate=True)

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
                        print('Entity "' + entity + '"' + ' not in model, replaced with "-".')
                        x[i, t] = self.word_to_index(entity)

                label_entity = sentence[-1]
                if self.flip_input:
                    x = flip(x, axis=1)

                try:
                    y[i] = array([self.word_to_index(label_entity)])
                except KeyError:
                    label_entity = '-'
                    print('Label "' + label_entity + '"' + ' not in model, replaced with "-".')
                    y[i] = array([self.word_to_index(label_entity)])

            batch_index += self.batch_size

            if batch_index >= len(input_sequence) - 1:
                batch_index = 0

            yield x, y

    def synthesize_text(self, epoch, logs={}, evaluate=False):
        print('Generating text sequence...')
        if not self.load_lstm_model or (self.load_lstm_model and not evaluate):
            self.losses.append(logs.get('loss'))
            self.validation_losses.append(logs.get('val_loss'))
            self.seq_iterations.append(self.seq_iteration)
            self.seq_iteration += 1

        self.load_neuron_intervals()

        table_data = [['Neuron ' + str(self.neurons_of_interest[int(i / 2)]), ''] if i % 2 == 0 else ['\n', '\n'] for i
                      in range(2 * len(self.neurons_of_interest))]
        table = SingleTable(table_data)
        table.table_data.insert(0, ['Neuron ', 'Predicted sentence '])

        max_width = table.column_max_width(1)

        y_n = [[] for _ in range(len(self.neurons_of_interest))]
        y = [[] for _ in range(len(self.neurons_of_interest))]

        neuron_activation_map = zeros((self.n_hidden_neurons, self.length_synthesized_text))

        # input = atleast_2d(K.eval(self.input)[0, :])
        input_indices = []
        for word in self.input_sequence_validation[0][:-1]:
            input_indices.append(self.word_to_index(word))
        input_indices = atleast_2d(input_indices)
        entity_indices = zeros(self.seq_length - 1 + self.length_synthesized_text)
        entity_indices[:self.seq_length - 1] = atleast_2d(input_indices[0, :])

        input_sentence = ''
        for i in range(len(input_indices[0, :])):
            input_sentence += self.index_to_word(int(input_indices[0, i]))

        print('Input sentence: ' + input_sentence)

        for t in range(self.length_synthesized_text):

            x = entity_indices[t:t + self.seq_length-1]
            if self.flip_input:
                x_predict = flip(atleast_2d(x), axis=1)
            else:
                x_predict = atleast_2d(deepcopy(x))

            output = self.lstm_model.predict(x=x_predict)
            lstm_layer = K.function([self.lstm_model.layers[0].input], [self.lstm_model.layers[self.n_hidden_layers].output])
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

            # sample_index = argmax(output)
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
        try:
            color_range_width = max_width - len(table.table_data[0][1]) - (
                    len(str(max_activation)) + len(str(min_activation)) + margin)
            color_range = arange(min_activation, max_activation,
                                 (max_activation - min_activation) / color_range_width)
        except ValueError:
            color_range_width = 2
            color_range = [min_activation, max_activation]

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

        if self.plot_color_map:
            self.plot_neural_activity(inputs, neuron_activation_map)

        if self.plot_process:
            self.plot_learning_curve()

        if not self.load_lstm_model or (self.load_lstm_model and not evaluate):
            print('\nEpoch: ' + str(epoch+1) + ', Loss: ' + str(
                '{0:.2f}'.format(self.losses[-1])) + ', Validation loss: ' + str(
                '{0:.2f}'.format(self.validation_losses[-1])) + ', Neuron of interest: ' + str(self.neurons_of_interest) + '(/' + str(
                self.n_hidden_neurons) + ')')
        print(table.table)

        if self.save_checkpoints:
            savetxt('Parameters/seqIterations.txt', self.seq_iterations, delimiter=',')
            savetxt('Parameters/losses.txt', self.losses, delimiter=',')
            savetxt('Parameters/validation_losses.txt', self.validation_losses, delimiter=',')

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
        plt.plot(self.seq_iterations[1:], self.losses[1:], LineWidth=2, label='Training')
        plt.plot(self.seq_iterations[1:], self.validation_losses[1:], LineWidth=2,  label='Validation')
        plt.grid()

        plt.legend(loc='upper left')

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
        # return self.words_to_indices[word]
        return self.vocabulary.vocab[word].index

    def index_to_word(self, index):
        # return self.indices_to_words[index]
        return self.vocabulary.index2word[index]


def randomize_hyper_parameters(n_configurations, attributes):

    attributes['n_epochs'] = 3

    for i in range(n_configurations):
        attributes['n_hidden_neurons'] = 128*int(7*random.rand()+1)
        attributes['n_hidden_layers'] = int(6*random.rand()) + 1
        attributes['batch_size'] = 32*int(10*random.rand() + 1)
        attributes['dropout'] = float(floor(30*random.rand() + 10)*.01)
        attributes['eta'] = int(8*random.rand() + 2)*10**int(-3 - 4*random.rand())
        attributes['flip_input'] = random.rand() < 0.2

        print('\nn: ' + str(attributes['n_hidden_neurons']))
        print('layers: ' + str(attributes['n_hidden_layers']))
        print('batch_size: ' + str(attributes['batch_size']))
        print('dropout: ' + str(attributes['dropout']))
        print('eta: ' + str(attributes['eta']))
        print('flip_input: ' + str(attributes['flip_input']) + '\n')

        lstm_vis = VisualizeLSTM(attributes)
        lstm_vis.train_lstm()

        K.clear_session()


def main():
    attributes = {
        'text_file': 'Data/ted_en.zip',  # 'Data/LordOfTheRings2.txt',  #
        'load_lstm_model': False,  # True to load lstm checkpoint model
        'optimizer': 'RMS Prop',
        'dropout': 0.25,
        'embedding_model_file': 'Word_Embedding_Model/ted_en_glove_840B_300d_extracted.model',
        'word_frequency_threshold': 10,
        'merge_embedding_model_corpus': False,  # Extract the intersection between embedding model and corpus
        'train_embedding_model': False,  # Further train the embedding model
        'save_embedding_model': False,  # Save trained embedding model
        'word_domain': True,  # True for words, False for characters
        'validation_proportion': .02,  # The proportion of data set used for validation
        'corpus_proportion': 1,  # The proportion of the corpus used for training and validation
        'n_hidden_neurons': 512,  # Number of hidden neurons, 'Auto' equals to word embedding size
        'n_hidden_layers': 3,  # Number of hidden LSTM layers
        'eta': 3e-4,  # Learning rate
        'batch_size': 224,  # Number of sentences for training for each epoch
        'n_epochs': 100,  # Total number of epochs, each corresponds to (n book characters)/(seq_length) seq iterations
        'seq_length': 9,  # Sequence length of each sequence iteration
        'flip_input': False,
        'length_synthesized_text': 100,  # Sequence length of each print of text evolution
        'remote_monitoring_ip': 'http://192.168.151.148:9000/',  # Ip for remote monitoring at http://localhost:9000/
        'save_checkpoints': True  # Save best weights with corresponding arrays iterations and smooth loss
    }

    # randomize_hyper_parameters(500, attributes)

    lstm_vis = VisualizeLSTM(attributes)
    lstm_vis.train_lstm()

    K.clear_session()

if __name__ == '__main__':
    random.seed(1)
    main()
    plt.show()
