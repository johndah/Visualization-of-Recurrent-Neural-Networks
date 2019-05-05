'''
@author: John Henry Dahlberg

2019-02-22
'''

from __future__ import print_function
import os
import pickle
import warnings
import platform
from sty import bg, RgbBg
from decimal import Decimal
from numpy import *
from copy import copy, deepcopy

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import nltk
import gensim
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA

from ctypes import windll, c_int, byref
import re
import zipfile
import lxml.etree
from terminaltables import SingleTable

import tensorflow as tf

import keras.backend as K
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, Dropout, Dense, Flatten, Activation
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

        self.pos_features = {
            'VB': 'Verb',
            'VBG': 'Verb, Gerund',
            'NN': 'Noun',
            'NNS': 'Noun, Plural',
            'NNP': 'Proper Noun',
            'NNP': 'Noun, Plural',
            'POS': 'Possessive Ending',
            'PRP': 'Personal Pronoun',
            'PRP$': 'Possessive Pronoun',
            'JJ': 'Adjective',
            'JJS': 'Adjective, Superlative'
        }

        if self.word_domain:

            if self.save_sentences:
                self.vocabulary, self.sentences, self.K = self.load_vocabulary()

                with open('./Data/vocabulary.word2VecKeyedVector', 'wb') as file:
                    pickle.dump(self.vocabulary, file)

                with open('./Data/sentences.list', 'wb') as file:
                    pickle.dump(self.sentences, file)

                with open('./Data/K.int', 'wb') as file:
                    pickle.dump(self.K, file)

            elif self.load_sentences:
                with open('./Data/vocabulary.word2VecKeyedVector', 'rb') as file:
                    self.vocabulary = pickle.load(file)

                with open('./Data/sentences.list', 'rb') as file:
                    self.sentences = pickle.load(file)

                with open('./Data/K.int', 'rb') as file:
                    self.K = pickle.load(file)
            else:
                self.vocabulary, self.sentences, self.K = self.load_vocabulary()

            if self.train_lstm_model:
                n_validation = int(len(self.sentences) * self.validation_proportion)
                n_training = len(self.sentences) - n_validation

                if self.shuffle_data_sets:
                    random.shuffle(self.sentences)
                self.input_sequence = self.sentences[:n_training]
                self.input_sequence_validation = self.sentences[n_training:]
            else:
                self.input_sequence = self.sentences
                self.input_sequence_validation = self.sentences

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
                         + '\n#' + 'Sample sequences in corpus:' + '\n' + str(len(self.sentences)) \
                         + '\n' + 'Proportion validation set: ' + str(self.validation_proportion)

        self.seq_iterations = []
        self.losses = []
        self.validation_losses = []
        self.seq_iteration = 0
        self.initial_epoch = 0

        if self.load_lstm_model:
            try:
                self.seq_iterations = [i for i in loadtxt('Parameters/seqIterations.txt', delimiter=",", unpack=False)]
                self.losses = [i for i in loadtxt('Parameters/losses.txt', delimiter=",", unpack=False)]
                self.validation_losses = [i for i in
                                          loadtxt('Parameters/validation_losses.txt', delimiter=",", unpack=False)]
                self.seq_iteration = self.seq_iterations[-1]
            # except AttributeError:
            except TypeError:
                warnings.warn(
                    'Text file arrays in the /Parameter/ folder needs to have at least two elements to plot..')

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

        # self.words_to_indices = dict((word, index) for index, word in enumerate(words_common))
        # self.indices_to_words = dict((index, word) for index, word in enumerate(words_common))

        sentences = []

        ignored_sentences = 0
        for i in range(len(words) - self.seq_length):
            if not len(set(words[i:i + self.seq_length]).intersection(words_to_ignore)):
                sentences.append(words[i:i + self.seq_length])
                if not self.train_lstm_model:
                    break
            else:
                ignored_sentences += 1

        print('Sample sequences used in corpus: ' + str(len(sentences)) + ' where ' + str(
            ignored_sentences) + ' were ignored')

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
                word2vec_model.save('Word_Embedding_Model/' + self.text_file.split('.')[0].split('/')[-1] + '_' +
                                    self.embedding_model_file.split('/')[-1][:-4] + "_extracted.model")
        else:
            K = 300
            print('Training word embedding model...')
            word2vec_model = gensim.models.Word2Vec(sentences, size=K, min_count=1, window=5, iter=100, sg=1)

            if self.save_embedding_model:
                word2vec_model.save('Word_Embedding_Model/' + self.text_file.split('.')[0].split('/')[-1] + ".model")

        if self.n_words_pca_plot:
            from mpl_toolkits.mplot3d import proj3d
            vocabulary = word2vec_model.wv
            #fig = plt.figure()
            #ax = Axes3D(fig)
            pca = PCA(n_components=2)
            X = word2vec_model[vocabulary.index2entity[:self.n_words_pca_plot]]
            result = pca.fit_transform(X)
            # create a scatter plot of the projection
            plt.title('PCA Projection')
            plt.scatter(result[:, 0], result[:, 1], color='green')
            # ax.view_init(-168, -12)
            words = list(vocabulary.index2entity[:self.n_words_pca_plot])

            for i, word in enumerate(words):
                #x2, y2, _ = proj3d.proj_transform(result[i, 0], result[i, 1], result[i, 2], ax.get_proj())
                x2 = result[i, 0] - 1 - len(word)/4
                y2 = result[i, 1]
                plt.annotate(word, xy=(x2, y2))

            plt.show()

        del word2vec_model

        return vocabulary, sentences[:int(self.corpus_proportion * len(sentences))], K

    def evenly_split(self, items, lengths):
        for i in range(0, len(items) - lengths, lengths):
            yield items[i:i + lengths]

    def run_lstm(self):
        print('\nInitiate LSTM training...')

        if 'RMS' in self.optimizer:
            self.lstm_optimizer = optimizers.RMSprop(lr=self.eta, rho=0.9)
        else:
            self.lstm_optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

        if not self.load_lstm_model:
            print('Initialize new LSTM model...')
            '''
            self.lstm_model = Sequential()
            self.lstm_model.add(Embedding(input_dim=self.M, output_dim=self.K, weights=[self.vocabulary.syn0]))
            for i in range(self.n_hidden_layers-1):
                self.lstm_model.add(LSTM(units=self.n_hidden_neurons, return_sequences=True, return_state=True))
            self.lstm_model.add(LSTM(units=self.n_hidden_neurons, return_state=True))
            if self.dropout > 0:
                self.lstm_model.add(Dropout(self.dropout))
            self.lstm_model.add(Dense(units=self.M))
            self.lstm_model.add(Activation('softmax'))
            '''

            self.lstm_model = self.create_lstm_model(batch_size=self.batch_size)

            self.lstm_model.compile(optimizer=self.lstm_optimizer, loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy'])
        else:
            model_directory = './LSTM Saved Models/Checkpoints/'
            models = os.listdir(model_directory)
            model_accuracies = [float(model.split('val_acc')[1][:5]) for model in models if 'val_acc' in model]
            best_model_index = argmax(model_accuracies)
            model = model_directory + os.listdir(model_directory)[best_model_index]
            print('\nLoading best performing LSTM model ' + model + ' with accuracy ' + str(model_accuracies[best_model_index]) + '...\n')
            self.lstm_model = load_model(model)
            self.n_hidden_neurons = int(model.split('neurons')[1][:3])
            self.n_hidden_layers = int(model.split('layers')[1][0])
            self.batch_size = int(model.split('batch_size-')[1][:3])
            self.dropout = float(model.split('drop')[1][:4])
            self.eta = float(model.split('eta')[1][:8])
            self.initial_epoch = int(model.split('epoch')[1][:3])

        # from keras.utils import plot_model
        # plot_model(self.lstm_model, to_file='model.png')

        if self.word_domain:
            self.domain_specification = 'Words'
        else:
            self.domain_specification = 'Characters'

        self.lstm_model.summary()

        synthesize_text = LambdaCallback(on_epoch_end=self.synthesize_text)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        callbacks = [synthesize_text, early_stopping]

        if self.train_lstm_model and self.save_checkpoints:
            file_path = "./LSTM Saved Models/Checkpoints/val_acc{val_acc:.4f}-val_loss{val_loss:.5f}-loss{loss:.5f}-epoch{epoch:03d}-neurons%d-layers%d-batch_size-%d-drop%.2f" % (
                self.n_hidden_neurons, self.n_hidden_layers, self.batch_size, self.dropout) + '-eta{:.3e}'.format(
                Decimal(self.eta))
            checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True)
            callbacks.append(checkpoint)

        if self.remote_monitoring_ip:
            remote = RemoteMonitor(root='http://' + self.remote_monitoring_ip + ':9000')
            callbacks.append(remote)

        fetch = [tf.assign(self.input, self.lstm_model.input, validate_shape=False)]
        self.lstm_model._function_kwargs = {'fetches': fetch}

        x = zeros((1, self.seq_length - 1))
        for i, entity in enumerate(self.input_sequence[0]):
            if i < len(self.input_sequence[0]) - 1:
                x[0, i] = self.word_to_index(entity)
            else:
                y = atleast_2d(array(self.word_to_index(entity)))

        # Re-create model with batch size 1 to evaluate performance
        self.lstm_model_evaluate = self.create_lstm_model(batch_size=1)
        weights = self.lstm_model.get_weights()
        self.lstm_model_evaluate.set_weights(weights)
        self.lstm_model_evaluate.compile(optimizer=self.lstm_optimizer, loss='sparse_categorical_crossentropy',
                                         metrics=['accuracy'])

        loss = self.lstm_model_evaluate.evaluate(x, y, batch_size=1)[0]

        class InitLog:
            def get(self, attribute):
                if attribute == 'loss' or attribute == 'val_loss':
                    return loss

        if self.train_lstm_model:
            logs = InitLog()
            self.synthesize_text(-1, logs, evaluate=True)

        if self.train_lstm_model:
            self.lstm_model.fit_generator(self.generate_words(self.input_sequence),
                                          steps_per_epoch=int(len(self.input_sequence) / self.batch_size) + 1,
                                          epochs=self.n_epochs,
                                          callbacks=callbacks,
                                          shuffle=False,
                                          initial_epoch=self.initial_epoch,
                                          validation_data=self.generate_words(self.input_sequence_validation),
                                          validation_steps=int(
                                              len(self.input_sequence_validation) / self.batch_size) + 1)
        else:
            for i in range(1):

                weights = self.lstm_model.get_weights()
                self.lstm_model_evaluate.set_weights(weights)
                self.lstm_model_evaluate.compile(optimizer=self.lstm_optimizer, loss='sparse_categorical_crossentropy',
                                                 metrics=['accuracy'])

                loss = self.lstm_model_evaluate.evaluate(x, y, batch_size=1)[0]

                logs = InitLog()
                self.synthesize_text(-1, logs, evaluate=True)
                # input("\nPress Enter to continue...")

    def create_lstm_model(self, batch_size):
        input_layer = Input(batch_shape=(batch_size, self.seq_length - 1,))
        embedding_layer = Embedding(input_dim=self.M, output_dim=self.K, weights=[self.vocabulary.syn0])(input_layer)

        lstm_input = embedding_layer
        for i in range(self.n_hidden_layers - 1):
            lstm_input = LSTM(units=self.n_hidden_neurons, return_sequences=True, return_state=True, dropout=self.dropout)(lstm_input)

        final_lstm_layer, lstm_outputs, lstm_gate_outputs = LSTM(units=self.n_hidden_neurons, return_state=True,
                                                                 stateful=True, dropout=self.dropout)(lstm_input)

        fully_connected_layer = Dense(units=self.M)(final_lstm_layer)

        output_layer = Activation('softmax')(fully_connected_layer)

        # self.lstm_model = Model(inputs=input_layer, outputs=[output_layer, lstm_outputs, lstm_gate_outputs])
        lstm_model = Model(inputs=input_layer, outputs=output_layer)

        return lstm_model

    def generate_words(self, input_sequence):
        batch_index = 0
        if self.shuffle_data_sets:
            random.shuffle(input_sequence)
        while True:
            x = zeros((self.batch_size, self.seq_length - 1), dtype=int32)
            y = zeros(self.batch_size, dtype=int32)

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
        print('Loaded intervals')

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

        # Update weights of evaluating model with batch size 1
        weights = self.lstm_model.get_weights()
        self.lstm_model_evaluate.set_weights(weights)
        self.lstm_model_evaluate.compile(optimizer=self.lstm_optimizer, loss='sparse_categorical_crossentropy',
                                         metrics=['accuracy'])

        prediction_input = input_sentence

        # Generate text sequence
        for t in range(self.length_synthesized_text):

            x = entity_indices[t:t + self.seq_length - 1]
            if self.flip_input:
                x_predict = flip(atleast_2d(x), axis=1)
            else:
                x_predict = atleast_2d(deepcopy(x))

            output = self.lstm_model_evaluate.predict(x=x_predict, batch_size=1)

            activations = self.get_neuron_activations(x_predict)

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

                colored_word = bg.activationColor + prediction_input + bg.rs
                prediction_input = sample

                # activation_color = bg.activationColor
                # reset_color = bg.rs


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

        if self.neurons_of_interest:
            max_activation = amax(neuron_activation_map[self.neurons_of_interest, :])
            min_activation = amin(neuron_activation_map[self.neurons_of_interest, :])
        else:
            max_activation, min_activation = 0, 0

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

        table.table_data[1:] = flip(table.table_data[1:], axis=0)

        inputs = y_n[0]

        neuron_window = []
        if self.plot_fft and not self.train_lstm_model:
            neuron_window = self.plot_fft_neural_activity(neuron_activation_map)

        if neuron_window:
            self.neurons_of_interest = range(neuron_window[0], neuron_window[1])
            self.neurons_of_interest_plot = [i for i in range(neuron_window[0], neuron_window[1])]
            self.neurons_of_interest_plot_intervals = [range(neuron_window[0], neuron_window[1])]

        if self.plot_color_map and not self.train_lstm_model:
            self.plot_neural_activity(inputs, neuron_activation_map)

        if self.plot_process and self.train_lstm_model:
            self.plot_learning_curve()

        if not self.load_lstm_model or (self.load_lstm_model and not evaluate):
            print('\nEpoch: ' + str(epoch + 1) + ', Loss: ' + str(
                '{0:.2f}'.format(self.losses[-1])) + ', Validation loss: ' + str(
                '{0:.2f}'.format(self.validation_losses[-1])) + ', Neuron of interest: ' + str(
                self.neurons_of_interest) + '(/' + str(
                self.n_hidden_neurons) + ')')

        # Allowing ANSI Escape Sequences for colors
        if platform.system().lower() == 'windows':
            stdout_handle = windll.kernel32.GetStdHandle(c_int(-11))
            mode = c_int(0)
            windll.kernel32.GetConsoleMode(c_int(stdout_handle), byref(mode))
            mode = c_int(mode.value | 4)
            windll.kernel32.SetConsoleMode(c_int(stdout_handle), mode)

        import subprocess
        subprocess.call('', shell=True)

        table_file = 'Results/' + str(self.lstm_gate_of_interest) + str(self.inference_iteration) + '.table'

        with open(table_file, 'wb') as file:
            pickle.dump(table.table, file)

        # print(table.table)



        if self.train_lstm_model and self.save_checkpoints:
            savetxt('Parameters/seqIterations.txt', self.seq_iterations, delimiter=',')
            savetxt('Parameters/losses.txt', self.losses, delimiter=',')
            savetxt('Parameters/validation_losses.txt', self.validation_losses, delimiter=',')

    def get_neuron_activations(self, x_predict):

        with open('FeaturesOfInterest.txt', 'r') as f:
            lines = f.readlines()
            lstm_gate_of_interest = []
            for line in lines:
                if 'lstm_gate_of_interest:' in line:
                    lstm_gate_of_interest = line.split(':')[1].split("'")[1]
                    break

        lstm_gate_of_interest = self.lstm_gate_of_interest

        if not lstm_gate_of_interest:
            warnings.warn('lstm_gate_of_interest not properly specified in "FeaturesOfInterest.txt", set to None')
            lstm_gate_of_interest = 'None'

        lstm_layer = self.lstm_model_evaluate.layers[self.lstm_layer_of_interest + 1]

        final_lstm_layer = K.function([self.lstm_model_evaluate.layers[0].input],
                                      [self.lstm_model_evaluate.layers[self.n_hidden_layers + 1].output[0]])

        lstm_cell_outputs = K.function([self.lstm_model_evaluate.layers[0].input],
                                       [self.lstm_model_evaluate.layers[self.n_hidden_layers + 1].output[2]])

        lstm_layer_input = K.function([self.lstm_model_evaluate.layers[0].input],
                                      [self.lstm_model_evaluate.layers[self.n_hidden_layers + 1].input])
        # lstm_final_layer = K.function([self.lstm_model.layers[0].input],
        #                              [self.lstm_model.layers[self.n_hidden_layers + 1].output])

        '''
        '''
        h_tm1 = final_lstm_layer([atleast_2d(x_predict)])[0].T
        c_tm1 = lstm_cell_outputs([atleast_2d(x_predict)])[0].T

        lstm_input = lstm_layer_input([atleast_2d(x_predict)])[0][0, -1, :]

        '''
        kernel_o = K.get_value(lstm_layer.cell.kernel_o)
        x_o = dot(lstm_input, kernel_o)

        o = self.recurrent_activation(x_o + K.dot(h_tm1_o,
                                                  self.recurrent_kernel_o))
        '''
        lstm_hidden_states = K.get_value(lstm_layer.states[0])
        lstm_cell_states = K.get_value(lstm_layer.states[1])  # previous carry state

        if lstm_gate_of_interest == 'input':
            activations = self.get_lstm_input_activations(lstm_input, lstm_layer, lstm_hidden_states)

        elif lstm_gate_of_interest == 'output':
            activations = self.get_lstm_output_activations(lstm_input, lstm_layer, lstm_hidden_states)

        elif lstm_gate_of_interest == 'forget':
            activations = self.get_lstm_forget_activations(lstm_input, lstm_layer, lstm_hidden_states)

        elif lstm_gate_of_interest == 'cell':
            input_gate_activations = self.get_lstm_input_activations(lstm_input, lstm_layer, lstm_hidden_states)
            forget_gate_activations = self.get_lstm_forget_activations(lstm_input, lstm_layer, lstm_hidden_states)
            activations = self.get_lstm_cell_activations(lstm_input, lstm_layer, lstm_hidden_states, lstm_cell_states,
                                                           input_gate_activations, forget_gate_activations)
        else:
            activations = final_lstm_layer([atleast_2d(x_predict)])[0].T

        return activations

    def get_lstm_output_activations(self, lstm_input, lstm_layer, lstm_hidden_states):
        kernel_o = K.get_value(lstm_layer.cell.kernel_o)
        x_o = dot(lstm_input, kernel_o)
        recurrent_kernel_o = K.get_value(lstm_layer.cell.recurrent_kernel_o)

        output = x_o + dot(lstm_hidden_states, recurrent_kernel_o)
        output_tensor = tf.convert_to_tensor(output, dtype=float32)
        o_tensor = lstm_layer.cell.recurrent_activation(output_tensor)

        return K.get_value(o_tensor).T

    def get_lstm_input_activations(self, lstm_input, lstm_layer, lstm_hidden_states):
        kernel_i = K.get_value(lstm_layer.cell.kernel_i)
        x_i = dot(lstm_input, kernel_i)
        recurrent_kernel_i = K.get_value(lstm_layer.cell.recurrent_kernel_i)

        input = x_i + dot(lstm_hidden_states, recurrent_kernel_i)
        input_tensor = tf.convert_to_tensor(input, dtype=float32)
        i_tensor = lstm_layer.cell.recurrent_activation(input_tensor)

        return K.get_value(i_tensor).T

    def get_lstm_forget_activations(self, lstm_input, lstm_layer, lstm_hidden_states):
        kernel_f = K.get_value(lstm_layer.cell.kernel_f)
        x_f = dot(lstm_input, kernel_f)
        recurrent_kernel_f = K.get_value(lstm_layer.cell.recurrent_kernel_f)

        forget_input = x_f + dot(lstm_hidden_states, recurrent_kernel_f)
        forget_input_tensor = tf.convert_to_tensor(forget_input, dtype=float32)
        f_tensor = lstm_layer.cell.recurrent_activation(forget_input_tensor)
        return K.get_value(f_tensor).T

    def get_lstm_cell_activations(self, lstm_input, lstm_layer, lstm_hidden_states, lstm_cell_states, i, f):
        kernel_c = K.get_value(lstm_layer.cell.kernel_c)
        x_c = dot(lstm_input, kernel_c)
        recurrent_kernel_c = K.get_value(lstm_layer.cell.recurrent_kernel_c)

        cell_input = x_c + dot(lstm_hidden_states, recurrent_kernel_c)
        cell_input_tensor = tf.convert_to_tensor(cell_input, dtype=float32)

        cell_bias = f * lstm_cell_states
        cell_bias_tensor = tf.convert_to_tensor(cell_bias, dtype=float32)

        input_tensor = tf.convert_to_tensor(i, dtype=float32)

        c_tensor = cell_bias_tensor + input_tensor * lstm_layer.cell.activation(cell_input_tensor)

        return K.get_value(c_tensor)

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
        plt.plot(self.seq_iterations[1:], self.validation_losses[1:], LineWidth=2, label='Validation')
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
                # doc = self.nlp(u''.join(inputs[i]))
                pos_tag = nltk.pos_tag([inputs[i]])[-1][-1]

                if (feature.isupper() and feature in pos_tag) or (bool(re.fullmatch(r''.join(feature), inputs[i]))):
                    input_indices_of_interest.append(i)
                    if inputs[i] == '\n':
                        inputs[i] = '\\n'
                    inputs_of_interest.append('"' + inputs[i] + '"')
        except Exception as ex:
            print(ex)

        f, axarr = plt.subplots(1, 2, num=1, gridspec_kw={'width_ratios': [5, 1]}, clear=True)
        axarr[0].set_title('Colormap of hidden neuron activations')

        feature_label = 'Feature: "' + self.pos_features.get(feature, feature) + '"'
        if not self.word_domain and (feature == '.' or feature == '\w+|[^\w]'):
            feature_label = 'Feature: ' + '$\it{Any}$'
        x = range(len(inputs_of_interest))
        axarr[0].set_xticks(x)
        axarr[0].set_xlabel('Predicted sequence (' + feature_label + ')')
        axarr[0].set_xticklabels(inputs_of_interest, fontsize=7,
                                 rotation=90 * self.word_domain * (len(feature) > 1) * (len(inputs_of_interest) >= 15))
        axarr[1].set_xticks([])

        y = range(len(self.neurons_of_interest_plot))
        intervals = [
            self.intervals_to_plot[where(self.interval_limits == i)[0][0]] if i in self.interval_limits else ' ' for i
            in self.neurons_of_interest_plot]

        for i in range(len(axarr)):
            axarr[i].set_yticks(y)
            axarr[i].set_yticklabels(flip(intervals), fontsize=7)
            axarr[0].set_ylabel('Neuron')

        neuron_activation_rows = neuron_activation_map[self.neurons_of_interest_plot, :]

        max_activation = amax(neuron_activation_map)
        min_activation = amin(neuron_activation_map)
        input_indices_of_interest_conjugate = list(set(range(len(inputs))) - set(input_indices_of_interest))
        neuron_feature_extracted_map = flip(neuron_activation_rows[:, input_indices_of_interest], axis=0)
        neuron_feature_remaining_map = flip(neuron_activation_rows[:, input_indices_of_interest_conjugate], axis=0)

        colmap = axarr[0].imshow(neuron_feature_extracted_map, cmap='coolwarm', interpolation='nearest', aspect='auto',
                                 vmin=min_activation, vmax=max_activation)
        extracted_mean = array([mean(neuron_feature_extracted_map, axis=1)]).T
        remaining_mean = array([mean(neuron_feature_remaining_map, axis=1)]).T
        diff = abs(extracted_mean - remaining_mean)
        relevance = diff / amax(diff)
        colmap = axarr[1].imshow(relevance, cmap='coolwarm',
                                 interpolation='nearest', aspect='auto', vmin=min_activation, vmax=max_activation)
        '''
        colmap = axarr[1].imshow(
            array([(mean(neuron_feature_extracted_map, axis=1))]).T / array(
                [abs(mean(neuron_activation_rows, axis=1))]).T - mean(neuron_activation_rows, axis=1),
            cmap='coolwarm', interpolation='nearest', aspect='auto', vmin=min_activation, vmax=max_activation)
        '''
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

    def plot_fft_neural_activity(self, neuron_activation_map):

        neuron_activations = neuron_activation_map[self.neurons_of_interest_fft, :]

        # neuron_activations[-1, :] = sin(2 * pi * 0.5 * arange(0, len(neuron_activations[-1, :])))
        fft_neuron_activations_complex = fft.fft(neuron_activations)

        fft_neuron_activations_abs = abs(fft_neuron_activations_complex / self.length_synthesized_text)

        fft_neuron_activations_single_sided = fft_neuron_activations_abs[:, 0:int(self.length_synthesized_text / 2)]
        fft_neuron_activations_single_sided[:, 2:-2] = 2 * fft_neuron_activations_single_sided[:, 2:-2]

        freq = arange(0, floor(self.length_synthesized_text / 2)) / self.length_synthesized_text

        neurons_of_interest_fft = self.neurons_of_interest_fft

        start_neuron_index = self.neurons_of_interest_plot_intervals[0][0]
        neuron_window = [start_neuron_index]*2

        if self.auto_detect_fft_peak:
            reduced_window_size = 10
            domain_relevant_freq = (freq > self.band_width[0]) & (freq < self.band_width[1])
            # freq = freq[domain_relevant_freq]
            domain_relevant_components = fft_neuron_activations_single_sided[:, domain_relevant_freq]
            argmax_row = where(fft_neuron_activations_single_sided == amax(domain_relevant_components))[0][0]
            # fft_neuron_activations_single_sided = domain_relevant_components[start_neuron:end_neuron, :]
            neuron_window[0] += max(argmax_row - int(reduced_window_size/2), 0)
            neuron_window[1] += min(argmax_row + int(reduced_window_size/2+1), size(domain_relevant_components, 0))
            fft_neuron_activations_single_sided = fft_neuron_activations_single_sided[neuron_window[0]-start_neuron_index:neuron_window[1]-start_neuron_index, :]
            neurons_of_interest_fft = range(neuron_window[0], neuron_window[1])
            print('\nAuto-detected FFT periodicity peak in band width interval ' + str(self.band_width) + ':')
            print('Neuron: ' + str(argmax_row))
            print('Value: ' + str(amax(domain_relevant_components)) + '\n')

        neurons_of_interest_fft, freq = meshgrid(neurons_of_interest_fft, freq)

        fig = plt.figure(3)
        plt.clf()
        ax = fig.gca(projection='3d')
        ax.view_init(20, -120)
        surf = ax.plot_surface(freq, neurons_of_interest_fft, fft_neuron_activations_single_sided.T, rstride=1,
                               cstride=1, cmap=cm.coolwarm, linewidth=0,
                               antialiased=False)
        # ax.set_zlim(-1.01, 1.01)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # max_activation = amax(fft_neuron_activations_single_sided)
        # min_activation = amin(fft_neuron_activations_single_sided)
        fig.colorbar(surf, shrink=0.5, aspect='auto')

        plt.title('Fourier Amplitude Spectrum of Neuron Activation')
        plt.xlabel('Frequency (/sequence time step)')
        plt.ylabel('Neurons of interest')

        plt.pause(.1)

        return neuron_window

    def load_neuron_intervals(self):
        self.neurons_of_interest = []
        self.neurons_of_interest_plot = []
        self.neurons_of_interest_plot_intervals = []
        self.neurons_of_interest_fft = []
        self.band_width = []

        with open('config.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('#')[0]
                if 'plot_color_map:' in line:
                    self.plot_color_map = ''.join(line.split()).split(':')[1] == 'True'
                if 'plot_process:' in line:
                    self.plot_process = ''.join(line.split()).split(':')[1] == 'True'
                if 'plot_fft:' in line:
                    self.plot_fft = ''.join(line.split()).split(':')[1] == 'True'
                if 'auto_detect_fft_peak:' in line:
                    self.auto_detect_fft_peak = ''.join(line.split()).split(':')[1] == 'True'

        with open('FeaturesOfInterest.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '#' not in line:
                    if 'Neurons to fourier transform:' in line:
                        line = line.replace('Neurons to fourier transform:', '')
                        intervals = ''.join(line.split()).split(',')
                        self.intervals_to_plot = []
                        self.interval_limits = []
                        self.interval_label_shift = '      '
                        for interval in intervals:
                            if ':' in interval:
                                interval = interval.split(':')
                                interval[0] = str(max(int(interval[0]), 0))
                                interval[-1] = str(min(int(interval[-1]), self.n_hidden_neurons - 1))
                                self.neurons_of_interest_fft.extend(range(int(interval[0]), int(interval[-1]) + 1))
                                if self.auto_detect_fft_peak:
                                    self.neurons_of_interest.extend(range(int(interval[0]), int(interval[-1]) + 1))

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
                                interval = str(min(int(interval), self.n_hidden_neurons - 1))
                                self.neurons_of_interest_fft.append(int(interval))

                                if self.auto_detect_fft_peak:
                                    self.neurons_of_interest.append(int(interval))

                                    self.neurons_of_interest_plot.append(int(interval))
                                    self.neurons_of_interest_plot_intervals.append([int(interval)])
                                    self.intervals_to_plot.append(interval)
                                    self.interval_limits.append(int(interval))

                        if self.auto_detect_fft_peak:
                            self.interval_limits = array(self.interval_limits)

                    if 'Neurons to print:' in line and not self.auto_detect_fft_peak:
                        line = line.replace('Neurons to print:', '')
                        intervals = ''.join(line.split()).split(',')
                        for interval in intervals:
                            if ':' in interval:
                                interval = interval.split(':')
                                interval[0] = str(max(int(interval[0]), 0))
                                interval[-1] = str(min(int(interval[-1]), self.n_hidden_neurons - 1))
                                self.neurons_of_interest.extend(range(int(interval[0]), int(interval[-1]) + 1))
                            else:
                                interval = str(max(int(interval), 0))
                                interval = str(min(int(interval), self.n_hidden_neurons - 1))
                                self.neurons_of_interest.append(int(interval))
                    if 'Neurons for heatmap:' in line and not self.auto_detect_fft_peak:
                        line = line.replace('Neurons for heatmap:', '')
                        intervals = ''.join(line.split()).split(',')
                        self.intervals_to_plot = []
                        self.interval_limits = []
                        self.interval_label_shift = '      '

                        for interval in intervals:
                            if ':' in interval:
                                interval = interval.split(':')
                                interval[0] = str(max(int(interval[0]), 0))
                                interval[-1] = str(min(int(interval[-1]), self.n_hidden_neurons - 1))
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
                                interval = str(min(int(interval), self.n_hidden_neurons - 1))
                                self.neurons_of_interest_plot.append(int(interval))
                                self.neurons_of_interest_plot_intervals.append([int(interval)])
                                self.intervals_to_plot.append(interval)
                                self.interval_limits.append(int(interval))
                        self.interval_limits = array(self.interval_limits)

                    if 'Band width to auto-detect prominent frequency components (Hz):' in line and self.auto_detect_fft_peak:
                        line = line.replace('Band width to auto-detect prominent frequency components (Hz):', '')
                        intervals = ''.join(line.split()).split(',')
                        for interval in intervals:
                            if ':' in interval:
                                interval = interval.split(':')
                                self.band_width.append(max(float(interval[0]), 0))
                                self.band_width.append(min(float(interval[-1]), 0.5))
                            else:
                                warnings.warn(
                                    'Band width in "FeaturesOfInterest.txt" needs to be an interval with limits seperated by ":".')

    def word_to_index(self, word):
        # return self.words_to_indices[word]
        return self.vocabulary.vocab[word].index

    def index_to_word(self, index):
        # return self.indices_to_words[index]
        return self.vocabulary.index2word[index]


def randomize_hyper_parameters(n_configurations, attributes):
    attributes['n_epochs'] = 6
    attributes['patience'] = 1

    for i in range(n_configurations):
        attributes['n_hidden_neurons'] = 32 * int(7 * random.rand() + 16)
        # attributes['n_hidden_layers'] = int(6 * random.rand()) + 1
        attributes['batch_size'] = 32 * int(6 * random.rand() + 7)
        attributes['dropout'] = float(floor(31 * random.rand() + 20) * .01)
        attributes['eta'] = int(8 * random.rand() + 2) * 10 ** int(-4 - 3 * random.rand())
        # attributes['flip_input'] = random.rand() < 0.2

        print('\nn: ' + str(attributes['n_hidden_neurons']))
        print('layers: ' + str(attributes['n_hidden_layers']))
        print('batch_size: ' + str(attributes['batch_size']))
        print('dropout: ' + str(attributes['dropout']))
        print('eta: ' + str(attributes['eta']))

        # print('flip_input: ' + str(attributes['flip_input']) + '\n')

        lstm_vis = VisualizeLSTM(attributes)
        lstm_vis.run_lstm()

        K.clear_session()


def main():
    attributes = {
        'text_file': 'Data/ted_en.zip',  # 'Data/LordOfTheRings.txt'
        'load_lstm_model': True,  # True to load lstm checkpoint model
        'train_lstm_model': False,  # True to train the model, otherwise only inference is applied
        'lstm_layer_of_interest': 1,  # LSTM layer to visualize
        'optimizer': 'RMS Prop',
        'dropout': 0.44,
        'embedding_model_file': 'Word_Embedding_Model/ted_en_glove_840B_300d_extracted.model',
        'word_frequency_threshold': 10,
        'shuffle_data_sets': False,
        'merge_embedding_model_corpus': False,  # Extract the intersection between embedding model and corpus
        'train_embedding_model': False,  # Further train the embedding model
        'save_embedding_model': False,  # Save trained embedding model
        'save_sentences': False,  # Save sentences and vocabulary
        'load_sentences': False,  # Load sentences and vocabulary
        'n_words_pca_plot': 25,
        'word_domain': True,  # True for words, False for characters
        'validation_proportion': .02,  # The proportion of data set used for validation
        'corpus_proportion': 1,  # The proportion of the corpus used for training and validation
        'n_hidden_neurons': 672,  # Number of hidden neurons, 'Auto' equals to word embedding size
        'n_hidden_layers': 1,  # Number of hidden LSTM layers
        'eta': 5e-5,  # Learning rate
        'batch_size': 352,  # Number of sentences for training for each epoch
        'patience': 2,  # Number of epochs to carry on training before early stopping while loss increases
        'n_epochs': 100,  # Total number of epochs, each corresponds to (n book characters)/(seq_length) seq iterations
        'seq_length': 9,  # Sequence length of each sequence iteration
        'flip_input': False,
        'length_synthesized_text': 500,  # Sequence length of each print of text evolution
        'remote_monitoring_ip': 'http://192.168.155.100:9000/',  # Ip for remote monitoring at http://localhost:9000/
        'save_checkpoints': True  # Save best weights with corresponding arrays iterations and smooth loss
    }

    # randomize_hyper_parameters(1000, attributes)

    # lstm_gates_of_interest = ['input', 'forget', 'cell', 'final_layer_output', 'output']

    print('lstm_gates_of_interest: ' + str(lstm_gates_of_interest))

    for inference_iteration in range(30):
        for lstm_gate_of_interest in lstm_gates_of_interest:
            print('Iteration: ' + str(inference_iteration) + '\t Gate: ' + lstm_gate_of_interest)
            try:
                attributes['lstm_gate_of_interest'] = lstm_gate_of_interest
                attributes['inference_iteration'] = inference_iteration
                lstm_vis = VisualizeLSTM(attributes)
                lstm_vis.run_lstm()
            except Exception as e:
                print(e)

    K.clear_session()


if __name__ == '__main__':
    random.seed(1)
    main()
    plt.show()
