'''
@author: John Henry Dahlberg

@created: 2019-02-22
'''

from __future__ import print_function
import os
import pickle
import warnings
import platform
from sty import bg, RgbBg
from decimal import Decimal
from numpy import *
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

import nltk
import gensim.models
from sklearn.decomposition import PCA

import ctypes
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

        # Allowing ANSI Escape Sequences for terminal coloring for windows. It works on Linux without this.
        if platform.system().lower() == 'windows':
            stdout_handle = ctypes.windll.kernel32.GetStdHandle(ctypes.c_int(-11))
            mode = ctypes.c_int(0)
            ctypes.windll.kernel32.GetConsoleMode(ctypes.c_int(stdout_handle), ctypes.byref(mode))
            mode = ctypes.c_int(mode.value | 4)
            ctypes.windll.kernel32.SetConsoleMode(ctypes.c_int(stdout_handle), mode)

        # Part of speech tags as feature of interest
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

        # Save preprocessed word embedding vocabularies and sentences
        if self.save_sentences:
            self.vocabulary, self.sentences, self.K = self.load_vocabulary()

            with open('./Data/vocabulary.word2VecKeyedVector', 'wb') as file:
                pickle.dump(self.vocabulary, file)

            with open('./Data/sentences.list', 'wb') as file:
                pickle.dump(self.sentences, file)

            with open('./Data/K.int', 'wb') as file:
                pickle.dump(self.K, file)

        # Load preprocessed word embedding vocabularies and sentences
        elif self.load_sentences:
            vocab = 'vocabulary.word2VecKeyedVector'
            sentences = 'sentences.list'
            vocab_size = 'vocabulary_size.int'
            print('Loading tokenized word models ' + str(vocab) + ', ' + str(sentences) + ' and ' + str(
                vocab_size) + '...')

            try:
                with open('./Data/' + str(vocab), 'rb') as file:
                    self.vocabulary = pickle.load(file)

                with open('./Data/' + str(sentences), 'rb') as file:
                    self.sentences = pickle.load(file)

                with open('./Data/' + str(vocab_size), 'rb') as file:
                    self.K = pickle.load(file)
            except FileNotFoundError:
                raise Exception(
                    'Tokenization not found. Download ' + str(vocab) + ', ' + str(sentences) + ' and ' + str(
                        vocab_size) + ' and place in Data/')
        else:
            # Extract word embedding vocabularies and sentences
            self.vocabulary, self.sentences, self.K = self.load_vocabulary()

        # Divide data set into training and validation
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

        weights_word_embedding = self.vocabulary.vectors

        # Number of words in vocabulary
        self.M = size(weights_word_embedding, 0)

        # Needed for featching input during training and inference
        self.input = tf.Variable(0., validate_shape=False)

        self.domain_specification = 'Words'

        # Paramenters to include in learning curve plot
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
        self.inference_iteration = 0

        # Load momentary epoch loss states if training is continued
        if self.load_training_process:
            try:
                self.seq_iterations = [i for i in loadtxt('Parameters/seqIterations.txt', delimiter=",", unpack=False)]
                self.losses = [i for i in loadtxt('Parameters/losses.txt', delimiter=",", unpack=False)]
                self.validation_losses = [i for i in
                                          loadtxt('Parameters/validation_losses.txt', delimiter=",", unpack=False)]
                self.seq_iteration = self.seq_iterations[-1]
            except OSError:
                raise Exception(
                    'Text file arrays seqIterations.txt, losses.txt and validation_losses.txt needs to be in the ' +
                    '/Parameter/ folder. These are created by enabling train_lstm_model and save_checkpoints.')
            except TypeError:
                raise Exception(
                    'Text file arrays in the /Parameter/ folder needs to have at least two elements to plot..')

        self.neurons_of_interest = []
        self.neurons_of_interest_plot = []
        self.neurons_of_interest_plot_intervals = []

        self.scale_color_values = 0.75

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

        sentences = []

        # Ignore rare words
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

            K = size(word2vec_model.wv.vectors, 1)

        elif '.txt' in self.embedding_model_file or '.bin' in self.embedding_model_file:
            is_binary = self.embedding_model_file[-4:] == '.bin'
            print('Loading model "' + self.embedding_model_file + '"...')

            if self.merge_embedding_model_corpus and '.txt' in self.embedding_model_file:
                print('Merging embeding model ' + self.embedding_model_file + ' with corpus ' + self.text_file)
                with open(self.embedding_model_file, 'r', encoding="utf8") as f:
                    lines = f.readlines()

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

        vocabulary = word2vec_model.wv

        # Plot a PCA visualization of word embedding
        if self.n_words_pca_plot:
            pca = PCA(n_components=2)
            X = word2vec_model[vocabulary.index2entity[:self.n_words_pca_plot]]
            result = pca.fit_transform(X)
            plt.title('PCA Projection')
            plt.scatter(result[:, 0], result[:, 1], color='green')
            # ax.view_init(-168, -12)
            words = list(vocabulary.index2entity[:self.n_words_pca_plot])

            for i, word in enumerate(words):
                # x2, y2, _ = proj3d.proj_transform(result[i, 0], result[i, 1], result[i, 2], ax.get_proj())
                x2 = result[i, 0] - 1 - len(word) / 4
                y2 = result[i, 1]
                plt.annotate(word, xy=(x2, y2))

            plt.show()

        del word2vec_model

        return vocabulary, sentences[:int(self.corpus_proportion * len(sentences))], K

    # The main function including the training and inference process
    def run_lstm(self):
        print('\nInitiate LSTM training...')

        if 'RMS' in self.optimizer:
            self.lstm_optimizer = optimizers.RMSprop(lr=self.eta, rho=0.9)
        else:
            self.lstm_optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

        if not self.load_lstm_model:
            print('Initialize new LSTM model...')

            self.lstm_model = self.create_lstm_model(batch_size=self.batch_size)

            self.lstm_model.compile(optimizer=self.lstm_optimizer, loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy'])
        else:
            # Load existing Keras lstm model  and extract its attributes and current epoch if training is to continue
            model_directory = './LSTM Saved Models/'
            models = os.listdir(model_directory)
            model_accuracies = [float(model.split('val_acc')[1][:5]) for model in models if 'val_acc' in model]
            best_model_index = argmax(model_accuracies)
            model = model_directory + os.listdir(model_directory)[best_model_index]
            print('\nLoading best performing LSTM model ' + model + ' with accuracy ' + str(
                model_accuracies[best_model_index]) + '...\n')
            self.lstm_model = load_model(model)
            self.n_hidden_neurons = int(model.split('neurons')[1][:3])
            self.n_hidden_layers = int(model.split('layers')[1][0])
            self.batch_size = int(model.split('batch_size-')[1][:3])
            self.dropout = float(model.split('drop')[1][:4])
            self.eta = float(model.split('eta')[1][:8])
            self.initial_epoch = int(model.split('epoch')[1][:3])

        self.domain_specification = 'Words'

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

        # Allowing fetching the input if visualizing the activations it triggers
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

        # Calculates initial loss when beginning training
        loss = self.lstm_model_evaluate.evaluate(x, y, batch_size=1)[0]

        # Providing initial loss
        class InitLog:
            def get(self, attribute):
                if attribute == 'loss' or attribute == 'val_loss':
                    return loss

        # Training process
        if self.train_lstm_model:
            # Perform epoch zero before training to check untrained performance
            logs = InitLog()
            self.synthesize_text(-1, logs, evaluate=True)

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
            # Inference process for 100 steps. Enter needs to be pressed to predict next text and plots of visualization
            for i in range(100):
                weights = self.lstm_model.get_weights()
                self.lstm_model_evaluate.set_weights(weights)
                self.lstm_model_evaluate.compile(optimizer=self.lstm_optimizer, loss='sparse_categorical_crossentropy',
                                                 metrics=['accuracy'])

                loss = self.lstm_model_evaluate.evaluate(x, y, batch_size=1)[0]

                logs = InitLog()
                self.synthesize_text(-1, logs, evaluate=True)
                input("\nPress Enter to continue...")

    # Initialization of new LSTM model
    def create_lstm_model(self, batch_size):
        input_layer = Input(batch_shape=(batch_size, self.seq_length - 1,))
        embedding_layer = Embedding(input_dim=self.M, output_dim=self.K, weights=[self.vocabulary.vectors])(input_layer)

        lstm_input = embedding_layer
        for i in range(self.n_hidden_layers - 1):
            lstm_input = LSTM(units=self.n_hidden_neurons, return_sequences=True, return_state=True,
                              dropout=self.dropout)(lstm_input)

        final_lstm_layer, lstm_outputs, lstm_gate_outputs = LSTM(units=self.n_hidden_neurons, return_state=True,
                                                                 stateful=True, dropout=self.dropout)(lstm_input)

        fully_connected_layer = Dense(units=self.M)(final_lstm_layer)

        output_layer = Activation('softmax')(fully_connected_layer)

        lstm_model = Model(inputs=input_layer, outputs=output_layer)

        return lstm_model

    # Yield one preprocessed batch of input words and targets at a time to not face memory issues preprocessing complete corpus set
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

    # Main function to predict text and to visualize neural activity
    def synthesize_text(self, epoch, logs={}, evaluate=False):
        print('Generating text sequence...')
        if not self.load_lstm_model or (self.load_lstm_model and not evaluate):
            self.losses.append(logs.get('loss'))
            self.validation_losses.append(logs.get('val_loss'))
            self.seq_iterations.append(self.seq_iteration)
            self.seq_iteration += 1

        self.load_neuron_intervals()
        print('Loaded intervals')

        # Title of table
        table_data = [['Neuron ' + str(self.neurons_of_interest[i]), ''] for i in range(len(self.neurons_of_interest))]

        table = SingleTable(table_data)
        table.table_data.insert(0, ['Neuron ', 'Predicted sentence '])

        # Width of terminal window
        max_width = table.column_max_width(1)

        # Rows with predicted sentences without color coding
        y_n = [[] for _ in range(len(self.neurons_of_interest))]
        # Rows with predicted sentences with color coding
        y = [[] for _ in range(len(self.neurons_of_interest))]

        # Initialization of activation map for the layer defined in FeatureOfInterest.txt
        neuron_activation_map = zeros((self.n_hidden_neurons, self.length_synthesized_text))

        # Indices of input words
        input_indices = []

        for word in self.seed_sentence:
            input_indices.append(self.word_to_index(word))
        input_indices = atleast_2d(input_indices)

        # Indices of predicted words
        entity_indices = zeros(self.seq_length - 1 + self.length_synthesized_text)
        entity_indices[:self.seq_length - 1] = atleast_2d(input_indices[0, :])

        # Printing seeding input sentence
        input_sentence = ''
        for i in range(len(input_indices[0, :])):
            input_sentence += self.index_to_word(int(input_indices[0, i]))

        print('Input sentence: ' + input_sentence)

        # Update weights of evaluating model with batch size 1
        weights = self.lstm_model.get_weights()
        self.lstm_model_evaluate.set_weights(weights)
        self.lstm_model_evaluate.compile(optimizer=self.lstm_optimizer, loss='sparse_categorical_crossentropy',
                                         metrics=['accuracy'])

        # Last input word
        prediction_input = input_sentence[-1]

        # Generate text sequence
        for t in range(self.length_synthesized_text):

            x = entity_indices[t:t + self.seq_length - 1]
            # Whether to flip input, shown improvement in some work but not recommended
            if self.flip_input:
                x_predict = flip(atleast_2d(x), axis=1)
            else:
                x_predict = atleast_2d(deepcopy(x))

            # Predicted probability distribution
            output = self.lstm_model_evaluate.predict(x=x_predict, batch_size=1)

            # Activations to visualize
            ''' Uncomment to set random activations for debugging (and comment line above), will save much time 
            activations = self.get_neuron_activations(x_predict)

            '''
            self.activation_lower_limit = 1
            activations = random.rand(self.n_hidden_neurons, self.length_synthesized_text)

            neuron_activation_map[:, t] = activations[:, 0]
            neuron_activations = activations[self.neurons_of_interest]

            # Sample predicted word
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

            # Color code words according to neural activation
            for i in range(len(self.neurons_of_interest)):
                neuron_activation = self.scale_color_values * neuron_activations[i, 0]

                w = self.white_background

                if neuron_activation > 0:
                    inactive_color = w * 255 - (2 * w - 1) * int(neuron_activation * 255)
                    bg.set_style('activationColor', RgbBg(w * 255, inactive_color, inactive_color))
                else:
                    inactive_color = w * 255 + (2 * w - 1) * int(neuron_activation * 255)
                    bg.set_style('activationColor', RgbBg(inactive_color, inactive_color, w * 255))

                colored_word = bg.activationColor + prediction_input + bg.rs

                y_n[i].append(prediction_input)
                y[i].append(colored_word)

            prediction_input = sample

        # Extract all predicted words and format to fit the table
        for i in range(len(self.neurons_of_interest)):
            wrapped_string = ''
            line_width = 0

            for j in range(len(y[i])):
                table_row = y[i][j]
                if '\n' in table_row:
                    for k in range(len(table_row.split('\n')) - 1):
                        wrapped_string += table_row.split('\n')[k] + '\\n' + bg.rs + '\n'
                        line_width = 0
                else:
                    wrapped_string += ''.join(table_row)

                line_width += len(y_n[i][j])
                if line_width > max_width - 10:
                    line_width = 0
                    wrapped_string += ' ' * (max_width - line_width) * 0 + bg.rs + '\n'

            table.table_data[i + 1][1] = wrapped_string + '\n'

        # Get limit activation for color range scale
        if self.neurons_of_interest:
            max_activation = amax(neuron_activation_map[self.neurons_of_interest, :])
            min_activation = amin(neuron_activation_map[self.neurons_of_interest, :])
        else:
            max_activation, min_activation = 0, 0

        # Color range scale
        margin = 12
        try:
            color_range_width = max_width - len(table.table_data[0][1]) - (
                    len(str(max_activation)) + len(str(min_activation)) + 2 + margin)
            color_range = arange(min_activation, max_activation,
                                 (max_activation - min_activation) / color_range_width)
        except ValueError:
            color_range_width = 2
            color_range = [min_activation, max_activation]

        color_range_str = ' ' * margin + str(round(min_activation, 1)) + ' '

        for i in range(color_range_width):

            color_range_value = self.scale_color_values * color_range[i]

            w = self.white_background

            if color_range_value > 0:
                inactive_color = w * 255 - (2 * w - 1) * int(color_range_value * 255)
                bg.set_style('activationColor', RgbBg(w * 255, inactive_color, inactive_color))
            else:
                inactive_color = w * 255 + (2 * w - 1) * int(color_range_value * 255)
                bg.set_style('activationColor', RgbBg(inactive_color, inactive_color, w * 255))

            colored_indicator = bg.activationColor + ' ' + bg.rs

            color_range_str += colored_indicator

        color_range_str += ' ' + str(round(max_activation, 1))
        table.table_data[0][1] += color_range_str

        inputs = y_n[0]

        # Auto-detect hypotheses based on relevance heat map
        if self.auto_detect_peak == 'Relevance':
            if self.plot_color_map and not self.train_lstm_model:
                neuron_activation_rows = self.plot_neural_activity(inputs, neuron_activation_map)

                data = []
                for interval in self.neurons_of_interest_plot_intervals:
                    data.extend(table.table_data[interval[0] + 1:(interval[-1] + 2)])
                    data.append(['', '\n'])
                table.table_data[1:] = data

            if self.plot_fft and not self.train_lstm_model:
                self.plot_fft_neural_activity(neuron_activation_rows)

        else:
            if self.plot_fft and not self.train_lstm_model:
                self.plot_fft_neural_activity(neuron_activation_map)

            # Confirm relevant hypotheses with DFT heat surface
            if self.auto_detect_peak == 'FFT':
                data = []
                for interval in self.neurons_of_interest_plot_intervals:
                    data.extend(table.table_data[interval[0] + 1:(interval[-1] + 2)])
                    data.append(['\n', '\n'])
                table.table_data[1:] = data

            if self.plot_color_map and not self.train_lstm_model:
                self.plot_neural_activity(inputs, neuron_activation_map)

        # Flip order to have the same order as the DFT heat surfance, having right oriented coordinate system
        table.table_data[1:] = flip(table.table_data[1:], axis=0)

        if self.plot_process:
            self.plot_learning_curve()

        if not self.load_lstm_model or (self.load_lstm_model and not evaluate):
            print('\nEpoch: ' + str(epoch + 1) + ', Loss: ' + str(
                '{0:.2f}'.format(self.losses[-1])) + ', Validation loss: ' + str(
                '{0:.2f}'.format(self.validation_losses[-1])))

        # Print in terminal or save as table
        if self.present_tables_in_terminal:
            print(table.table)
        else:
            table_file = 'Results/' + str(self.lstm_gate_of_interest) + str(self.inference_iteration) + '.table'
            self.inference_iteration += 1
            with open(table_file, 'wb') as file:
                pickle.dump(table.table, file)

        # Save losses to continue training
        if self.train_lstm_model and self.save_checkpoints:
            savetxt('Parameters/seqIterations.txt', self.seq_iterations, delimiter=',')
            savetxt('Parameters/losses.txt', self.losses, delimiter=',')
            savetxt('Parameters/validation_losses.txt', self.validation_losses, delimiter=',')

    # Extract neural activation in the layer defined in FeaturesOfInterest.txt. Code is based on the Keras source code
    def get_neuron_activations(self, x_predict):

        with open('FeaturesOfInterest.txt', 'r') as f:
            lines = f.readlines()
            lstm_gate_of_interest = []
            for line in lines:
                if 'lstm_gate_of_interest:' in line:
                    lstm_gate_of_interest = line.split(':')[1].split("'")[1]
                    break

            self.lstm_gate_of_interest = lstm_gate_of_interest

        self.activation_lower_limit = -int(self.lstm_gate_of_interest in ['cell', 'final_layer_output'])

        if not lstm_gate_of_interest:
            warnings.warn('lstm_gate_of_interest not properly specified in "FeaturesOfInterest.txt", set to None')
            lstm_gate_of_interest = 'None'

        lstm_layer = self.lstm_model_evaluate.layers[self.lstm_layer_of_interest + 1]

        final_lstm_layer = K.function([self.lstm_model_evaluate.layers[0].input],
                                      [self.lstm_model_evaluate.layers[self.n_hidden_layers + 1].output[0]])

        lstm_layer_input = K.function([self.lstm_model_evaluate.layers[0].input],
                                      [self.lstm_model_evaluate.layers[self.n_hidden_layers + 1].input])

        lstm_input = lstm_layer_input([atleast_2d(x_predict)])[0][0, -1, :]

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

        plt.pause(0.5)

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
                from nltk.corpus import stopwords
                if (feature == 'stopwords' and inputs[i] in stopwords.words('english')) or (
                        feature.isupper() and pos_tag in feature.split(', ')) or (
                        bool(re.fullmatch(r''.join(feature), inputs[i]))):
                    input_indices_of_interest.append(i)
                    if inputs[i] == '\n':
                        inputs[i] = '\\n'
                    inputs_of_interest.append('"' + inputs[i] + '"')
        except Exception as ex:
            print(ex)

        print('Relevant input_indices_of_interest ')
        print(input_indices_of_interest)
        print('Relevant stuff ')
        neuron_activation_rows = neuron_activation_map[self.neurons_of_interest_plot, :]

        input_indices_of_interest_conjugate = list(set(range(len(inputs))) - set(input_indices_of_interest))
        # Flip order to have the same order as the DFT heat surfance, having right oriented coordinate system
        neuron_feature_extracted_map = flip(neuron_activation_rows[:, input_indices_of_interest], axis=0)
        neuron_feature_remaining_map = flip(neuron_activation_rows[:, input_indices_of_interest_conjugate], axis=0)

        # Used for absolute valued difference for relevance heat map
        extracted_mean = array([mean(neuron_feature_extracted_map, axis=1)]).T
        remaining_mean = array([mean(neuron_feature_remaining_map, axis=1)]).T

        # Calculating impulses for the relevance heatmap
        before_action_potential = array(input_indices_of_interest) - 1
        after_action_potential = array(input_indices_of_interest) + 1
        before_action_potential[array(input_indices_of_interest) - 1 == -1] = 1
        after_action_potential[array(input_indices_of_interest) + 1 == size(neuron_activation_rows, 1)] = size(
            neuron_activation_rows, 1) - 2
        prominences = 2 * neuron_activation_rows[:, input_indices_of_interest] - neuron_activation_rows[:,
                                                                                 before_action_potential] - neuron_activation_rows[
                                                                                                            :,
                                                                                                            after_action_potential]
        prominence = atleast_2d(mean(abs(prominences), axis=1)).T
        difference = atleast_2d(mean(abs(extracted_mean - remaining_mean), axis=1)).T
        score = prominence + difference
        relevance = score / amax(score)

        # To tomatically detect peaks in relevance heatamp
        if self.auto_detect_peak == 'Relevance':
            self.reduced_window_size = 3

            argmaxima = relevance[:, 0].argsort(axis=0)[-self.n_auto_detect:]

            self.intervals_to_plot = []
            self.interval_limits = []

            neuron_windows = []
            extracted_relevances = zeros((self.n_auto_detect * self.reduced_window_size, 1))
            neurons_of_interest_relevance = []
            self.neurons_of_interest_plot = []
            self.neurons_of_interest_plot_intervals = []

            neurons = []
            values = []

            # Extract scores and neuron labels for each proposed hypothesis
            for i in range(len(argmaxima)):
                argmax_row = sort(argmaxima)[i]
                neuron_window = [0] * 2
                neuron_window[0] = max(argmax_row - int(self.reduced_window_size / 2), 0)
                neuron_window[1] = min(argmax_row + int(self.reduced_window_size / 2 + 1), size(relevance, 0))
                neuron_windows.append(neuron_window)
                start_range = self.reduced_window_size * i
                end_range = self.reduced_window_size * (i + 1)
                extracted_relevances[start_range:end_range, 0] = relevance[neuron_window[0]:neuron_window[1], 0]
                neurons_of_interest_relevance.extend(range(neuron_window[0], neuron_window[1]))

                print('\nAuto-detected relevance peak for feature "' + feature + '":')
                print('Neuron: ' + str(argmax_row))
                print('Value: ' + str(amax(relevance[neuron_window[0]:neuron_window[1], 0])) + '\n')
                neurons.append(argmax_row)
                values.append(amax(relevance[neuron_window[0]:neuron_window[1], 0]))
                self.neurons_of_interest_plot.extend(range(int(neuron_window[0]), int(neuron_window[-1])))
                self.neurons_of_interest_plot_intervals.append(
                    range(int(neuron_window[0]), int(neuron_window[-1])))

                intermediate_range = [neuron_window[0], argmax_row, neuron_window[1] - 1]
                intermediate_range_str = [str(i) for i in intermediate_range]
                intermediate_range_str[-1] += self.interval_label_shift
                self.intervals_to_plot.extend(intermediate_range_str)
                self.interval_limits.extend(intermediate_range)

            # Proposing most relevant hypotheses in order
            neurons = array(neurons)
            values = array(values)
            sorted_order = flip(values.argsort())
            print('Neurons sorted after relevance and their values:')
            print(neurons[sorted_order])
            print(values[sorted_order])

            relevance = extracted_relevances

            neuron_activation_rows = neuron_activation_map[neurons_of_interest_relevance, :]

            self.neurons_of_interest_fft = range(len(neurons_of_interest_relevance))

            # Flip order to have the same order as the DFT heat surfance, having right oriented coordinate system
            neuron_feature_extracted_map = flip(neuron_activation_rows[:, input_indices_of_interest], axis=0)
            relevance = flip(relevance, axis=0)

        self.plot_color_maps(feature, inputs_of_interest, neuron_feature_extracted_map, relevance)

        return neuron_activation_rows

    # Plot heatmap and relevance heatmap
    def plot_color_maps(self, feature, inputs_of_interest, neuron_feature_extracted_map, relevance):
        f, axarr = plt.subplots(1, 2, num=1, gridspec_kw={'width_ratios': [5, 1]}, clear=True)
        axarr[0].set_title('Colormap of hidden neuron activations')

        feature_label = 'Extracted feature: "' + self.pos_features.get(feature, feature) + '"'
        if (feature == '.' or feature == '\w+|[^\w]'):
            feature_label = 'Feature: ' + '$\it{Any}$'
        x = range(len(inputs_of_interest))

        axarr[0].set_xlabel('Predicted sequence (' + feature_label + ')')
        if (len(inputs_of_interest) < 15):
            axarr[0].set_xticks(x)
            axarr[0].set_xticklabels(inputs_of_interest, fontsize=7,
                                     rotation=90 * (len(feature) > 1) * (
                                             len(inputs_of_interest) >= 6))
        else:
            axarr[0].set_xticks([])

        axarr[1].set_xticks([])

        y = range(len(self.neurons_of_interest_plot))
        intervals = [
            self.intervals_to_plot[where(array(self.interval_limits) == i)[0][0]] if i in self.interval_limits else ' '
            for i in self.neurons_of_interest_plot]

        for i in range(len(axarr)):
            axarr[i].set_yticks(y)
            # Flip order to have the same order as the DFT heat surfance, having right oriented coordinate system
            axarr[i].set_yticklabels(flip(intervals), fontsize=7)
            axarr[0].set_ylabel('Neurons of Interest')

        if self.activation_lower_limit == 0:
            cmap = self.adjusted_color_range(mpl.cm.coolwarm, start=0.5, midpoint=0.75, stop=1, name='shrunk')
        else:
            cmap = 'coolwarm'

        # Heatmap
        colmap = axarr[0].imshow(neuron_feature_extracted_map, cmap=cmap, interpolation='nearest', aspect='auto',
                                 vmin=self.activation_lower_limit, vmax=1)
        # Relevance heatmap
        colmap = axarr[1].imshow(relevance, cmap=cmap, interpolation='nearest', aspect='auto',
                                 vmin=self.activation_lower_limit, vmax=1)

        # Plot separating lines for broken axis of neuron intervals
        if self.auto_detect_peak in ['Relevance', 'FFT']:
            for y in range(self.reduced_window_size, (self.n_auto_detect - 1) * self.reduced_window_size + 1,
                           self.reduced_window_size):
                axarr[0].plot([0, size(neuron_feature_extracted_map, 1) - 1], array([y, y]) - .5, color='black',
                              LineStyle='dashed', LineWidth=2)

            for y in range(self.reduced_window_size, (self.n_auto_detect - 1) * self.reduced_window_size + 1,
                           self.reduced_window_size):
                axarr[1].plot([-.35, .35], array([y, y]) - .5, color='black', LineStyle='dashed', LineWidth=2)

        axarr[1].set_title('Relevance')

        f.colorbar(colmap, ax=axarr.ravel().tolist())

        plt.pause(.1)

    # If activation range goes from 0 - 1, this adjustes the color range for plotting of heatmaps
    def adjusted_color_range(self, cmap, start=0, midpoint=0.5, stop=1.0, name='adjusted_color_range'):

        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }

        reg_index = linspace(start, stop, 257)

        shift_index = hstack([
            linspace(0.0, midpoint, 128, endpoint=False),
            linspace(midpoint, 1.0, 129, endpoint=True)
        ])

        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))

        newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)

        return newcmap

    # Plot of DFT neural activity
    def plot_fft_neural_activity(self, neuron_activation_map):

        neuron_activations = neuron_activation_map[self.neurons_of_interest_fft, :]

        fft_neuron_activations_complex = fft.fft(neuron_activations)

        fft_neuron_activations_abs = abs(fft_neuron_activations_complex / self.length_synthesized_text)

        # Ignore negative frequency components as they are identical to the positive
        fft_neuron_activations_single_sided = fft_neuron_activations_abs[:, 0:int(self.length_synthesized_text / 2)]
        fft_neuron_activations_single_sided[:, 2:-2] = 2 * fft_neuron_activations_single_sided[:, 2:-2]

        freq = arange(0, floor(self.length_synthesized_text / 2)) / self.length_synthesized_text

        neurons_of_interest_fft = self.neurons_of_interest_fft

        # Automatically detect significant frequency components
        if self.auto_detect_peak == 'FFT':
            start_neuron_index = self.neurons_of_interest_plot_intervals[0][0]
            self.reduced_window_size = 5  # 10
            domain_irrelevant_freq = (freq < self.band_width[0]) | (freq > self.band_width[1])

            fft_neuron_activations_single_sided_argpos = deepcopy(fft_neuron_activations_single_sided)
            fft_neuron_activations_single_sided_argpos[:, domain_irrelevant_freq] = 0

            # Find best hypothesis
            argmaxima = amax(fft_neuron_activations_single_sided_argpos, axis=1).argsort(axis=0)[-self.n_auto_detect:]

            self.intervals_to_plot = []
            self.interval_limits = []

            neuron_windows = []
            extracted_fft = zeros(
                (self.n_auto_detect * self.reduced_window_size, size(fft_neuron_activations_single_sided, 1)))
            self.neurons_of_interest_plot = []
            self.neurons_of_interest_plot_intervals = []

            neurons = []
            values = []

            # Extract components and neuron labels of best hypotheses.
            for i in range(len(argmaxima)):
                argmax_row = sort(argmaxima)[i]
                neuron_window = [0] * 2
                neuron_window[0] = max(argmax_row - int(self.reduced_window_size / 2), 0)
                neuron_window[1] = min(argmax_row + int(self.reduced_window_size / 2 + 1),
                                       size(fft_neuron_activations_single_sided, 0))
                neuron_windows.append(neuron_window)
                start_range = self.reduced_window_size * i
                end_range = self.reduced_window_size * (i + 1)

                extracted_fft[start_range:end_range, :] = fft_neuron_activations_single_sided[
                                                          neuron_window[0]:neuron_window[1], :]

                print('\nAuto-detected FFT periodicity peak in band width interval ' + str(self.band_width) + ':')
                print('Neuron: ' + str(argmax_row))
                print('Value: ' + str(
                    amax(fft_neuron_activations_single_sided_argpos[neuron_window[0]:neuron_window[1], :])) + '\n')

                neurons.append(argmax_row)
                values.append(amax(fft_neuron_activations_single_sided_argpos[neuron_window[0]:neuron_window[1], :]))

                self.neurons_of_interest_plot.extend(range(int(neuron_window[0]), int(neuron_window[-1])))
                self.neurons_of_interest_plot_intervals.append(
                    range(int(neuron_window[0]), int(neuron_window[-1])))

                intermediate_range = [neuron_window[0], argmax_row, neuron_window[1] - 1]
                intermediate_range_str = [str(i) for i in intermediate_range]
                intermediate_range_str[-1] += self.interval_label_shift
                self.intervals_to_plot.extend(intermediate_range_str)
                self.interval_limits.extend(intermediate_range)

            # Provide best hypotheses with values in order
            neurons = array(neurons)
            values = array(values)
            sorted_order = flip(values.argsort())
            print('Neurons sorted after relevance and their values:')
            print(neurons[sorted_order])
            print(values[sorted_order])

            neurons_of_interest_fft = range(0, size(extracted_fft, 0))
            fft_neuron_activations_single_sided = extracted_fft

        neurons_of_interest_fft, freq = meshgrid(neurons_of_interest_fft, freq)

        '''
        Plot DFT 3D heat surface of frequency components
        '''
        fig = plt.figure(3, figsize=(18 / 4 * self.n_auto_detect, 6.4))

        plt.clf()
        ax = fig.gca(projection='3d')
        ax.view_init(30, -5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        cmap_color = cm.coolwarm
        surf = ax.plot_surface(freq, neurons_of_interest_fft, fft_neuron_activations_single_sided.T, rstride=1,
                               cstride=1, cmap=cmap_color, linewidth=0,
                               antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.zaxis.set_rotate_label(False)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        plt.title('Fourier Amplitude Spectrum of Neuron Activation')
        plt.xlabel('Frequency')
        plt.ylabel('Neurons of Interest')
        ax.set_zlabel(r'$|\mathcal{F}|$')

        plt.tick_params(labelsize=10)

        y = range(len(self.neurons_of_interest_plot))
        intervals = [
            self.intervals_to_plot[where(array(self.interval_limits) == i)[0][0]] if i in self.interval_limits else ' '
            for i in self.neurons_of_interest_plot]

        plt.yticks(y, (intervals))

        # Plot separating planes for broken axis of neuron intervals
        if self.auto_detect_peak in ['Relevance', 'FFT']:
            for y in arange(self.reduced_window_size - .5, (self.n_auto_detect - 1) * self.reduced_window_size + 1,
                            self.reduced_window_size):
                xs = linspace(0, self.band_width[1], 100)
                zs = linspace(0, amax(fft_neuron_activations_single_sided), 100)

                X, Z = meshgrid(xs, zs)
                Y = y * ones(shape(X))

                ax.plot_surface(X, Y, Z, alpha=.2, color='black')

                directions = ['x', 'z']
                limits = [self.band_width[1], amax(fft_neuron_activations_single_sided)]
                offsets = dict(zip(directions, limits))
                for zdir in directions:
                    for limit in range(len(limits)):
                        ax.contour(X, Y, Z, zdir=zdir, offset=limit * offsets[zdir], colors='black',
                                   linestyles='dashed', linewidths=2)

        fft_neuron_activations_single_sided_argpos = deepcopy(fft_neuron_activations_single_sided)
        fft_neuron_activations_single_sided_argpos[:, :10] = 0
        print('Values: ' + str(amax(fft_neuron_activations_single_sided_argpos, axis=1)) + '\n')
        plt.pause(.5)

    # Load defined neurons of interest and automatical detection boolean from PlotConfigurations.txt
    def load_neuron_intervals(self):
        self.neurons_of_interest = []
        self.neurons_of_interest_plot = []
        self.neurons_of_interest_plot_intervals = []
        self.neurons_of_interest_fft = []
        self.band_width = []

        with open('PlotConfigurations.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('#')[0]
                if 'plot_color_map:' in line:
                    self.plot_color_map = ''.join(line.split()).split(':')[1] == 'True'
                if 'plot_process:' in line:
                    self.plot_process = ''.join(line.split()).split(':')[1] == 'True'
                if 'plot_fft:' in line:
                    self.plot_fft = ''.join(line.split()).split(':')[1] == 'True'
                if 'auto_detect_peak:' in line:
                    self.auto_detect_peak = line.split("'")[1]
                if 'n_auto_detect:' in line:
                    self.n_auto_detect = int(line.split()[1])

        if self.auto_detect_peak in ['Relevance', 'FFT']:
            self.intervals_to_plot = []
            self.interval_limits = []
            self.interval_label_shift = ''

            interval = ['0', str(self.n_hidden_neurons - 1)]
            self.neurons_of_interest.extend(range(int(interval[0]), int(interval[-1]) + 1))

            self.neurons_of_interest_plot.extend(range(int(interval[0]), int(interval[-1]) + 1))
            self.neurons_of_interest_plot_intervals.append(
                range(int(interval[0]), int(interval[-1]) + 1))
            frequency = 1 + 4 * int(len(self.neurons_of_interest) > 5)

            intermediate_range = [i for i in range(int(interval[0]) + 1, int(interval[-1])) if
                                  i % frequency == 0]
            intermediate_range.insert(0, int(interval[0]))
            intermediate_range.append(int(interval[-1]))
            intermediate_range_str = [str(i) for i in intermediate_range]
            intermediate_range_str[-1] += self.interval_label_shift
            self.intervals_to_plot.extend(intermediate_range_str)
            self.interval_limits.extend(intermediate_range)

            self.neurons_of_interest_fft.extend(range(int(interval[0]), int(interval[-1]) + 1))
            '''

            '''
            if self.auto_detect_peak in ['FFT', 'Relevance']:
                with open('FeaturesOfInterest.txt', 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'Band width to auto-detect prominent frequency components (Hz):' in line:
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
        else:
            self.set_relevant_neurons()

    # If not automatic detection is utilzed, simly set neurons of interest in accordance with PlotConfigurations.txt
    def set_relevant_neurons(self):
        with open('FeaturesOfInterest.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '#' not in line:
                    if 'Neurons to fourier transform:' in line:
                        line = line.replace('Neurons to fourier transform:', '')
                        intervals = ''.join(line.split()).split(',')
                        for interval in intervals:
                            if ':' in interval:
                                interval = interval.split(':')
                                interval[0] = str(max(int(interval[0]), 0))
                                interval[-1] = str(min(int(interval[-1]), self.n_hidden_neurons - 1))
                                self.neurons_of_interest_fft.extend(range(int(interval[0]), int(interval[-1]) + 1))
                            else:
                                interval = str(max(int(interval), 0))
                                interval = str(min(int(interval), self.n_hidden_neurons - 1))
                                self.neurons_of_interest_fft.append(int(interval))

                    if 'Neurons to print:' in line:
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
                    if 'Neurons for heatmap:' in line:
                        line = line.replace('Neurons for heatmap:', '')
                        intervals = ''.join(line.split()).split(',')
                        self.intervals_to_plot = []
                        self.interval_limits = []
                        self.interval_label_shift = ''

                        for interval in intervals:
                            if ':' in interval:
                                interval = interval.split(':')
                                interval[0] = str(max(int(interval[0]), 0))
                                interval[-1] = str(min(int(interval[-1]), self.n_hidden_neurons - 1))
                                self.neurons_of_interest_plot.extend(range(int(interval[0]), int(interval[-1]) + 1))
                                self.neurons_of_interest_plot_intervals.append(
                                    range(int(interval[0]), int(interval[-1]) + 1))
                                frequency = 1 + 4 * int(len(self.neurons_of_interest) > 5)

                                intermediate_range = [i for i in range(int(interval[0]) + 1, int(interval[-1])) if
                                                      i % frequency == 0]
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

    def word_to_index(self, word):
        return self.vocabulary.vocab[word].index

    def index_to_word(self, index):
        return self.vocabulary.index2word[index]


# To find good hyper parameters, perform randomized search
def randomize_hyper_parameters(n_configurations, attributes):
    attributes['n_epochs'] = 6
    attributes['patience'] = 1

    for i in range(n_configurations):
        attributes['n_hidden_neurons'] = int(6 * random.rand()) + 1  # 32 * int(7 * random.rand() + 16)
        attributes['batch_size'] = 32 * int(6 * random.rand() + 7)
        attributes['dropout'] = float(floor(31 * random.rand() + 20) * .01)
        attributes['eta'] = int(8 * random.rand() + 2) * 10 ** int(-4 - 3 * random.rand())
        # attributes['flip_input'] = random.rand() < 0.2

        print('\nn: ' + str(attributes['n_hidden_neurons']))
        print('layers: ' + str(attributes['n_hidden_layers']))
        print('batch_size: ' + str(attributes['batch_size']))
        print('dropout: ' + str(attributes['dropout']))
        print('eta: ' + str(attributes['eta']))

        lstm_vis = VisualizeLSTM(attributes)
        lstm_vis.run_lstm()

        K.clear_session()


# Suggested seeded input sentences with suitable text length. Note that even if the input sentences and seeds are the
# same as used in the examples of the paper, the outcome of generated sentences are not necessarily the same. It will
# locally be deterministic, though.
def template_configurations(attributes, configuaration):
    if configuaration:
        attributes['seed_sentence'] = ['Years', ' ', 'of', ' ', 'intelligent', ' ', 'humans', ':']
        attributes['length_synthesized_text'] = 406
        seed = 173  # Yields different texts in different environments
    else:
        attributes['seed_sentence'] = ['Here', ' ', 'are', ' ', 'two', ' ', 'reasons', ' ']
        attributes['length_synthesized_text'] = 222
        seed = 0  # Yields different texts in different environments

    return attributes, seed


def main():
    attributes = {
        'present_tables_in_terminal': True,
        # True to only print visualization tables in terminal, else to save in files
        'white_background': False,  # True for white background, else black (Terminal properties needs to be adjusted)
        'text_file': '../Corpus/ted_en.zip',  # Corpus file for training
        'load_lstm_model': True,  # True to load lstm checkpoint model with best validation accuracy
        'load_training_process': False,
        # True to load training and validation accuracy. Requires training with save_checkpoints = True.
        'train_lstm_model': False,  # True to train model, otherwise inference process is applied for text generation
        'lstm_layer_of_interest': 1,  # LSTM layer to visualize
        'optimizer': 'RMS Prop',  # Either 'RMS Prop', or Adadelta is used as default
        'dropout': 0.44,  # LSTM dropout rate (in connections from input to output i.e. not recurrent connections)
        'embedding_model_file': 'Word_Embedding_Model/ted_en_glove_840B_300d_extracted.model',
        # Optional path to embedding model, leave value string empty for embedding new corpus.
        'word_frequency_threshold': 10,  # Ignore words less common than this frequency
        'shuffle_data_sets': False,  # Whether to shuffle data sets, note that LSTM stateful should then be False
        'merge_embedding_model_corpus': False,  # Extract the intersection between embedding model and corpus
        'train_embedding_model': False,  # Further train the embedding model
        'save_embedding_model': False,  # Save trained embedding model
        'save_sentences': False,  # Save sentences and vocabulary
        'load_sentences': True,  # Load sentences and vocabulary
        'n_words_pca_plot': 0,  # > 0 if this number of most common words should be plotted through PCA
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
        # If input context window should be flipped, empirically this should be false but may improve performance
        'length_synthesized_text': 200,  # Sequence length of each print of text evolution
        'remote_monitoring_ip': '',  # Ip for remote monitoring at http://localhost:9000/
        'save_checkpoints': False  # Save best weights with corresponding arrays iterations and smooth loss
    }

    configuration = 1  # Tested seeds and text lengths for good text generation instances. These may also be arbitrary.
    attributes, seed = template_configurations(attributes, configuration)

    random.seed(seed)

    # Initialize class attributes
    lstm_vis = VisualizeLSTM(attributes)

    # Run main function
    lstm_vis.run_lstm()

    # Release memory when done
    K.clear_session()


if __name__ == '__main__':
    main()
    plt.show()
