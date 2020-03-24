'''
@author: John Henry Dahlberg

2019-02-05
'''

from __future__ import print_function
import sklearn.preprocessing
from numpy import *
from copy import *
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import AnchoredText
import platform
from sty import bg, RgbBg
from gensim.models import KeyedVectors
import ctypes
import re
import zipfile
import lxml.etree
from terminaltables import SingleTable
import time
import datetime
import os
import pickle
from decimal import Decimal


class VisualizeRNN(object):

    def __init__(self, attributes=None):
        if not attributes:
            raise Exception('Dictionary argument "attributes" is required.')

        self.__dict__ = attributes

        # Allowing ANSI Escape Sequences for colors
        if platform.system().lower() == 'windows':
            stdout_handle = ctypes.windll.kernel32.GetStdHandle(ctypes.c_int(-11))
            mode = ctypes.c_int(0)
            ctypes.windll.kernel32.GetConsoleMode(ctypes.c_int(stdout_handle), ctypes.byref(mode))
            mode = ctypes.c_int(mode.value | 4)
            ctypes.windll.kernel32.SetConsoleMode(ctypes.c_int(stdout_handle), mode)

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
                    self.word2vec_model = pickle.load(file)

                with open('./Data/sentences.list', 'rb') as file:
                    self.sentences = pickle.load(file)

                with open('./Data/K.int', 'rb') as file:
                    self.K = pickle.load(file)
            else:
                self.word2vec_model, input_sequence, self.K = self.load_vocabulary()
        else:
            input_sequence, self.char_to_ind, self.ind_to_char = self.load_characters()
            self.K = len(self.ind_to_char)

        if self.n_hidden_neurons == 'Auto':
            self.n_hidden_neurons = self.K

        n_validation = int(len(input_sequence) * self.validation_proportion)
        n_training = len(input_sequence) - n_validation

        input_sequence = input_sequence[:int(self.corpus_proportion*len(input_sequence))]

        self.input_sequence = input_sequence[:n_training]
        self.input_sequence_validation = input_sequence[n_training:]

        self.weights = ['W', 'V', 'U', 'b', 'c']
        self.gradients = ['dLdW', 'dLdV', 'dLdU', 'dLdB', 'dLdC']
        self.num_gradients = ['gradWnum', 'gradVnum', 'gradUnum', 'gradBnum', 'gradCnum']

        self.sizes = [(self.n_hidden_neurons, self.n_hidden_neurons), (self.K, self.n_hidden_neurons), \
                      (self.n_hidden_neurons, self.K), (self.n_hidden_neurons, 1), (self.K, 1)]

        self.init_epoch = 0
        self.init_iteration = 0

        # Weight initialization
        if self.weight_init == 'Load':
            print('Loading weights...')
        else:
            print('Initializing weights...')

        for weight, grad_index in zip(self.weights, range(len(self.gradients))):
            if self.sizes[grad_index][1] > 1:
                if self.weight_init == 'Load':
                    self.init_sigma = loadtxt(self.model_directory + 'initSigma.txt', unpack=False)
                    setattr(self, weight, array(loadtxt(self.model_directory + 'Weights/' + weight + ".txt", comments="#", delimiter=",", unpack=False)))
                else:
                    if self.weight_init == 'He':
                        self.init_sigma = sqrt(2 / sum(self.sizes[grad_index]))
                    else:
                        self.init_sigma = 0.01
                    setattr(self, weight, self.init_sigma*random.randn(self.sizes[grad_index][0], self.sizes[grad_index][1]))
            else:
                if self.weight_init == 'Load':
                    self.init_sigma = loadtxt(self.model_directory + 'initSigma.txt', unpack=False)
                    setattr(self, weight, array([loadtxt(self.model_directory + 'Weights/' + weight + ".txt", comments="#", delimiter=",", unpack=False)]).T)
                else:
                    setattr(self, weight, zeros(self.sizes[grad_index]))

        if self.weight_init == 'Load':
            self.seq_iterations = loadtxt(self.model_directory + 'seqIterations.txt', delimiter=",", unpack=False)
            self.smooth_losses = loadtxt(self.model_directory + 'smoothLosses.txt', delimiter=",", unpack=False)
            self.validation_losses = loadtxt(self.model_directory + 'validationLosses.txt', delimiter=",", unpack=False)
            self.init_epoch = int(self.model_directory.split('epoch')[1][0])
            self.init_iteration = 0 # int(self.model_directory.split('iteration')[1].split('-')[0])
            self.n_hidden_neurons = int(self.model_directory.split('neurons')[1][:3])
            self.eta = float(self.model_directory.split('eta-')[1][:7])

        self.x0 = ' '
        self.h0 = zeros((self.n_hidden_neurons, 1))

        self.loss_momentum = 1e-3

    def load_characters(self):

        print('Loading text file "' + self.text_file + '"...')
        if self.text_file[-4:] == '.zip':
            with zipfile.ZipFile(self.text_file, 'r') as z:
                doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
            print('Extracting characters...')
            input_text = '\n'.join(doc.xpath('//content/text()'))

        characters = []
        [characters.append(char) for sentences in input_text for char in sentences if char not in characters]
        print('Unique characters:\n' + str(characters))
        k = len(characters)
        indicators = array(range(k))

        ind_one_hot = self.to_one_hot(indicators)

        char_to_ind = dict((characters[i], array(ind_one_hot[i])) for i in range(k))
        ind_to_char = dict((indicators[i], characters[i]) for i in range(k))

        return input_text, char_to_ind, ind_to_char

    def load_vocabulary(self):
        self.model_file = 'Data/glove_840B_300d.txt'  # Word tokenization text
        is_binary = self.model_file[-4:] == '.bin'
        print('Loading model "' + self.model_file + '"...')
        word2vec_model = KeyedVectors.load_word2vec_format(self.model_file, binary=is_binary)
        K = size(word2vec_model.vectors, 1)

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

        return word2vec_model, words, K

    def get_words(self, e):
        x_sequence = self.input_sequence[e:e + self.seq_length]
        y_sequence = self.input_sequence[e + 1:e + self.seq_length + 1]

        x = []
        y = []

        for i in range(len(x_sequence)):
            x_word = x_sequence[i]
            y_word = y_sequence[i]
            try:
                x.append(array([self.word2vec_model[x_word]]).T)
            except KeyError:
                self.word2vec_model[x_word] = random.uniform(-0.25, 0.25, self.K)
                x.append(array([self.word2vec_model[x_word]]).T)
                print("Word '" + x_word + "'" + ' added to model.')
            try:
                y.append(array([self.word2vec_model[y_word]]).T)
            except KeyError:
                self.word2vec_model[y_word] = random.uniform(-0.25, 0.25, self.K)
                y.append(array([self.word2vec_model[y_word]]).T)
        return x_sequence, y_sequence, x, y

    def get_characters(self, e, input_sequence, seq_length=None):
        if not seq_length:
            seq_length = self.seq_length

        x_sequence = input_sequence[e:e+seq_length]
        y_sequence = input_sequence[e+1:e+seq_length + 1]
        x = self.seq_to_one_hot(x_sequence)
        y = self.seq_to_one_hot(y_sequence)

        return x_sequence, y_sequence, x, y

    def run_vanilla_rnn(self):
        if self.weight_init == 'Load':
            smooth_loss = self.smooth_losses[-1]
            validation_loss = self.validation_losses[-1]
            lowest_validation_loss = validation_loss
        else:
            smooth_loss = None

        if self.word_domain:
            self.domain_specification = 'Words'
        else:
            self.domain_specification = 'Characters'

        constants = 'Max Epochs: ' + str(self.n_epochs) + ' (' + str(len(self.input_sequence)/self.seq_length * self.n_epochs) + ' seq. iter.)' \
                    + '\n# Hidden neurons: ' + str(self.n_hidden_neurons) \
                    + '\nWeight initialization: ' + str(self.weight_init) \
                    + '\n' + r'$\sigma$ = ' + "{:.2e}".format(self.init_sigma) \
                    + '\n' + r'$\eta$ = ' + "{:.2e}".format(self.eta) \
                    + '\n' + 'Sequence length: ' + str(self.seq_length) \
                    + '\n#' + self.domain_specification + ' in training text:' + '\n' + str(len(self.input_sequence)) \
                    + '\n' + 'AdaGrad: ' + str(self.ada_grad_sgd) \
                    + '\n' + 'RMS Prop: ' + str(self.rms_prop)

        if self.rms_prop:
            constants += '\n' + r'$\gamma$ = ' + "{:.2e}".format(self.gamma)

        m = []
        for weight in self.weights:
            m.append(zeros(getattr(self, weight).shape))

        if self.weight_init == 'Load':
            seq_iteration = self.seq_iterations[-1]
            seq_iterations = [s for s in self.seq_iterations]
            smooth_losses = [s for s in self.smooth_losses]
            validation_losses = [s for s in self.validation_losses]
        else:
            seq_iteration = 0
            seq_iterations = []
            smooth_losses = []
            validation_losses = []
        smooth_losses_temp = []
        validation_losses_temp = []
        seq_iterations_temp = []

        start_time = time.time()
        previous_time = start_time
        for epoch in range(self.init_epoch, self.n_epochs):

            h_prev = deepcopy(self.h0)

            for e in range(self.init_iteration, len(self.input_sequence)-self.seq_length-1, self.seq_length):

                if self.word_domain:
                    x_sequence, y_sequence, x, y = self.get_words(e)
                else:
                    x_sequence, y_sequence, x, y = self.get_characters(e, self.input_sequence)

                output, h, a = self.forward_prop(x, h_prev)

                if (self.train_model):
                    self.back_prop(x, y, output, h)

                loss, accuracy = self.compute_loss(output, y)
                if not smooth_loss:
                    smooth_loss = loss

                smooth_loss = (1 - self.loss_momentum) * smooth_loss + self.loss_momentum * loss

                if (not self.train_model) or time.time() - previous_time > 900 or (time.time() - start_time < 5 and time.time() - previous_time > 3) or e >= len(self.input_sequence)-2*self.seq_length-1:
                    print("Evaluating and presenting current model..")
                    seq_iterations_temp.append(seq_iteration)
                    smooth_losses_temp.append(smooth_loss)

                    x0 = self.input_sequence[e]

                    if self.word_domain:
                        x_sequence, y_sequence, x, y = self.get_words(e)
                    else:
                        x_sequence, y_sequence, x, y = self.get_characters(0, self.input_sequence_validation, self.length_synthesized_text)

                    output, h, a = self.forward_prop(x, h_prev)

                    validation_loss, accuracy = self.compute_loss(output, y)
                    lowest_validation_loss = copy(validation_loss)
                    validation_losses_temp.append(validation_loss)

                    table, neuron_activation_map, inputs = self.synthesize_text(x0, h_prev, self.length_synthesized_text)

                    with open('PlotConfigurations.txt', 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.split('#')[0]
                            
                            if 'plot_process:' in line:
                                self.plot_process = ''.join(line.split()).split(':')[1] == 'True'
                            elif 'plot_color_map:' in line:
                                self.plot_color_map = ''.join(line.split()).split(':')[1] == 'True'
                            elif 'plot_fft:' in line:
                                self.plot_fft = ''.join(line.split()).split(':')[1] == 'True'
                            elif 'auto_detect_peak:' in line:
                                self.auto_detect_peak = line.split("'")[1]

                    if self.plot_color_map:
                        self.plot_neural_activity(inputs, neuron_activation_map)

                    if self.plot_fft:
                        self.plot_fft_neural_activity(neuron_activation_map)

                    time_passed = time.time() - start_time
                    estimated_total_time = time_passed/(max(e, 1)/len(self.input_sequence))
                    remaining_time = estimated_total_time - time_passed
                    previous_time = time.time()
                    print('\nSequence iteration: ' + str(seq_iteration) + ', Epoch: ' + str(epoch)
                          + ', Epoch ETA: ' + str(datetime.timedelta(seconds=int(remaining_time)))
                          + ', Epoch process: ' + str('{0:.2f}'.format(e/len(self.input_sequence)*100)) + '%'
                          + ', Training loss: ' + str('{0:.2f}'.format(smooth_loss)) + ', Neuron of interest: '
                          + ', Validation loss: ' + str('{0:.2f}'.format(validation_loss)) + ', Neuron of interest: ' +
                          str(self.neurons_of_interest) + '(/' + str(self.n_hidden_neurons) + ')')

                    print(table)

                    if self.plot_process:
                        fig = plt.figure(3)
                        plt.clf()
                        ax = fig.add_subplot(111)
                        fig.subplots_adjust(top=0.85)
                        anchored_text = AnchoredText(constants, loc=1)
                        ax.add_artist(anchored_text)

                        plt.title(self.domain_specification[:-1] + ' prediction learning curve of Recurrent Neural Network')
                        plt.ylabel('Smooth loss')
                        plt.xlabel('Sequence iteration')
                        plt.plot(seq_iterations+seq_iterations_temp, smooth_losses+smooth_losses_temp, LineWidth=2, label='Training')
                        plt.plot(seq_iterations+seq_iterations_temp, validation_losses+validation_losses_temp, LineWidth=2,  label='Validation')
                        plt.grid()

                        plt.legend(loc='upper left')
                        plt.pause(.5)

                    if not self.train_model:
                        input("\nPress Enter to continue...")
                    else:
                        if validation_loss <= lowest_validation_loss:
                            seq_iterations += seq_iterations_temp
                            smooth_losses += smooth_losses_temp
                            validation_losses += validation_losses_temp
                            smooth_losses_temp = []
                            validation_losses_temp = []
                            seq_iterations_temp = []
                            lowest_validation_loss = copy(smooth_loss)
                            h_prev_best = copy(h_prev)

                            if self.train_model and self.save_parameters:
                                state = "val_loss%.3f-val_acc%.3f-loss%.3f-epoch%d-iteration%d-neurons%d-eta-"%(validation_loss, accuracy, loss, epoch, int(e/self.seq_length), self.n_hidden_neurons) + '{:.2e}'.format(Decimal(self.eta)) + "/"
                                try:
                                    for weight in self.weights:
                                        file_name = 'Vanilla RNN Saved Models/' + state + 'Weights/' + weight + '.txt'
                                        os.makedirs(os.path.dirname(file_name), exist_ok=True)
                                        savetxt(file_name, getattr(self, weight), delimiter=',')

                                    os.makedirs(os.path.dirname('Vanilla RNN Saved Models/' + state + 'init_sigma.txt'), exist_ok=True)
                                    savetxt('Vanilla RNN Saved Models/' + state + 'init_sigma.txt', array([[self.init_sigma]]))
                                    os.makedirs(os.path.dirname('Vanilla RNN Saved Models/' + state + 'seq_iterations.txt'), exist_ok=True)
                                    savetxt('Vanilla RNN Saved Models/' + state + 'seq_iterations.txt', seq_iterations, delimiter=',')
                                    os.makedirs(os.path.dirname('Vanilla RNN Saved Models/' + state + 'smooth_losses.txt'), exist_ok=True)
                                    savetxt('Vanilla RNN Saved Models/' + state + 'smooth_losses.txt', smooth_losses, delimiter=',')
                                    os.makedirs(os.path.dirname('Vanilla RNN Saved Models/' + state + 'validation_losses.txt'), exist_ok=True)
                                    savetxt('Vanilla RNN Saved Models/' + state + 'validation_losses.txt', validation_losses, delimiter=',')

                                except Exception as ex:
                                    print(ex)

                        print('Continuing training...')

                if self.train_model:
                    epsilon = 1e-10

                    if self.rms_prop:
                        c_m = self.gamma
                        c_g = 1 - self.gamma
                    else:
                        c_m, c_g, = 1, 1

                    for grad, weight, grad_index in zip(self.gradients, self.weights, range(len(self.gradients))):
                        if self.ada_grad_sgd:
                            m[grad_index] = c_m * m[grad_index] + c_g*getattr(self, grad)**2
                            sqrt_inv_m = (m[grad_index]+epsilon)**-0.5
                            updated_weight = getattr(self, weight) - self.eta * multiply(sqrt_inv_m, getattr(self, grad))
                        else:
                            updated_weight = deepcopy(getattr(self, weight)) - self.eta * deepcopy(getattr(self, grad))

                        setattr(self, weight, updated_weight)

                    h_prev = deepcopy(h[-1])

                    seq_iteration += 1

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

        if len(inputs_of_interest) > 20:
            inputs_of_interest = [' ']*len(inputs_of_interest)
        elif len(input_indices_of_interest) < 1:
            warnings.warn('The feature of interest is not found in generated sequence')
            return False;

        f, axarr = plt.subplots(1, 2, num=1, gridspec_kw={'width_ratios': [5, 1]}, clear=True)
        axarr[0].set_title('Colormap of hidden neuron activations')

        feature_label = 'Feature: "' + feature + '"'
        if not self.word_domain and feature == '.':
            feature_label = 'Feature: ' + '$\it{Any}$'
        x = range(len(inputs_of_interest))
        axarr[0].set_xticks(x)
        axarr[0].set_xlabel('Predicted sequence (' + feature_label + ')')
        axarr[0].set_xticklabels(inputs_of_interest, fontsize=7, rotation=90 * self.word_domain)
        axarr[1].set_xticks([])

        if self.auto_detect_peak == 'Relevance':
            neuron_activation_rows = neuron_activation_map
        else:
            neuron_activation_rows = neuron_activation_map[self.neurons_of_interest_plot, :]

        max_activation = amax(neuron_activation_map)
        min_activation = amin(neuron_activation_map)

        input_indices_of_interest_conjugate = list(set(range(len(inputs))) - set(input_indices_of_interest))
        neuron_feature_extracted_map = flip(neuron_activation_rows[:, input_indices_of_interest], axis=0)
        neuron_feature_remaining_map = flip(neuron_activation_rows[:, input_indices_of_interest_conjugate], axis=0)

        before_action_potential = array(input_indices_of_interest) - 1
        after_action_potential = array(input_indices_of_interest) + 1
        before_action_potential[array(input_indices_of_interest) - 1 == -1] = 1
        after_action_potential[array(input_indices_of_interest) + 1 == size(neuron_activation_rows, 1)] = size(
            neuron_activation_rows, 1) - 2

        prominences = 2 * neuron_activation_rows[:, input_indices_of_interest] - neuron_activation_rows[:, before_action_potential] - neuron_activation_rows[:,after_action_potential]
        prominence = atleast_2d(mean(abs(prominences), axis=1)).T

        extracted_mean = array([mean(neuron_feature_extracted_map, axis=1)]).T
        remaining_mean = array([mean(neuron_feature_remaining_map, axis=1)]).T

        difference = atleast_2d(mean(abs(extracted_mean - remaining_mean), axis=1)).T

        score = prominence + difference
        relevance = score / amax(score)

        if self.auto_detect_peak == 'Relevance':
            reduced_window_size = 10

            argmax_row = where(relevance == amax(relevance))[0][0]
            neuron_window = [0]*2
            neuron_window[0] = max(argmax_row - int(reduced_window_size / 2), 0)
            neuron_window[1] = min(argmax_row + int(reduced_window_size / 2 + 1), size(relevance, 0))
            relevance = relevance[neuron_window[0]:neuron_window[1], :]
            neurons_of_interest_relevance = range(neuron_window[0], neuron_window[1])
            print('\nAuto-detected relevance peak for feature "' + feature + '":')
            print('Neuron: ' + str(argmax_row))
            print('Value: ' + str(amax(relevance)) + '\n')

            neuron_activation_rows = neuron_activation_map[neurons_of_interest_relevance, :]

            neuron_feature_extracted_map = flip(neuron_activation_rows[:, input_indices_of_interest], axis=0)

            self.intervals_to_plot = []
            self.interval_limits = []

            interval = [str(neuron_window[0]),  str(neuron_window[1])]

            interval[0] = str(max(int(interval[0]), 0))
            interval[-1] = str(min(int(interval[-1]), self.K - 1))
            self.neurons_of_interest_plot.extend(range(int(interval[0]), int(interval[-1]) + 1))
            self.neurons_of_interest_plot_intervals.append(range(int(interval[0]), int(interval[-1]) + 1))
            intermediate_range = [i for i in range(int(interval[0]) + 1, int(interval[-1]))]
            intermediate_range.insert(0, int(interval[0]))
            intermediate_range.append(int(interval[-1]))
            intermediate_range_str = [str(i) for i in intermediate_range]
            intermediate_range_str[-1] += self.interval_label_shift
            self.intervals_to_plot.extend(intermediate_range_str)
            self.interval_limits.extend(intermediate_range)

            self.interval_limits = array(self.interval_limits)

            self.neurons_of_interest_plot = range(neuron_window[0], neuron_window[1])

        y = range(len(self.neurons_of_interest_plot))
        intervals = [
            self.intervals_to_plot[where(self.interval_limits == i)[0][0]] if i in self.interval_limits else ' ' for i
            in self.neurons_of_interest_plot]

        for i in range(len(axarr)):
            axarr[i].set_yticks(y)
            axarr[i].set_yticklabels(flip(intervals), fontsize=7)
            axarr[0].set_ylabel('Neuron')


        colmap = axarr[0].imshow(neuron_feature_extracted_map, cmap='coolwarm', interpolation='nearest', aspect='auto',
                                 vmin=min_activation, vmax=max_activation)
        colmap = axarr[1].imshow(relevance,
            cmap='coolwarm', interpolation='nearest', aspect='auto', vmin=min_activation, vmax=max_activation)
        axarr[1].set_title('Relevance')

        if self.auto_detect_peak != 'Relevance':
            interval = 0
            for i in range(len(self.neurons_of_interest_plot_intervals) + 1):
                if i > 0:
                    limit = self.neurons_of_interest_plot_intervals[i - 1]
                    interval += 1 + limit[-1] - limit[0]
                axarr[0].plot(arange(-.5, len(input_indices_of_interest) + .5),
                              (len(input_indices_of_interest) + 1) * [interval - 0.5], 'k--', LineWidth=1)

        f.colorbar(colmap, ax=axarr.ravel().tolist())

        plt.pause(.1)

        return True

    def plot_fft_neural_activity(self, neuron_activation_map):

        neurons_of_interest_fft = range(16, 21)
        if self.auto_detect_peak != 'FFT':
            neuron_activations = neuron_activation_map[neurons_of_interest_fft, :]
        else:
            neuron_activations = neuron_activation_map

        fft_neuron_activations_complex = fft.fft(neuron_activations)

        fft_neuron_activations_abs = abs(fft_neuron_activations_complex / self.length_synthesized_text)

        fft_neuron_activations_single_sided = fft_neuron_activations_abs[:, 0:int(self.length_synthesized_text / 2)]
        fft_neuron_activations_single_sided[:, 2:-2] = 2 * fft_neuron_activations_single_sided[:, 2:-2]

        freq = arange(0, floor(self.length_synthesized_text / 2)) / self.length_synthesized_text

        if self.auto_detect_peak == 'FFT':
            self.band_width = [0.1, 0.4]
            start_neuron_index = 0
            neuron_window = [start_neuron_index] * 2
            reduced_window_size = 10
            domain_relevant_freq = (freq > self.band_width[0]) & (freq < self.band_width[1])
            # freq = freq[domain_relevant_freq]
            domain_relevant_components = fft_neuron_activations_single_sided[:, domain_relevant_freq]
            argmax_row = where(fft_neuron_activations_single_sided == amax(domain_relevant_components))[0][0]

            neuron_window[0] += max(argmax_row - int(reduced_window_size / 2), 0)
            neuron_window[1] += min(argmax_row + int(reduced_window_size / 2 + 1), size(domain_relevant_components, 0))
            fft_neuron_activations_single_sided = fft_neuron_activations_single_sided[
                                                  neuron_window[0] - start_neuron_index:neuron_window[
                                                                                            1] - start_neuron_index, :]
            neurons_of_interest_fft = range(neuron_window[0], neuron_window[1])
            print('\nAuto-detected FFT periodicity peak in band width interval ' + str(self.band_width) + ':')
            print('Neuron: ' + str(argmax_row))
            print('Value: ' + str(amax(domain_relevant_components)) + '\n')

        neurons_of_interest_fft, freq = meshgrid(neurons_of_interest_fft, freq)

        fig = plt.figure(2)
        plt.clf()
        ax = fig.gca(projection='3d')
        ax.view_init(20, -120)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        cmap_color = cm.coolwarm  # cm.coolwarm
        surf = ax.plot_surface(freq, neurons_of_interest_fft, fft_neuron_activations_single_sided.T, rstride=1,
                               cstride=1, cmap=cmap_color, linewidth=0,
                               antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        ax.zaxis.set_rotate_label(False)

        plt.title('Fourier Amplitude Spectrum of Neuron Activation')
        plt.xlabel('Frequency')
        plt.ylabel('Neurons of interest')
        ax.set_zlabel(r'$|\mathcal{F}|$')

        plt.pause(.5)


    def forward_prop(self, x, h_prev, weights={}):
        if not weights:
            weights_tuples = [(self.weights[i], getattr(self, self.weights[i])) for i in range(len(self.weights))]
            weights = dict(weights_tuples)

        tau = len(x)

        h = [h_prev]
        a = []
        o = []
        if not self.word_domain:
            p = []

        for t in range(0, tau):
            a.append(dot(weights['W'], h[t]) + dot(weights['U'], x[t]) + weights['b'])
            h.append(self.tanh(a[t]))
            if self.word_domain:
                o.append(dot(weights['V'], h[t+1]) + weights['c'])
            else:
                o = dot(weights['V'], h[t+1]) + weights['c']
                p.append(self.softmax(o))

        if self.word_domain:
            return o, h, a
        else:
            return p, h, a

    def synthesize_text(self, x0, h_prev, seq_length, weights={}):
        if not weights:
            weights_tuples = [(self.weights[i], getattr(self, self.weights[i])) for i in range(len(self.weights))]
            weights = dict(weights_tuples)

        self.neurons_of_interest = []
        self.neurons_of_interest_plot = []
        self.neurons_of_interest_plot_intervals = []

        self.load_neuron_intervals()

        print('Predicting sentence from previous ' + self.domain_specification[:-1].lower() + ' "' + x0 + '"')

        table_data = [['Neuron ' + str(self.neurons_of_interest[int(i/2)]), ''] if i % 2 == 0 else ['\n', '\n'] for i in range(2*len(self.neurons_of_interest))]
        table = SingleTable(table_data)

        table.table_data.insert(0, ['Neuron ', 'Predicted sentence'])

        max_width = table.column_max_width(1)

        y_n = [[] for i in range(len(self.neurons_of_interest))]
        y = [[] for i in range(len(self.neurons_of_interest))]

        if self.word_domain:
            x = [array([self.word2vec_model[x0]]).T]
        else:
            sample = copy(x0)

        neuron_activation_map = zeros((self.n_hidden_neurons, seq_length))

        for t in range(seq_length):
            if not self.word_domain:
                x = self.seq_to_one_hot(sample)
            output, h, a = self.forward_prop(x, h_prev, weights)
            h_prev = deepcopy(h[-1])

            neuron_activation_map[:, t] = a[-1][:, 0]
            neuron_activations = a[-1][self.neurons_of_interest]

            if self.word_domain:
                output_word_vector = output[0][:, 0]
                list_most_similar = self.word2vec_model.most_similar(positive=[output_word_vector], topn=200)
                similarities = array([list_most_similar[i][1] for i in range(len(list_most_similar))])
                p = similarities[similarities > 0]/sum(similarities[similarities > 0])
            else:
                p = output[-1]

            cp = cumsum(p)
            rand = random.uniform()
            diff = cp - rand

            sample_index = [i for i in range(len(diff)) if diff[i] > 0]
            if sample_index:
                sample_index = sample_index[0]
            else:
                sample_index = len(diff) - 1

            if self.word_domain:
                sample = list_most_similar[sample_index][0]
            else:
                sample = self.ind_to_char.get(sample_index)

            for i in range(len(self.neurons_of_interest)):

                neuron_activation = nan_to_num(neuron_activations[i, 0])/20.0

                active_color = abs(int(neuron_activation * 255)) if self.dark_theme else 255
                inactive_color = 0 if self.dark_theme else 255 - abs(int(neuron_activation * 255))

                if neuron_activation > 0:
                    red = active_color
                    green = inactive_color
                    blue = inactive_color
                else:
                    red = inactive_color
                    green = inactive_color
                    blue = active_color

                bg.set_style('activationColor', RgbBg(red, green, blue))
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
                    wrapped_string += ' '*(max_width - line_width)*0 + '\n'

            table.table_data[2*i+1][1] = wrapped_string

        max_activation = amax(neuron_activation_map[self.neurons_of_interest, :])
        min_activation = amin(neuron_activation_map[self.neurons_of_interest, :])
        margin = 8
        color_range_width = max_width - len(table.table_data[0][1]) - (len(str(max_activation)) + len(str(min_activation)) + 2)
        color_range = arange(min_activation, max_activation,
                             (max_activation - min_activation) / color_range_width)

        color_range_str = ' '*margin + str(round(min_activation, 1)) + ' '

        for i in range(color_range_width):

            color_range_value = nan_to_num(color_range[i])/20.0

            active_color = abs(int(color_range_value  * 255)) if self.dark_theme else 255
            inactive_color = 0 if self.dark_theme else 255 - abs(int(color_range_value * 255))

            if color_range_value > 0:
                red = active_color
                green = inactive_color
                blue = inactive_color
            else:
                red = inactive_color
                green = inactive_color
                blue = active_color

            bg.set_style('activationColor', RgbBg(red, green, blue))
            colored_indicator = bg.activationColor + ' ' + bg.rs

            color_range_str += colored_indicator

        color_range_str += ' ' + str(round(max_activation, 1))
        table.table_data[0][1] += color_range_str

        table.table_data[1:] = flip(table.table_data[1:], axis=0)

        return table.table, neuron_activation_map, y_n[0]

    def load_neuron_intervals(self):
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
                        self.interval_label_shift = '' # '      '

                        for interval in intervals:
                            if ':' in interval:
                                interval = interval.split(':')
                                interval[0] = str(max(int(interval[0]), 0))
                                interval[-1] = str(min(int(interval[-1]), self.K-1))
                                self.neurons_of_interest_plot.extend(range(int(interval[0]), int(interval[-1]) + 1))
                                self.neurons_of_interest_plot_intervals.append(range(int(interval[0]), int(interval[-1]) + 1))
                                intermediate_range = [i for i in range(int(interval[0])+1, int(interval[-1])) if i%5 == 0]
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

    def back_prop(self, x, y, output, h):
        tau = len(x)

        # Initialize gradients
        for grad, weight in zip(self.gradients, self.weights):
            setattr(self, grad, zeros(getattr(self, weight).shape))

        dLdO = []

        for t in range(tau):
            dLdO.append(output[t].T - y[t].T)
            self.dLdV += dot(dLdO[t].T, h[t+1].T)
            self.dLdC += dLdO[t].T

        dLdAt = zeros((1, self.n_hidden_neurons))

        for t in range(tau - 1, -1, -1):
            dLdHt = dot(dLdO[t], self.V) + dot(dLdAt, self.W)
            dLdAt = dot(dLdHt, diag(1 - h[t+1][:, 0]**2))

            self.dLdW += dot(dLdAt.T, h[t].T)
            self.dLdU += dot(dLdAt.T, x[t].T)
            self.dLdB += dLdAt.T

        if self.clip_gradients:
            for grad in self.gradients:
                setattr(self, grad, maximum(minimum(getattr(self, grad), self.gradient_clip_threshold), -self.gradient_clip_threshold))

    def compute_loss(self, output, y):
        # Cross entropy loss
        tau = len(y)
        loss = 0
        accuracy = 0

        if self.word_domain:
            for t in range(tau):
                loss += .5*sum((output[t] - y[t])**2)
                accuracy += sum(output[t] - y[t])
        else:
            for t in range(tau):
                loss -= sum(log(dot(y[t].T, output[t])))

                p = output[t]
                cp = cumsum(p)
                rand = random.uniform()
                diff = cp - rand

                sample_index = [i for i in range(len(diff)) if diff[i] > 0]
                if sample_index:
                    sample_index = sample_index[0]
                else:
                    sample_index = len(diff) - 1

                accuracy += y[t][sample_index]

        loss = loss/max(float(tau), 1e-6)
        accuracy = float(accuracy)/max(float(tau), 1e-6)

        return loss, accuracy

    def to_one_hot(self, x):
        binarizer = sklearn.preprocessing.LabelBinarizer()
        binarizer.fit(range(max(x.astype(int)) + 1))
        X = array(binarizer.transform(x.astype(int))).T

        return X

    def seq_to_one_hot(self, x):
        X = [array([self.char_to_ind.get(xt)]).T for xt in x]

        return X

    def seq_to_one_hot_matrix(self, x):
        x_ind = self.seq_to_one_hot(x)
        X = concatenate(x_ind, axis=1)

        return X

    def tanh(self, x):
        return (exp(x) - exp(-x))/(exp(x) + exp(-x))

    def softmax(self, s):
        ex_p = exp(s)
        p = ex_p/ex_p.sum()

        return p


def randomize_hyper_parameters(n_configurations, attributes):

    attributes['nEpochs'] = 5
    attributes['weightInit'] = 'He'


    for i in range(n_configurations):
        attributes['n_hidden_neurons'] = 16*int(5*random.rand()+12)
        attributes['eta'] = int(9*random.rand() + 1)*10**(-5 - int(2*random.rand()))

        print('\n')
        print('n: ' + str(attributes['n_hidden_neurons']))
        print('Learning rate: ' + str(attributes['eta']))
        print('\n')

        rnn = VisualizeRNN(attributes)
        rnn.run_vanilla_rnn()


def main():

    attributes = {
        'text_file': '../Corpus/ted_en.zip',  # Corpus file for training
        'dark_theme': True,  # True for dark theme, else light (then terminal text/background color needs to be adjusted)
        'train_model': False,  # True to train model, otherwise inference process is applied for text generation
        'model_directory': 'Vanilla RNN Saved Models/val_loss2.072-val_acc0.445-loss1.335-epoch6-iteration674039-neurons224-eta-9.00e-5/',
        'word_domain': False,  # True for words, False for characters
        'save_sentences': False,  # Save sentences and vocabulary
        'load_sentences': True,  # Load sentences and vocabulary
        'validation_proportion': .02,  # The proportion of data set used for validation
        'corpus_proportion': 1.0,  # The proportion of the corpus used for training and validation
        'ada_grad_sgd': True,  # Stochastic gradient decent, True for ada_grad, False for regular SGD
        'clip_gradients': True,  # True to avoid exploding gradients
        'gradient_clip_threshold': 5,  # Threshold for clipping gradients
        'weight_init': 'Load',  # 'He', 'Load' or 'Random'
        'eta': 9.00e-5,  # Learning rate
        'n_hidden_neurons': 224,  # Number of hidden neurons
        'n_epochs': 10,  # Total number of epochs, each corresponds to (n book characters)/(seq_length) seq iterations
        'seq_length': 25,  # Sequence length of each sequence iteration
        'length_synthesized_text': 300,  # Sequence length of each print of text evolution
        'length_synthesized_text_best': 1000,  # Sequence length of final best sequence, requires save_parameters
        'rms_prop': True,  # Implementation of rms_prop to ada_grad_sgd
        'gamma': 0.9,  # Weight factor of rms_prop
        'save_parameters': False  # Save best weights with corresponding arrays iterations and smooth loss
    }

    # randomize_hyper_parameters(500, attributes)

    rnn_vis = VisualizeRNN(attributes)
    rnn_vis.run_vanilla_rnn()


if __name__ == '__main__':
    random.seed(2)
    main()
    plt.show()
