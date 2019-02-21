'''
@author: John Henry Dahlberg

2019-02-05
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

    def loadVocabulary(self):
        is_binary = self.model_file[-4:] == '.bin'
        print('Loading model "' + self.model_file + '"...')
        word2vec_model = KeyedVectors.load_word2vec_format(self.model_file, binary=is_binary)
        K = size(word2vec_model.vectors, 1)

        words = []

        print('Loading text file "' + self.textFile + '"...')
        if self.textFile[-4:] == '.zip':
            with zipfile.ZipFile(self.textFile, 'r') as z:
                doc = lxml.etree.parse(z.open(z.filelist[0].filename, 'r'))
            print('Extracting words...')
            input_text = '\n'.join(doc.xpath('//content/text()'))
            words.extend(re.findall(r"\w+|[^\w]", input_text))
        else:
            with open(self.textFile, 'r') as f:
                lines = f.readlines()
                print('Extracting words...')
                for line in lines:
                    words.extend(re.findall(r"\w+|[^\w]", line))
                    words.append('\n')

        return word2vec_model, words, K

    def getWords(self, e):
        x_sequence = self.input_sequence[e:e + self.seqLength]
        y_sequence = self.input_sequence[e + 1:e + self.seqLength + 1]

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
                # print("Word '" + y_word + "'" + ' added to model.')
        return x_sequence, y_sequence, x, y

    def synthesizeText(self, x0, hPrev, seqLength, weights={}):
        if not weights:
            weightsTuples = [(self.weights[i], getattr(self, self.weights[i])) for i in range(len(self.weights))]
            weights = dict(weightsTuples)

        self.neuronsOfInterest = []
        self.neuronsOfInterestPlot = []
        self.neuronsOfInterestPlotIntervals = []

        self.loadNeuronIntervals()

        table_data = [['Neuron ' + str(self.neuronsOfInterest[int(i/2)]), ''] if i % 2 == 0 else ['\n', '\n'] for i in range(2*len(self.neuronsOfInterest))]
        table = SingleTable(table_data)
        table.table_data.insert(0, ['Neuron ', 'Predicted sentence from previous ' + self.domain_specification[:-1].lower() + ' "' + x0 + '"'])

        max_width = table.column_max_width(1)

        y_n = [[] for i in range(len(self.neuronsOfInterest))]
        y = [[] for i in range(len(self.neuronsOfInterest))]

        if self.word_domain:
            x = [array([self.word2vec_model[x0]]).T]
        else:
            sample = copy(x0)

        neuron_activation_map = zeros((self.nHiddenNeurons, seqLength))

        for t in range(seqLength):
            if not self.word_domain:
                x = self.seqToOneHot(sample)
            output, h, a = self.forwardProp(x, hPrev, weights)
            hPrev = deepcopy(h[-1])

            neuron_activation_map[:, t] = a[-1][:, 0]
            neuronActivations = a[-1][self.neuronsOfInterest]

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
            sample_index = [i for i in range(len(diff)) if diff[i] > 0][0]

            if self.word_domain:
                sample = list_most_similar[sample_index][0]
            else:
                sample = self.indToChar.get(sample_index)

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


def main():

    attributes = {
        'textFile': 'Data/ted_en.zip',  # Name of book text file, needs to be longer than lengthSynthesizedTextBest
        'model_file': 'Data/glove_short.txt',  # 'Data/glove_840B_300d.txt',  #
        'word_domain': False,  # True for words, False for characters
        'weightInit': 'Load',  # 'He', 'Load' or 'Random'
        'nHiddenNeurons': 'Auto',  # Number of hidden neurons
        'nEpochs': 100,  # Total number of epochs, each corresponds to (n book characters)/(seqLength) seq iterations
        'seqLength': 25,  # Sequence length of each sequence iteration
        'lengthSynthesizedText': 200,  # Sequence length of each print of text evolution
        'lengthSynthesizedTextBest': 1000,  # Sequence length of final best sequence, requires saveParameters
        'saveParameters': False  # Save best weights with corresponding arrays iterations and smooth loss
    }

    if not attributes['adaGradSGD']:
        attributes['eta'] = 0.01*attributes['eta']

    lstm_vis = VisualizeLSTM(attributes)


if __name__ == '__main__':
    random.seed(1)
    main()
    plt.show()
