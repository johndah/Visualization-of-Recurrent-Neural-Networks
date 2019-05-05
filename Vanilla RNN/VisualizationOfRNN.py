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
            stdout_handle = windll.kernel32.GetStdHandle(c_int(-11))
            mode = c_int(0)
            windll.kernel32.GetConsoleMode(c_int(stdout_handle), byref(mode))
            mode = c_int(mode.value | 4)
            windll.kernel32.SetConsoleMode(c_int(stdout_handle), mode)

        if self.word_domain:
            if self.save_sentences:
                self.vocabulary, self.sentences, self.K = self.loadVocabulary()

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
                self.word2vec_model, input_sequence, self.K = self.loadVocabulary()

            # self.word2vec_model, input_sequence, self.K = self.loadVocabulary()
        else:
            input_sequence, self.charToInd, self.indToChar = self.loadCharacters()
            self.K = len(self.indToChar)

        if self.nHiddenNeurons == 'Auto':
            self.nHiddenNeurons = self.K

        n_validation = int(len(input_sequence) * self.validation_proportion)
        n_training = len(input_sequence) - n_validation

        input_sequence = input_sequence[:int(self.corpus_proportion*len(input_sequence))]

        self.input_sequence = input_sequence[:n_training]
        self.input_sequence_validation = input_sequence[n_training:]

        self.weights = ['W', 'V', 'U', 'b', 'c']
        self.gradients = ['dLdW', 'dLdV', 'dLdU', 'dLdB', 'dLdC']
        self.numGradients = ['gradWnum', 'gradVnum', 'gradUnum', 'gradBnum', 'gradCnum']

        self.sizes = [(self.nHiddenNeurons, self.nHiddenNeurons), (self.K, self.nHiddenNeurons), \
                      (self.nHiddenNeurons, self.K), (self.nHiddenNeurons, 1), (self.K, 1)]

        # Weight initialization
        if self.weightInit == 'Load':
            print('Loading weights...')
        else:
            print('Initializing weights...')

        for weight, gradIndex in zip(self.weights, range(len(self.gradients))):
            if self.sizes[gradIndex][1] > 1:
                if self.weightInit == 'Load':
                    self.initSigma = loadtxt(self.model_directory + 'initSigma.txt', unpack=False)
                    setattr(self, weight, array(loadtxt(self.model_directory + 'Weights/' + weight + ".txt", comments="#", delimiter=",", unpack=False)))
                else:
                    if self.weightInit == 'He':
                        self.initSigma = sqrt(2 / sum(self.sizes[gradIndex]))
                    else:
                        self.initSigma = 0.01
                    setattr(self, weight, self.initSigma*random.randn(self.sizes[gradIndex][0], self.sizes[gradIndex][1]))
            else:
                if self.weightInit == 'Load':
                    self.initSigma = loadtxt(self.model_directory + '/initSigma.txt', unpack=False)
                    setattr(self, weight, array([loadtxt(self.model_directory + 'Weights/' + weight + ".txt", comments="#", delimiter=",", unpack=False)]).T)
                else:
                    setattr(self, weight, zeros(self.sizes[gradIndex]))

        if self.weightInit == 'Load':
            self.seqIterations = loadtxt(self.model_directory + 'seqIterations.txt', delimiter=",", unpack=False)
            self.smoothLosses = loadtxt(self.model_directory + 'smoothLosses.txt', delimiter=",", unpack=False)
            self.validationLosses = loadtxt(self.model_directory + 'validationLosses.txt', delimiter=",", unpack=False)
            self.initEpoch = int(self.model_directory.split('epoch')[1][0])
            self.initIteration = 0 # int(self.model_directory.split('iteration')[1].split('-')[0])
            self.nHiddenNeurons = int(self.model_directory.split('neurons')[1][:3])
            self.eta = float(self.model_directory.split('eta-')[1][:8])

        self.x0 = ' '
        self.h0 = zeros((self.nHiddenNeurons, 1))

        self.lossMomentum = 1e-3

    def loadCharacters(self):

        print('Loading text file "' + self.textFile + '"...')
        if self.textFile[-4:] == '.zip':
            with zipfile.ZipFile(self.textFile, 'r') as z:
                doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
            print('Extracting characters...')
            input_text = '\n'.join(doc.xpath('//content/text()'))

        characters = []
        [characters.append(char) for sentences in input_text for char in sentences if char not in characters]
        print('Unique characters:\n' + str(characters))
        k = len(characters)
        indicators = array(range(k))

        indOneHot = self.toOneHot(indicators)

        charToInd = dict((characters[i], array(indOneHot[i])) for i in range(k))
        indToChar = dict((indicators[i], characters[i]) for i in range(k))

        return input_text, charToInd, indToChar

    def loadVocabulary(self):
        is_binary = self.model_file[-4:] == '.bin'
        print('Loading model "' + self.model_file + '"...')
        word2vec_model = KeyedVectors.load_word2vec_format(self.model_file, binary=is_binary)
        K = size(word2vec_model.vectors, 1)

        words = []

        # for subdir, dirs, files in os.walk(self.textFile):
        #     for file in files:
        # print os.path.join(subdir, file)
        # filepath = subdir + os.sep + file

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

        # text_model = Word2Vec(words, size=300, min_count=1)

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

    def getCharacters(self, e, input_sequence, seqLength=None):
        if not seqLength:
            seqLength = self.seqLength

        x_sequence = input_sequence[e:e+seqLength]
        y_sequence = input_sequence[e+1:e+seqLength + 1]
        x = self.seqToOneHot(x_sequence)
        y = self.seqToOneHot(y_sequence)

        return x_sequence, y_sequence, x, y

    def adaGrad(self):
        if self.weightInit == 'Load':
            smoothLoss = self.smoothLosses[-1]
            validationLoss = self.validationLosses[-1]
            lowestValidationLoss = validationLoss
        else:
            smoothLoss = None

        if self.word_domain:
            self.domain_specification = 'Words'
        else:
            self.domain_specification = 'Characters'

        # if self.plotProcess:
        #fig = plt.figure(2)
        constants = 'Max Epochs: ' + str(self.nEpochs) + ' (' + str(len(self.input_sequence)/self.seqLength * self.nEpochs) + ' seq. iter.)' \
                    + '\n# Hidden neurons: ' + str(self.nHiddenNeurons) \
                    + '\nWeight initialization: ' + str(self.weightInit) \
                    + '\n' + r'$\sigma$ = ' + "{:.2e}".format(self.initSigma) \
                    + '\n' + r'$\eta$ = ' + "{:.2e}".format(self.eta) \
                    + '\n' + 'Sequence length: ' + str(self.seqLength) \
                    + '\n#' + self.domain_specification + ' in training text:' + '\n' + str(len(self.input_sequence)) \
                    + '\n' + 'AdaGrad: ' + str(self.adaGradSGD) \
                    + '\n' + 'RMS Prop: ' + str(self.rmsProp)

        if self.rmsProp:
            constants += '\n' + r'$\gamma$ = ' + "{:.2e}".format(self.gamma)

        m = []
        for weight in self.weights:
            m.append(zeros(getattr(self, weight).shape))

        if self.weightInit == 'Load':
            seqIteration = self.seqIterations[-1]
            seqIterations = [s for s in self.seqIterations]
            smoothLosses = [s for s in self.smoothLosses]
            validationLosses = [s for s in self.validationLosses]
        else:
            seqIteration = 0
            seqIterations = []
            smoothLosses = []
            validationLosses = []
        smoothLossesTemp = []
        validationLossesTemp = []
        seqIterationsTemp = []

        start_time = time.time()
        previous_time = start_time
        for epoch in range(self.initEpoch, self.nEpochs):

            hPrev = deepcopy(self.h0)

            # for e in range(0, len(self.bookData)-self.seqLength-1, self.seqLength):
            for e in range(self.initIteration, len(self.input_sequence)-self.seqLength-1, self.seqLength):

                if self.word_domain:
                    x_sequence, y_sequence, x, y = self.getWords(e)
                else:
                    x_sequence, y_sequence, x, y = self.getCharacters(e, self.input_sequence)

                output, h, a = self.forwardProp(x, hPrev)
                self.backProp(x, y, output, h)

                loss, accuracy = self.computeLoss(output, y)
                if not smoothLoss:
                    smoothLoss = loss

                smoothLoss = (1 - self.lossMomentum) * smoothLoss + self.lossMomentum * loss

                if time.time() - previous_time > 1800 or (time.time() - start_time < 5 and time.time() - previous_time > 3) or e >= len(self.input_sequence)-2*self.seqLength-1:
                    seqIterationsTemp.append(seqIteration)
                    smoothLossesTemp.append(smoothLoss)

                    x0 = self.input_sequence[e]

                    if self.word_domain:
                        x_sequence, y_sequence, x, y = self.getWords(e)
                    else:
                        x_sequence, y_sequence, x, y = self.getCharacters(0, self.input_sequence_validation, self.lengthSynthesizedText)

                    output, h, a = self.forwardProp(x, hPrev)

                    validationLoss, accuracy = self.computeLoss(output, y)
                    lowestValidationLoss = copy(validationLoss)
                    validationLossesTemp.append(validationLoss)

                    table, neuron_activation_map, inputs = self.synthesizeText(x0, hPrev, self.lengthSynthesizedText)

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

                    time_passed = time.time() - start_time
                    estimated_total_time = time_passed/(max(e, 1)/len(self.input_sequence))
                    remaining_time = estimated_total_time - time_passed
                    previous_time = time.time()
                    print('\nSequence iteration: ' + str(seqIteration) + ', Epoch: ' + str(epoch)
                          + ', Epoch ETA: ' + str(datetime.timedelta(seconds=int(remaining_time)))
                          + ', Epoch process: ' + str('{0:.2f}'.format(e/len(self.input_sequence)*100)) + '%'
                          + ', Training loss: ' + str('{0:.2f}'.format(smoothLoss)) + ', Neuron of interest: '
                          + ', Validation loss: ' + str('{0:.2f}'.format(validationLoss)) + ', Neuron of interest: ' +
                          str(self.neuronsOfInterest) + '(/' + str(self.nHiddenNeurons) + ')')

                    print(table)

                    if validationLoss <= lowestValidationLoss:
                        seqIterations += seqIterationsTemp
                        smoothLosses += smoothLossesTemp
                        validationLosses += validationLossesTemp
                        smoothLossesTemp = []
                        validationLossesTemp = []
                        seqIterationsTemp = []
                        lowestValidationLoss = copy(smoothLoss)
                        hPrevBest = copy(hPrev)
                        x0Best = copy(x0)

                        if self.saveParameters:
                            state = "val_loss%.3f-val_acc%.3f-loss%.3f-epoch%d-iteration%d-neurons%d-eta-"%(validationLoss, accuracy, loss, epoch, int(e/self.seqLength), self.nHiddenNeurons) + '{:.2e}'.format(Decimal(self.eta)) + "/"
                            try:
                                for weight in self.weights:
                                    #if not os.path.isdir('./' + fileName):
                                    fileName = 'Models/' + state + 'Weights/' + weight + '.txt'
                                    os.makedirs(os.path.dirname(fileName), exist_ok=True)
                                    savetxt(fileName, getattr(self, weight), delimiter=',')

                                os.makedirs(os.path.dirname('Models/' + state + 'initSigma.txt'), exist_ok=True)
                                savetxt('Models/' + state + 'initSigma.txt', array([[self.initSigma]]))
                                os.makedirs(os.path.dirname('Models/' + state + 'seqIterations.txt'), exist_ok=True)
                                savetxt('Models/' + state + 'seqIterations.txt', seqIterations, delimiter=',')
                                os.makedirs(os.path.dirname('Models/' + state + 'smoothLosses.txt'), exist_ok=True)
                                savetxt('Models/' + state + 'smoothLosses.txt', smoothLosses, delimiter=',')
                                os.makedirs(os.path.dirname('Models/' + state + 'validationLosses.txt'), exist_ok=True)
                                savetxt('Models/' + state + 'validationLosses.txt', validationLosses, delimiter=',')

                                #savetxt('Models/Weights/' + fileName + 'h0.txt', hPrevBest, delimiter=',')

                                # with open('Weights/x0.txt', 'w') as f:
                                #    f.write(x0Best)

                            except Exception as ex:
                                print(ex)

                    with open('config.txt', 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.split('#')[0]
                            if 'plotProcess:' in line:
                                self.plotProcess = ''.join(line.split()).split(':')[1] == 'True'
                                break
                            else:
                                self.plotProcess = False

                    if self.plotProcess:
                        fig = plt.figure(2)
                        plt.clf()
                        ax = fig.add_subplot(111)
                        fig.subplots_adjust(top=0.85)
                        anchored_text = AnchoredText(constants, loc=1)
                        ax.add_artist(anchored_text)

                        plt.title(self.domain_specification[:-1] + ' prediction learning curve of Recurrent Neural Network')
                        plt.ylabel('Smooth loss')
                        plt.xlabel('Sequence iteration')
                        plt.plot(seqIterations+seqIterationsTemp, smoothLosses+smoothLossesTemp, LineWidth=2, label='Training')
                        plt.plot(seqIterations+seqIterationsTemp, validationLosses+validationLossesTemp, LineWidth=2,  label='Validation')
                        plt.grid()

                        plt.legend(loc='upper left')

                epsilon = 1e-10

                if self.rmsProp:
                    cM = self.gamma
                    cG = 1 - self.gamma
                else:
                    cM, cG, = 1, 1

                for grad, weight, gradIndex in zip(self.gradients, self.weights, range(len(self.gradients))):
                    if self.adaGradSGD:
                        m[gradIndex] = cM * m[gradIndex] + cG*getattr(self, grad)**2
                        sqrtInvM = (m[gradIndex]+epsilon)**-0.5
                        updatedWeight = getattr(self, weight) - self.eta * multiply(sqrtInvM, getattr(self, grad))
                    else:
                        updatedWeight = deepcopy(getattr(self, weight)) - self.eta * deepcopy(getattr(self, grad))

                    setattr(self, weight, updatedWeight)

                hPrev = deepcopy(h[-1])

                seqIteration += 1

            if False and self.saveParameters:
                bestWeights = []

                for weight in self.weights[:3]:
                    bestWeights.append(array(loadtxt('Weights/' + weight + ".txt", comments="#", delimiter=",", unpack=False)))
                for weight in self.weights[3:]:
                    bestWeights.append(array([loadtxt('Weights/' + weight + ".txt", comments="#", delimiter=",", unpack=False)]).T)

                weightsTuples = [(self.weights[i], bestWeights[i]) for i in range(len(self.weights))]

                weights = dict(weightsTuples)
                bestSequence = self.synthesizeText(x0Best, hPrevBest, self.lengthSynthesizedTextBest, weights)
                print('\n\nEpoch: ' + str(epoch) + ', Lowest validation loss: ' + str(lowestValidationLoss))
                print('    ' + bestSequence)

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
            array([abs(mean(neuron_feature_extracted_map, axis=1))]).T / array([abs(mean(neuron_activation_rows, axis=1))]).T,
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

    def forwardProp(self, x, hPrev, weights={}):
        if not weights:
            weightsTuples = [(self.weights[i], getattr(self, self.weights[i])) for i in range(len(self.weights))]
            weights = dict(weightsTuples)

        tau = len(x)

        h = [hPrev]
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

            sample_index = [i for i in range(len(diff)) if diff[i] > 0]
            if sample_index:
                sample_index = sample_index[0]
            else:
                sample_index = len(diff) - 1

            if self.word_domain:
                sample = list_most_similar[sample_index][0]
            else:
                sample = self.indToChar.get(sample_index)

            for i in range(len(self.neuronsOfInterest)):

                neuronActivation = nan_to_num(neuronActivations[i, 0])

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

            color_range_value = nan_to_num(color_range[i])

            if color_range_value > 0:
                bg.set_style('activationColor', RgbBg(int(color_range_value * 255), 0, 0))
            else:
                bg.set_style('activationColor', RgbBg(0, 0, int(abs(color_range_value) * 255)))

            colored_indicator = bg.activationColor + ' ' + bg.rs

            color_range_str += colored_indicator

        color_range_str += str(round(max_activation, 1))
        table.table_data[0][1] += color_range_str

        return table.table, neuron_activation_map, y_n[0]

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
                                interval[-1] = str(min(int(interval[-1]), self.K-1))
                                self.neuronsOfInterestPlot.extend(range(int(interval[0]), int(interval[-1]) + 1))
                                self.neuronsOfInterestPlotIntervals.append(range(int(interval[0]), int(interval[-1]) + 1))
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
                                self.neuronsOfInterestPlot.append(int(interval))
                                self.neuronsOfInterestPlotIntervals.append([int(interval)])
                                self.intervals_to_plot.append(interval)
                                self.interval_limits.append(int(interval))
                        self.interval_limits = array(self.interval_limits)

    def backProp(self, x, y, output, h):
        tau = len(x)

        # Initialize gradients
        for grad, weight in zip(self.gradients, self.weights):
            setattr(self, grad, zeros(getattr(self, weight).shape))

        dLdO = []

        for t in range(tau):
            # dLdO.append(p[t].T - y[t].T)
            dLdO.append(output[t].T - y[t].T)
            self.dLdV += dot(dLdO[t].T, h[t+1].T)
            self.dLdC += dLdO[t].T

        dLdAt = zeros((1, self.nHiddenNeurons))

        for t in range(tau - 1, -1, -1):
            dLdHt = dot(dLdO[t], self.V) + dot(dLdAt, self.W)
            dLdAt = dot(dLdHt, diag(1 - h[t+1][:, 0]**2))

            self.dLdW += dot(dLdAt.T, h[t].T)
            self.dLdU += dot(dLdAt.T, x[t].T)
            self.dLdB += dLdAt.T

        # Clip gradients
        if self.clipGradients:
            for grad in self.gradients:
                setattr(self, grad, maximum(minimum(getattr(self, grad), self.gradientClipThreshold), -self.gradientClipThreshold))

    def computeLoss(self, output, y):
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

        loss = loss/float(tau)
        accuracy = float(accuracy)/float(tau)

        return loss, accuracy

    def toOneHot(self, x):
        binarizer = sklearn.preprocessing.LabelBinarizer()
        binarizer.fit(range(max(x.astype(int)) + 1))
        X = array(binarizer.transform(x.astype(int))).T

        return X

    def seqToOneHot(self, x):
        X = [array([self.charToInd.get(xt)]).T for xt in x]

        return X

    def seqToOneHotMatrix(self, x):
        xInd = self.seqToOneHot(x)
        X = concatenate(xInd, axis=1)

        return X

    def tanh(self, x):
        return (exp(x) - exp(-x))/(exp(x) + exp(-x))

    def softmax(self, s):
        exP = exp(s)
        p = exP/exP.sum()

        return p

    def computeGradsNumSlow(self, x, y, hPrev):
        hStep = 1e-4

        for numGrad, weight in zip(self.numGradients, self.weights):
            setattr(self, numGrad, zeros(getattr(self, weight).shape))

        for numGrad, weight, gradIndex in zip(self.numGradients, self.weights, range(len(self.numGradients))):
            if getattr(self, weight).shape[1] == 1:
                for i in range(getattr(self, weight).shape[0]):
                    weightTry = deepcopy(getattr(self, weight))
                    weightTry[i, 0] -= hStep
                    weightsTuples = [(self.weights[i], deepcopy(getattr(self, self.weights[i]))) for i in range(len(self.weights))]
                    weightsTuples[gradIndex] = (self.weights[gradIndex], weightTry)
                    weights = dict(weightsTuples)
                    o, h, a = self.forwardProp(x, hPrev, weights)
                    c1, accuracy = self.computeLoss(o, y)

                    weightTry = deepcopy(getattr(self, weight))
                    weightTry[i, 0] += hStep
                    weightsTuples = [(self.weights[i], deepcopy(getattr(self, self.weights[i]))) for i in range(len(self.weights))]
                    weightsTuples[gradIndex] = (self.weights[gradIndex], weightTry)
                    weights = dict(weightsTuples)
                    o, h, a = self.forwardProp(x, hPrev, weights)
                    c2, accuracy = self.computeLoss(o, y)

                    updatedNumGrad = deepcopy(getattr(self, numGrad))
                    updatedNumGrad[i, 0] = (c2 - c1) / (2 * hStep)
                    setattr(self, numGrad, updatedNumGrad)
            else:
                iS = [0, 1, 2, getattr(self, weight).shape[0] - 3, getattr(self, weight).shape[0] - 2, getattr(self, weight).shape[0] - 1]
                jS = [0, 1, 2, getattr(self, weight).shape[1] - 3, getattr(self, weight).shape[1] - 2, getattr(self, weight).shape[1] - 1]
                for i in iS:
                    for j in jS:
                        weightTry = deepcopy(getattr(self, weight))
                        weightTry[i, j] -= hStep
                        weightsTuples = [(self.weights[i], deepcopy(getattr(self, self.weights[i]))) for i in
                                         range(len(self.weights))]
                        weightsTuples[gradIndex] = (self.weights[gradIndex], weightTry)
                        weights = dict(weightsTuples)
                        o, h, a = self.forwardProp(x, hPrev, weights)
                        c1, accuracy = self.computeLoss(o, y)

                        weightTry = deepcopy(getattr(self, weight))
                        weightTry[i, j] += hStep
                        weightsTuples = [(self.weights[i], copy(getattr(self, self.weights[i]))) for i in
                                         range(len(self.weights))]
                        weightsTuples[gradIndex] = (self.weights[gradIndex], weightTry)
                        weights = dict(weightsTuples)
                        o, h, a = self.forwardProp(x, hPrev, weights)
                        c2, accuracy = self.computeLoss(o, y)

                        updatedNumGrad = deepcopy(getattr(self, numGrad))
                        updatedNumGrad[i, j] = (c2 - c1) / (2 * hStep)
                        setattr(self, numGrad, updatedNumGrad)

    def testComputedGradients(self):

        if self.word_domain:
            x_sequence, y_sequence, x, y = self.getWords(0)
        else:
            x_sequence, y_sequence, x, y = self.getCharacters(0, self.input_sequence)

        epsilon = 1e-20
        hPrev = self.h0
        output, h, a = self.forwardProp(x, hPrev)
        self.backProp(x, y, output, h)

        differenceGradients = []
        differenceGradientsSmall = []
        self.computeGradsNumSlow(x, y, hPrev)

        for numGrad, grad, gradIndex in zip(self.numGradients, self.gradients, range(len(self.numGradients))):
            gradObj = deepcopy(getattr(self, grad))
            numGradObj = deepcopy(getattr(self, numGrad))

            differenceGradients.append(abs(gradObj - numGradObj) / maximum(epsilon, (abs(gradObj) + abs(numGradObj))))

            if gradObj.shape[1] > 1:
                # Only calculate first and last three rows and columns
                differenceGradientsSmall.append(zeros((6, 6)))

                iS = [0, 1, 2, gradObj.shape[0] - 3, gradObj.shape[0] - 2, gradObj.shape[0] - 1]
                jS = [0, 1, 2, gradObj.shape[1] - 3, gradObj.shape[1] - 2, gradObj.shape[1] - 1]

                for i in range(6):
                    for j in range(6):
                        differenceGradientsSmall[gradIndex][i, j] = "{:.2e}".format(differenceGradients[gradIndex][iS[i], jS[j]])
            else:
                differenceGradientsSmall.append(zeros((1, 6)))

                bS = [0, 1, 2, gradObj.shape[0] - 3, gradObj.shape[0] - 2, gradObj.shape[0] - 1]

                for i in range(6):
                    differenceGradientsSmall[gradIndex][0, i] = "{:.2e}".format(differenceGradients[gradIndex][bS[i]][0])

            print('\nAbsolute differences gradient ' + grad + ':')
            print(differenceGradientsSmall[gradIndex])
            # print(pMatrix(differenceGradientsSmall[gradIndex]))


def pMatrix(array):
    rows = str(array).replace('[', '').replace(']', '').splitlines()
    rowString = [r'\begin{pmatrix}']
    for row in rows:
        rowString += [r'  \num{' + r'} & \num{'.join(row.split()) + r'}\\']

    rowString += [r'\end{pmatrix}']

    return '\n'.join(rowString)


def randomize_hyper_parameters(n_configurations, attributes):

    models = ['Models/val_loss37.407713-loss39.718531-epoch2-iteration382960-neurons224-eta-0.000090/',
              'Models/val_loss41.122846-loss42.047568-epoch2-iteration222329-neurons192-eta-0.000070/',
              'Models/val_loss42.466916-loss42.452354-epoch1-iteration889785-neurons192-eta-0.000070/']

    for model in models:
        print('Continuing on model' + model)

        attributes['model_directory'] = model
        attributes['nEpochs'] = 1
        attributes['weightInit'] = 'Load'

        rnn = VisualizeRNN(attributes)
        rnn.adaGrad()

    attributes['nEpochs'] = 5
    attributes['weightInit'] = 'He'

    for i in range(n_configurations):
        attributes['nHiddenNeurons'] = 16*int(5*random.rand()+12)
        attributes['eta'] = int(9*random.rand() + 1)*10**(-5 - int(2*random.rand()))

        print('\n')
        print('n: ' + str(attributes['nHiddenNeurons']))
        print('Learning rate: ' + str(attributes['eta']))
        print('\n')

        rnn = VisualizeRNN(attributes)
        rnn.adaGrad()

def main():

    attributes = {
        'textFile': 'Data/ted_en.zip',  # Name of book text file, needs to be longer than lengthSynthesizedTextBest
        'model_file': 'Data/glove_short.txt',  # 'Data/glove_840B_300d.txt',  #
        'word_domain': False,  # True for words, False for characters
        'save_sentences': False,  # Save sentences and vocabulary
        'load_sentences': True,  # Load sentences and vocabulary
        'validation_proportion': .02,  # The proportion of data set used for validation
        'corpus_proportion': 1,  # The proportion of the corpus used for training and validation
        'adaGradSGD': True,  # Stochastic gradient decent, True for adaGrad, False for regular SGD
        'clipGradients': True,  # True to avoid exploding gradients
        'weightInit': 'He',  # 'He', 'Load' or 'Random'
        'eta': 5e-4,  # Learning rate
        'gradientClipThreshold': 5,  # Threshold for clipping gradients
        'nHiddenNeurons': 10,  # Number of hidden neurons
        'nEpochs': 3,  # Total number of epochs, each corresponds to (n book characters)/(seqLength) seq iterations
        'seqLength': 25,  # Sequence length of each sequence iteration
        'lengthSynthesizedText': 200,  # Sequence length of each print of text evolution
        'lengthSynthesizedTextBest': 1000,  # Sequence length of final best sequence, requires saveParameters
        'rmsProp': True,  # Implementation of rmsProp to adaGradSGD
        'gamma': 0.9,  # Weight factor of rmsProp
        'saveParameters': True  # Save best weights with corresponding arrays iterations and smooth loss
    }

    if not attributes['adaGradSGD']:
        attributes['eta'] = 0.01*attributes['eta']

    # model_directory = './LSTM Saved Models/Checkpoints/'
    # model = model_directory + os.listdir(model_directory)[-1]
    # print('Loading LSTM model ' + model + '...')


    randomize_hyper_parameters(500, attributes)
    # rnn = VisualizeRNN(attributes)
    # rnn.testComputedGradients()
    # rnn.adaGrad()
    print('Finished!')


if __name__ == '__main__':
    # random.seed(1)
    main()
    plt.show()
