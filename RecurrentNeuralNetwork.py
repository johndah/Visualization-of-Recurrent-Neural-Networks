'''
@author: John Henry Dahlberg

2019-02-05
'''

from __future__ import print_function
import os
import sklearn.preprocessing
from numpy import *
from copy import *
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import platform
from sty import bg, RgbBg
from gensim.models import KeyedVectors, Word2Vec
from ctypes import windll, c_int, byref
import re
from textwrap import wrap
from terminaltables import SingleTable


class RecurrentNeuralNetwork(object):

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

        if self.wordDomain:
            self.word2vec_model, self.words, self.K = self.loadVocabulary()
        else:
            self.bookData, self.charToInd, self.indToChar = self.loadCharacters()
            # self.bookData = bookData
            # self.charToInd = charToInd
            # self.indToChar = indToChar
            self.K = len(self.indToChar)

        # self.x0 = '.'

        self.weights = ['W', 'V', 'U', 'b', 'c']
        self.gradients = ['dLdW', 'dLdV', 'dLdU', 'dLdB', 'dLdC']
        self.numGradients = ['gradWnum', 'gradVnum', 'gradUnum', 'gradBnum', 'gradCnum']

        self.sizes = [(self.nHiddenNeurons, self.nHiddenNeurons), (self.K, self.nHiddenNeurons), \
                      (self.nHiddenNeurons, self.K), (self.nHiddenNeurons, 1), (self.K, 1)]

        # Weight initialization
        for weight, gradIndex in zip(self.weights, range(len(self.gradients))):
            if self.sizes[gradIndex][1] > 1:
                if self.weightInit == 'Load':
                    self.initSigma = loadtxt('initSigma.txt', unpack=False)
                    setattr(self, weight, array(loadtxt(weight + ".txt", comments="#", delimiter=",", unpack=False)))
                else:
                    if self.weightInit == 'He':
                        self.initSigma = sqrt(2 / sum(self.sizes[gradIndex]))
                    else:
                        self.initSigma = 0.01
                    setattr(self, weight, self.initSigma*random.randn(self.sizes[gradIndex][0], self.sizes[gradIndex][1]))
            else:
                if self.weightInit == 'Load':
                    self.initSigma = loadtxt('initSigma.txt', unpack=False)
                    setattr(self, weight, array([loadtxt(weight + ".txt", comments="#", delimiter=",", unpack=False)]).T)
                else:
                    setattr(self, weight, zeros(self.sizes[gradIndex]))

        if self.weightInit == 'Load':
            self.seqIterations = loadtxt('seqIterations.txt', delimiter=",", unpack=False)
            self.smoothLosses = loadtxt('smoothLosses.txt', delimiter=",", unpack=False)
            self.h0 = array([loadtxt('h0.txt', unpack=False)]).T
            try:
                with open('x0.txt', 'r') as f:
                    self.x0 = f.readline()[0]
            except Exception as ex:
                print(ex)
        else:
            self.x0 = ' '
            self.h0 = zeros((self.nHiddenNeurons, 1))

        self.lossMomentum = 1e-3

    def loadCharacters(self):
        with open(self.textFile, 'r') as f:
            lines = f.readlines()
        bookData = ''.join(lines)

        characters = []
        [characters.append(char) for sentences in lines for char in sentences if char not in characters]
        print('Unique characters:\n' + str(characters))
        k = len(characters)
        indicators = array(range(k))

        indOneHot = self.toOneHot(indicators)

        charToInd = dict((characters[i], array(indOneHot[i])) for i in range(k))
        indToChar = dict((indicators[i], characters[i]) for i in range(k))

        return bookData, charToInd, indToChar

    def loadVocabulary(self):
        is_binary = self.model_file[-4:] == '.bin'
        word2vec_model = KeyedVectors.load_word2vec_format(self.model_file, binary=is_binary)
        K = size(word2vec_model.vectors, 1)

        words = []

        directory = os.fsencode(self.textFile)

        for subdir, dirs, files in os.walk(self.textFile):
            for file in files:
                # print os.path.join(subdir, file)
                filepath = subdir + os.sep + file

                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        words.extend(re.findall(r"\w+|[^\w]", line))
                        words.append('\n')

        # text_model = Word2Vec(words, size=300, min_count=1)

        return word2vec_model, words, K

    def getWords(self, e):
        x_words = self.words[e:e + self.seqLength]
        y_words = self.words[e + 1:e + self.seqLength + 1]

        x = []
        y = []

        for i in range(len(x_words)):
            x_word = x_words[i]
            y_word = y_words[i]
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
        return x_words, y_words, x, y


    def adaGrad(self):
        if self.weightInit == 'Load':
            smoothLoss = self.smoothLosses[-1]
            lowestSmoothLoss = smoothLoss
        else:
            smoothLoss = None

        if self.plotProcess:
            fig = plt.figure()
            constants = 'Max Epochs: ' + str(self.nEpochs) + ' (' + str(len(self.words)/self.seqLength * self.nEpochs) + ' seq. iter.)' \
                        + '\n# Hidden neurons: ' + str(self.nHiddenNeurons) \
                        + '\nWeight initialization: ' + str(self.weightInit) \
                        + '\n' + r'$\sigma$ = ' + "{:.2e}".format(self.initSigma) \
                        + '\n' + r'$\eta$ = ' + "{:.2e}".format(self.eta) \
                        + '\n' + 'Sequence length: ' + str(self.seqLength) \
                        + '\n' + '# training words in text:' + '\n' + str(len(self.words)) \
                        + '\n' + 'AdaGrad: ' + str(self.adaGradSGD) \
                        + '\n' + 'RMS Prop: ' + str(self.rmsProp)

            if self.rmsProp:
                constants += '\n' + r'$\gamma$ = ' + "{:.2e}".format(self.gamma) \

        m = []
        for weight in self.weights:
            m.append(zeros(getattr(self, weight).shape))

        if self.weightInit == 'Load':
            seqIteration = self.seqIterations[-1]
            seqIterations = [s for s in self.seqIterations];
            smoothLosses = [s for s in self.smoothLosses];
        else:
            seqIteration = 0
            seqIterations = []
            smoothLosses = []
        smoothLossesTemp = []
        seqIterationsTemp = []

        for epoch in range(0, self.nEpochs):

            hPrev = deepcopy(self.h0)

            # for e in range(0, len(self.bookData)-self.seqLength-1, self.seqLength):
            for e in range(0, len(self.words)-self.seqLength-1, self.seqLength):

                # xChars = self.bookData[e:e+self.seqLength]
                # yChars = self.bookData[e+1:e+self.seqLength + 1]
                # x = self.seqToOneHot(xChars)
                # y = self.seqToOneHot(yChars)

                x_words, y_words, x, y = self.getWords(e)

                '''
                x_words = self.words[e:e+self.seqLength]
                y_words = self.words[e+1:e+self.seqLength+1]

                x = []
                y = []

                for i in range(len(x_words)):
                    x_word = x_words[i]
                    y_word = y_words[i]
                    try:
                        x.append(array([self.word2vec_model[x_word]]).T)
                    except KeyError:
                        self.word2vec_model[x_word] = random.uniform(-0.25, 0.25, self.K)
                        x.append(array([self.word2vec_model[x_word]]).T)
                        print("x word '" + x_word + "'" + ' added to model.')
                    try:
                        y.append(array([self.word2vec_model[y_word]]).T)
                    except KeyError:
                        self.word2vec_model[y_word] = random.uniform(-0.25, 0.25, self.K)
                        y.append(array([self.word2vec_model[y_word]]).T)
                        # print("label '" + y_word + "'" + ' added to model.')
                '''

                o, h, a = self.forwardProp(x, hPrev)
                self.backProp(x, y, o, h)

                loss = self.computeLoss(o, y)
                if not smoothLoss:
                    smoothLoss = loss
                    lowestSmoothLoss = copy(smoothLoss)

                smoothLoss = (1 - self.lossMomentum) * smoothLoss + self.lossMomentum * loss

                if e % (self.seqLength*3e2) == 0:
                    seqIterationsTemp.append(seqIteration)
                    smoothLossesTemp.append(smoothLoss)

                    # x0 = self.bookData[e]
                    x0 = self.words[e]

                    table = self.synthesizeText(x0, hPrev, self.lengthSynthesizedText)

                    print('\nSequence iteration: ' + str(seqIteration) + ', Epoch: ' + str(epoch)
                          + ', Epoch process: ' + str('{0:.2f}'.format(e/len(self.words)*100)) + '%'
                          + ', Smooth loss: ' + str('{0:.2f}'.format(smoothLoss)) + ', Neuron of interest: ' +
                          str(self.neuronsOfInterest) + '(/' + str(self.nHiddenNeurons) + ')')
                    #print('    ' + sequence)
                    print(table)

                    if smoothLoss < lowestSmoothLoss:
                        seqIterations += seqIterationsTemp
                        smoothLosses += smoothLossesTemp
                        smoothLossesTemp = []
                        seqIterationsTemp = []
                        lowestSmoothLoss = copy(smoothLoss)
                        hPrevBest = copy(hPrev)
                        x0Best = copy(x0)
                        if self.saveParameters:
                            try:
                                for weight in self.weights:
                                    savetxt(weight + '.txt', getattr(self, weight), delimiter=',')

                                savetxt('initSigma.txt', array([[self.initSigma]]))
                                savetxt('seqIterations.txt', seqIterations, delimiter=',')
                                savetxt('smoothLosses.txt', smoothLosses, delimiter=',')
                                savetxt('h0.txt', hPrevBest, delimiter=',')
                                # savetxt('x0.txt', x0Best, delimiter=',', fmt='c')
                                with open('x0.txt', 'w') as f:
                                    f.write(x0Best)

                            except Exception as ex:
                                print(ex)

                    if self.plotProcess:
                        plt.clf()
                        ax = fig.add_subplot(111)
                        fig.subplots_adjust(top=0.85)
                        anchored_text = AnchoredText(constants, loc=1)
                        ax.add_artist(anchored_text)

                        plt.title('Text synthesization learning curve of Recurrent Neural Network')
                        plt.ylabel('Smooth loss')
                        plt.xlabel('Sequence iteration')
                        plt.plot(seqIterations+seqIterationsTemp, smoothLosses+smoothLossesTemp, LineWidth=2)
                        plt.grid()

                        plt.pause(0.1)

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

            if self.saveParameters:
                bestWeights = []

                for weight in self.weights[:3]:
                    bestWeights.append(array(loadtxt(weight + ".txt", comments="#", delimiter=",", unpack=False)))
                for weight in self.weights[3:]:
                    bestWeights.append(array([loadtxt(weight + ".txt", comments="#", delimiter=",", unpack=False)]).T)

                weightsTuples = [(self.weights[i], bestWeights[i]) for i in range(len(self.weights))]

                weights = dict(weightsTuples)
                bestSequence = self.synthesizeText(x0Best, hPrevBest, self.lengthSynthesizedTextBest, weights)
                print('\n\nEpoch: ' + str(epoch) + ', Lowest smooth loss: ' + str(lowestSmoothLoss))
                print('    ' + bestSequence)

    def forwardProp(self, x, hPrev, weights={}):
        if not weights:
            weightsTuples = [(self.weights[i], getattr(self, self.weights[i])) for i in range(len(self.weights))]
            weights = dict(weightsTuples)

        tau = len(x)

        h = [hPrev]
        a = []
        o = []
        p = []
        for t in range(0, tau):
            a.append(dot(weights['W'], h[t]) + dot(weights['U'], x[t]) + weights['b'])
            h.append(self.tanh(a[t]))
            o.append(dot(weights['V'], h[t+1]) + weights['c'])
            #p.append(self.softmax(o))

        #return p, h, a
        return o, h, a

    def synthesizeText(self, x0, hPrev, seqLength, weights={}):
        if not weights:
            weightsTuples = [(self.weights[i], getattr(self, self.weights[i])) for i in range(len(self.weights))]
            weights = dict(weightsTuples)

        table_data = [['Neuron ' + str(self.neuronsOfInterest[int(i/2)]), ''] if i % 2 == 0 else ['\n', '\n'] for i in range(2*len(self.neuronsOfInterest))]
        table = SingleTable(table_data)
        table.table_data.insert(0, ['Neuron ', 'Predicted sentence from previous word "' + x0 + '"'])

        max_width = table.column_max_width(1)

        y_n = [[] for i in range(len(self.neuronsOfInterest))]
        y = [[] for i in range(len(self.neuronsOfInterest))]

        x = [array([self.word2vec_model[x0]]).T]

        for t in range(seqLength):
            # x = self.seqToOneHot(xChar)
            o, h, a = self.forwardProp(x, hPrev, weights)
            hPrev = deepcopy(h[-1])

            neuronActivations = a[0][self.neuronsOfInterest]/max(abs(a[0]))

            output_word_vector = o[0][:, 0]
            list_most_similar = self.word2vec_model.most_similar(positive=[output_word_vector], topn=100)
            sample_index = min(int(abs(30*random.randn())), 99)
            sample_word = list_most_similar[sample_index][0]

            #cp = cumsum(p[-1])
            #rand = random.uniform()
            #diff = cp - rand
            #sample = [i for i in range(len(diff)) if diff[i] > 0][0]
            #xChar = self.indToChar.get(sample)


            for i in range(len(self.neuronsOfInterest)):

                neuronActivation = neuronActivations[i, 0]

                if neuronActivation > 0:
                    bg.set_style('activationColor', RgbBg(0, int(neuronActivation * 255), 0))
                else:
                    bg.set_style('activationColor', RgbBg(int(abs(neuronActivation) * 255), 0, 0))

                coloredWord = bg.activationColor + sample_word + bg.rs

                y_n[i].append(sample_word)
                y[i].append(coloredWord)


        # sequences = []
        for i in range(len(self.neuronsOfInterest)):
            # sequences.append(''.join(y[i]))

            wrapped_string = ''
            line_width = 0
            n_characters = 0

            while n_characters < len(''.join(y_n[i])):
                for j in range(len(y[i])):
                    a = y[i][j]
                    if '\n' in a:
                        wrapped_string += a.split('\n')[0] + '\\n' + a.split('\n')[1] + '\n'
                        #break
                        n_characters += line_width
                        line_width = 0
                        wrapped_string += ' ' * (max_width - line_width) * 0 + '\n'
                    else:
                       wrapped_string += ''.join(y[i][j])

                    line_width += len(y_n[i][j])
                    if line_width > max_width - 10:
                        #break
                        n_characters += line_width
                        line_width = 0
                        wrapped_string += ' '*(max_width - line_width)*0 + '\n'

            table.table_data[2*i+1][1] = wrapped_string

        return table.table

    def backProp(self, x, y, o, h):
        tau = len(x)

        # Initialize gradients
        for grad, weight in zip(self.gradients, self.weights):
            setattr(self, grad, zeros(getattr(self, weight).shape))

        dLdO = []

        for t in range(tau):
            # dLdO.append(p[t].T - y[t].T)
            dLdO.append(-y[t].T + o[t].T)
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

    def computeLoss(self, o, y):
        # Cross entropy loss
        tau = len(y)
        loss = 0
        for t in range(tau):
            # dotprod = dot(y[t].T, p[t])
            # logdot = log(dotprod)

            # loss -= sum(log(dot(y[t].T, p[t])))
            loss += .5*sum((o[t] - y[t])**2)

        return loss

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
                    c1 = self.computeLoss(o, y)

                    weightTry = deepcopy(getattr(self, weight))
                    weightTry[i, 0] += hStep
                    weightsTuples = [(self.weights[i], deepcopy(getattr(self, self.weights[i]))) for i in range(len(self.weights))]
                    weightsTuples[gradIndex] = (self.weights[gradIndex], weightTry)
                    weights = dict(weightsTuples)
                    o, h, a = self.forwardProp(x, hPrev, weights)
                    c2 = self.computeLoss(o, y)

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
                        c1 = self.computeLoss(o, y)

                        weightTry = deepcopy(getattr(self, weight))
                        weightTry[i, j] += hStep
                        weightsTuples = [(self.weights[i], copy(getattr(self, self.weights[i]))) for i in
                                         range(len(self.weights))]
                        weightsTuples[gradIndex] = (self.weights[gradIndex], weightTry)
                        weights = dict(weightsTuples)
                        o, h, a = self.forwardProp(x, hPrev, weights)
                        c2 = self.computeLoss(o, y)

                        updatedNumGrad = deepcopy(getattr(self, numGrad))
                        updatedNumGrad[i, j] = (c2 - c1) / (2 * hStep)
                        setattr(self, numGrad, updatedNumGrad)

    def testComputedGradients(self):
        #xChars = self.bookData[:self.seqLength]
        #yChars = self.bookData[1:self.seqLength+1]
        #x = self.seqToOneHot(xChars)
        #y = self.seqToOneHot(yChars)

        x_words, y_words, x, y = self.getWords(0)

        epsilon = 1e-20
        hPrev = self.h0
        o, h, a = self.forwardProp(x, hPrev)
        self.backProp(x, y, o, h)

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


def main():

    attributes = {
        'textFile': 'Data/bbc', # 'LordOfTheRings.txt',  # Name of book text file, needs to be longer than lengthSynthesizedTextBest
        'model_file': 'Data/GoogleNews-vectors-negative300.bin', # 'Data/glove_short.txt',  #
        'wordDomain': True,  # True for words, False for characters
        'adaGradSGD': True,  # Stochastic gradient decent, True for adaGrad, False for regular SGD
        'clipGradients': True,  # True to avoid exploding gradients
        'weightInit': 'Load',  # 'He', 'Load' or 'Random'
        'eta': .02,  # Learning rate
        'gradientClipThreshold': 5,  # Threshold for clipping gradients
        'nHiddenNeurons': 300,  # Number of hidden neurons
        'seqLength': 25,  # Sequence length of each sequence iteration
        'lengthSynthesizedText': 50,  # Sequence length of each print of text evolution
        'lengthSynthesizedTextBest': 100,  # Sequence length of final best sequence, requires saveParameters
        'rmsProp': False,  # Implementation of rmsProp to adaGradSGD
        'gamma': 0.9,  # Weight factor of rmsProp
        'nEpochs': 40,  # Total number of epochs, each corresponds to (n book characters)/(seqLength) seq iterations
        'plotProcess': True,  # Plot learning curve
        'saveParameters': False,  # Save best weights with corresponding arrays iterations and smooth loss
        'neuronsOfInterest': [0, 20, 60, 100, 200, 299] # arange(5)   # Choose index of neuron to watch in the visualization
    }

    if not attributes['adaGradSGD']:
        attributes['eta'] = 0.01*attributes['eta']

    rnn = RecurrentNeuralNetwork(attributes)
    # rnn.testComputedGradients()
    rnn.adaGrad()
    print('Finished iterating through ', str(attributes['eta']), 'epochs.')


if __name__ == '__main__':
    random.seed(1)
    main()
    plt.show()
