# Visualization of Recurrent Neural Networks

This developed system is the implementation of my [Thesis](https://kth.diva-portal.org/smash/get/diva2:1394892/FULLTEXT01.pdf) to visualize Recurrent Neural Networks with text prediction as a testbed. It contributes with user trust, trust for developers of model's generalization and insights how to improve neural network architecture to optimize performance.

## Word prediction with LSTM

### Sample of terminal output
Predicted text highligted based on neural activity of the LSTM hidden state output, marked in figure below.
![LSTM Hidden States to be visualized](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/Hidden_States.png)

Neuron 295 and 376 are automatically hypothesized to be relevant for spaces independently through the heatmap and fourier heat surfance. They seem to have an action potential that goes from negative up to zero when triggered by the spaces.
![Sample of terminal output](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMSpaces-TerminalSeed1Hypotheses2.PNG)
### Corresponding heatmap with relevance scores
Heatmap of extracted feature of interest " " (space) (left) and automatic suggested relevant neurons 295 and 376 through relevance heatmap (right).

![Corresponding heatmap](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMSpaces-HeatmapSeed1Hypotheses2.png)
### Corresponding Fourier transform heat surface
Fourier transform of detected neuron 295 and 376 comfirmig the sensitivity to spaces by showing significant frequency components of 0.5 (every second word).

![Corresponding Fourier transform heat surface](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMSpaces-FouriermapSeed1Hypotheses2RelevanceDetected.png)

### Another sample of terminal output
Predicted text highligted based on neural activity of the LSTM output gate (marked below). 
![LSTM Output Gate to be visualized](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/Output_Gate.png)

Neuron 455 and 561 are automatically hypothesized to be significantly active (red) during sentences and get deactivated by dots and question marks.

![Sample of terminal output](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMDots-TerminalSeed0.PNG)
### Corresponding heatmap with relevance scores
Heatmap of extracted feature of interest ".", "?" and "!" (left) and automatically suggested relevant neuron 455 and 561 through relevance heatmap (right).

![Corresponding heatmap](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMDots-HeatmapSeed0.png)

## Character prediction with Vanilla RNN

### Sample of terminal output
Predicted text highligted based on neural activity of the LSTM hidden state output. Neuron 18 is detected to have many frequency components. The neuron seems to have a pattern to get deactivated (blue) when triggered by the second letter of synthesized words.
![Sample of terminal output](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/AllCharacters-Terminal.PNG)
### Corresponding heatmap with relevance scores
Heatmap extracting all features. Thus, the relevance is zero.
![Corresponding heatmap](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/AllCharacters-Heatmap.png)
### Corresponding Fourier transform heat surface
Fourier transform and automatically detected peaks for neuron 18, the neuron has multiple significant frequency components.  
![Corresponding Fourier transform heat surface](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/AllCharacters-Fouriermap.png)

# Setup
Clone or download this repository ```https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks.git``` for running the models of visualization.

## Dependencies
Download (e.g. with `conda install` or `pip install`) the required dependencies
```buildoutcfgf
keras==2.3.1
numpy==1.16.4
sty==1.0.0b12
matplotlib==3.1.3
nltk==3.4.5
gensim==3.8.1
lxml==4.5.0
terminaltables==3.1.0
```
and finally `tensorflow==1.13.1` or `tensorflow-gpu` for running either on CPU or GPU, respectively. Note that the former choice may be computationally demanding and thus very time consuming. Furthermore, even if `tensorflow-gpu` is installed, compatibility between dependency versions is crucial for utilizing the GPU to full extent (this can make the difference between a training epoch taking minutes or hours ). Thus, it is recommended to use `conda install` for version compatibility. 

The natural language toolkit (NLTK) vocabulary `averaged_perceptron_tagger` needs to be downloaded through
```sh
$ python
```
with the syntaxes
```python class:"lineNu"
import nltk
nltk.download('averaged_perceptron_tagger')
```

## Using pre-trained models
The RNN tokenization models can be trained from scratch. The recommended (and default) configuration, however, loads trained RNN and tokenization models based on the provided corpus. To do this, download the models from `https://sourceforge.net/projects/visualization-of-rnns-data/files/` and place in corresponding folders, i.e. `LSTM/Data` and `Vanilla RNN/Data` respectively `LSTM/LSTM Saved Models` and `Vanilla RNN/Vanilla RNN Saved Models` (they are too large to upload to this repository).

## Configurations
User defined configuration can be specified in
1. The `attributes` dictionary in the main function of respective LSTM and Vanilla RNN python file, for configurations of hyper-parameters, loading/saving options and light/dark theme in terminal.
2. `FeaturesOfInterest.txt`, for specifying text features, neurons and band-width of interests,
3. `PlotConfigurations.txt`, for enabling or disabling plots of training process and visualization of heatmaps and Fourier transform. 

# Running 
The main programs to run are the python files `LSTM/VisualizationOfLSTM.py` and 
`Vanilla RNN/VisualizationOfRNN.py` for visualization of word prediction respectively character prediction.
By default, the inference process (i.e. text generation without training), is defined as well as loading pre-trained models of RNN and for the word prediction case text vocabulary and tokenization.

Alternatively, the RNN models (as well as word embedding and tokenization models) can be trained from scratch, or even loaded and further trained.






