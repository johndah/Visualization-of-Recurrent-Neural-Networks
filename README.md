# Visualization of Recurrent Neural Networks

This developed system is the implementation of my Master Thesis to visualize Recurrent Neural Networks with text prediction as a testbed. It contributes with user trust, trust for developers of model's generalization and insights how to improve neural network architecture to optimize performance.

## Word prediction with LSTM

### Sample of terminal output
Predicted text highligted based on neural activity of the LSTM hidden state output, marked in figure below.
![LSTM Hidden States to be visualized](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/Hidden_States.png)

Neuron 376 and 295 are automatically hypothesized to be relevant for spaces independently through the heatmap and fourier heat surfance. It seem to have an action potential that goes from negative up to zero when triggered by the spaces.
![Sample of terminal output](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMSpaces-TerminalSeed1Hypotheses2.PNG)
### Corresponding heatmap with relevance scores
Heatmap of extracted feature of interest " " (space) (left) and automatic suggested relevant neurons 376 and 295 through relevance heatmap (right).
![Corresponding heatmap](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMSpaces-HeatmapSeed1Hypotheses2.png)
### Corresponding Fourier transform heat surface
Fourier transform of detected neuron 376 and 295 comfirmig the sensitivity to spaces by showing significant frequency components of 0.5 (every second word).
![Corresponding Fourier transform heat surface](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMSpaces-FouriermapSeed1Hypotheses2RelevanceDetected.png)

### Another sample of terminal output
Predicted text highligted based on neural activity of the LSTM output gate (marked below). 
![LSTM Output Gate to be visualized](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/Output_Gate.png)

Neuron 455 and 561 are automatically hypothesized to be significantly active (red) during sentences and get deactivated by dots and question marks.
![Sample of terminal output](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMDots-TerminalSeed0Hypotheses2.PNG)
### Corresponding heatmap with relevance scores
Heatmap of extracted feature of interest ".", "?" and "!" (left) and automatically suggested relevant neuron 455 and 561 through relevance heatmap (right).
![Corresponding heatmap](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMDots-TerminalSeed0Hypotheses2.png)

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
