# Visualization of Recurrent Neural Networks

This developed system is the implementation of my Master Thesis to visualize Recurrent Neural Networks with text prediction as a testbed. It contributes with user trust, trust for developers of model's generalization and insights how to improve neural network architecture to optimize performance.

## Sample of terminal output
Predicted text highligted based on neural activity of the LSTM hidden state output. Neuron 597 is detected to be relevant for spaces.
![Sample of terminal output](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMSpaces-Terminal0.PNG)
## Corresponding heatmap with relevance scores
Heatmap of extracted feature of interest " " (space) (left) and automatic found relevant neuron 597 through relevance heatmap (right)
![Corresponding heatmap](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMSpaces-Heatmap0.png)
## Corresponding Fourier transform heat surface
Fourier transform and automatically detected peak for neuron 597 at frequency 0.5 (every second word) 
![Corresponding Fourier transform heat surface](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMSpaces-Fouriermap0.png)

## Another sample of terminal output
Predicted text highligted based on neural activity of the LSTM hidden state output. Neuron 455 is detected to be significantly active durong sentences and get deactivated by dots and question marks.
![Sample of terminal output](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMDots-Terminal0.PNG)
## Corresponding heatmap with relevance scores
Heatmap of extracted feature of interest ".", "?" and "!" (left) and automatic found relevant neuron 455 through relevance heatmap (right).
![Corresponding heatmap](https://github.com/johndah/Visualization-of-Recurrent-Neural-Networks/blob/master/LSTMDots-Heatmap0.png)
