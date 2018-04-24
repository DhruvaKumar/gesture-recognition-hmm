# Gesture recognition using Hidden Markov Model


The goal of the project is to perform gesture recognition from a smartphone IMU's gyroscope and accelerometer data. A [Hidden Markov Model (HMM)](https://en.wikipedia.org/wiki/Hidden_Markov_model) is used to build a generative model to classify new gestures.

This project was done as a part of ESE650 Learning in Robotics, University of Pennsylvania in the spring of 2015.

The training data consists of multiple observation sequences of 6 different gestures (circle, eight, inf, beat3, beat4, wave). The raw gyroscope and acclerometer data is first preprocessed and then quantized using k-means clustering. The 6 dimensional data is discretized into 20 clusters. A HMM is an extension of Markov models where observations are probabilistic functions of unobserved hidden states. Here, the discretized gestures are observations.

A HMM is parametrized by `lambda = {pi, A, B}`
```
N: number of hidden states
M: number of discrete observation possibilities
A [NxN]: transition probability distribution
B [NxM]: observation probability distribution
pi [Nx1]: initial state distribution
```

There are 3 basic problems for HMMs
1. Compute the probability of the observation sequence `O` given the model `lambda` `P(O|lambda)`
2. Given the observation sequence `O` and the model `lambda`, find an optimal state sequence that best explains `O`
3. Compute model parameters `lambda` that maximizes `P(O|lambda)`

In this project, I solve problem 3 for training the HMM model given the input discretized gestures. This is done using the Baum Welch algorithm which uses an iterative EM procedure, alternating between the E and M step, trying to maximize `P(O|lambda)` until a convergence criteria is met. 

Once I have 6 different models for 6 different gestures, I solve problem 1 to find the likelihood of a new observation for each model. The maximum is chosen as the classified gesture. 

### Results

The model achieved 100% accuracy on the validation set and 80% on the training set.

The code was vectorized that decreased the runtime by 80%. This aided in cross validation to tune the hyperparameters.

More information can be found in the [report](./report/project3.pdf)

