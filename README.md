# ML-algorithms
<b>Implementation of several supervised and non-supervised machine learning algorithms using both probabilistic and non-probabilistic approaches</b>

This repository provides python implementations for the following machine learning algorithms:

1. <b>Supervised Learning</b>
      - <b>Regression</b>: L2 regularized least squares linear regression (ridge regression), gradient descent and active learning procedure for linear regression
      - <b>Classification</b>: Perceptron and K-class Bayes classifier
2. <b>Unsupervised Learning</b>
      - <b>Clustering</b>: K-means and EM for Gaussian mixture models
      - <b>Probabilistic Matrix Factorization</b>

## Supervised Learning

Seeks to learn a function that, given an input data x, predicts the corresponding value. These algorithms are based on a function mapping from input to output.

### Regression

Using a set of inputs, will predict a real-valued output.

#### Ridge Regression (non-probabilistic)

L2 regularized least squares linear regression. Contrains the model parameters by adding two terms:

- Lambda > 0: regularization parameter
- g(w) > 0: penalty function encouraging the desired properties

In this case, g(w) = ||w||<sup>2</sup>, addressing high variance issues.

#### Gradient Descent (non-probabilistic)

Iterative optimization method which aims to get the best parameters of the linear model by simultaneously updating all weights of a model. It uses a learning rate alpha which is set at the beggining and will determine the size of the steps at each iteration. It converges to a local optima.

#### Active Learning (probabilistic)

Probabilistic learning strategy which aims to model the predictive distribution for unknown (test) data given known (train) data and the posterior distribution. Successively predicts unmeasured data starting by those points where a higher variance is estimated. During this iterative process it continuously updates the posterior with the new observations learnt. 

### Classification

Using a set of inputs, will predict a discrete label.

#### Perceptron (non-probabilistic)

Most simple neural network and classification method used to split perfectly separable data. It is a linear classification method which assigns the data to one of two classes, performing a linear combination of features and using the sign as activation function. It learns by adjusting the weights (add or substract x) if the hyperplane mistakenly classifies current x evaluated. It converges only with perfectly separable data and to the first hyperplane that correctly splits all training data points.

#### K-class Bayes classifier (probabilistic)

Plug-in classifier (discriminative model) which aims to predict for a given input x, the most probable label conditioned on x. It uses the Bayes rule to make this predictions based on the class prior (distribution on y) and the class conditional disctibution of X. Both this distributions are approximated from available data. It can produce either a decision boundary or region.

## Unsupervised Learning

Seeks to learn the underlying structure of an input data x.

### Clustering

#### K-Means (non-probabilistic)

Simple hard-clustering algorithm. Given input data x, it aims to predict vector c of cluster assignments and K mean vectors mu. 

To perform this assignments it minimizes the Euclidean distance from each data point to the centroid of its assigned cluster.

#### EM for Gaussian Mixture Models (probabilistic)

Soft-clustering algorithm. Given input data x, it breaks data across the K clusters intelligently, accounting for borderline cases.

To estimate this cluster probability distribution, the Expectation-Minimization (EM) algorithm is used in combination with K Gaussian generative distributions.

### Probabilistic Matrix Factorization

Assuming a low rank rating data matrix is provided, this alorithm aims to factorize it into the product of two (also) low rank matrices U and V; reducing computational complexity. 

A distribution on the data is assumed considering user and object locations to be generated from the same (generative model) distribution. To estimate matrices U and V, the log joint likelihood is maximized.

## Running the algorithms

All implementations are modular and can be applied to any dataset provided through command line. When Xy are given as an input, y response variable is expected to be in the last column. To run the different modules use the following commands:

<b>Regression</b>

<code> $ python3 Regression.py lambda sigma2 X_train.csv y_train.csv X_test.csv </code>

<b>Gradient Descent</b>

<code> $ python3 GradientDescent.py Xy_train.csv output_filename.csv </code>

<b>Perceptron</b>

<code> $ python3 Perceptron.py Xy_train.csv output_filename.csv </code>

<b>K-class Bayes classifier</b>

<code> $ python3 Classification.py X_train.csv y_train.csv X_test.csv </code>

<b>Clustering</b>

<code> $ python3 Clustering.py X.csv </code>

<b>Probabilistic Matrix Factorization</b>

<code> $ python3 ProbabilisticMatrixFactorization.py Ratings.csv </code>
