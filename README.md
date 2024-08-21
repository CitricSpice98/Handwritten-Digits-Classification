# Handwritten-Digits-Classification

File included in this exercise
• mnist all.mat: original dataset from MNIST. In this file, there are 10 matrices for testing set and 10 matrices for training set, which corresponding to 10 digits. You will have to split the training data into training and validation data.
• face all.pickle: sample of face images from the CelebA data set. In this file there is one data matrix and one corresponding labels vector. The preprocess routines in the script files will split the data into training and testing data.
• nnScript.py: Python script for this programming project. Contains function definitions - 1
Warning: In this project, you will have to handle many computing intensive tasks such as training a neural network. YOU MUST USE PYTHON 3 FOR IMPLEMENTATION. In addition, training such a big dataset will take a very long time, maybe many hours or even days to complete. Therefore, we suggest that you should start doing this project as soon as possible so that the computer will have time to do heavy computational jobs.
– preprocess(): performs some preprocess tasks, and output the preprocessed train, validation and test data with their corresponding labels. You need to make changes to this function.
– sigmoid(): compute sigmoid function. The input can be a scalar value, a vector or a matrix. You need to make changes to this function.
– nnObjFunction(): compute the error function of Neural Network. You need to make changes to this function.
– nnPredict(): predicts the label of data given the parameters of Neural Network. You need to make changes to this function.
– initializeWeights(): return the random weights for Neural Network given the number of unit in the input layer and output layer.
• facennScript.py: Python script for running your neural network implementation on the CelebA dataset. This function will call your implementations of the functions sigmoid(), nnObjFunc() and nnPredict() that you will have to copy from your nnScript.py. You need to make changes to this function.
• deepnnScript.py: Python script for calling the PyTorch library for running the deep neural network. You need to make changes to this function.
• cnnScript.py: Python script for calling the PyTorch library for a convolutional neural network. You need to change this function.
1.2 Datasets
Two datasets have been provided. Both consist of images.
1.2.1 MNIST Dataset
The MNIST dataset [1] consists of a training set of 60000 examples and test set of 10000 examples. All digits have been size-normalized and centered in a fixed image of 28 × 28 size. In original dataset, each pixel in the image is represented by an integer between 0 and 255, where 0 is black, 255 is white and anything between represents different shade of gray.
You will need to split the training set of 60000 examples into two sets. First set of 50000 randomly sampled examples will be used for training the neural network. The remainder 10000 examples will be used as a validation set to estimate the hyper-parameters of the network (regularization constant λ, number of hidden units).
1.2.2 CelebFaces Attributes Dataset (CelebA)
CelebFaces Attributes Dataset (CelebA) [3] is a large-scale face attributes dataset with more than 200K celebrity images. CelebA has large diversities, large quantities, and rich annotations, including:
• 10,177 number of identities,
• 202,599 number of face images, and
• 5 landmark locations, 40 binary attributes annotations per image.
For this programming assignment, we will have provided a subset of the images. The subset will consist of data for 26407 face images, split into two classes. One class will be images in which the individual is wearing glasses and the other class will be images in which the individual is not wearing glasses. Each image is a 54 × 44 matrix, flattened into a vector of length 2376.
2

 2
• • •
• • •
3 3.1
Your tasks
Figure 1: Neural network
Implement Neural Network (forward pass and back propagation) Incorporate regularization on the weights (λ)
Use validation set to tune hyper-parameters for Neural Network (number of units in the hidden layer and λ).
Run the deep neural network code we provided and compare the results with normal neural network. Run the convolutional neural network code and print out the results, for example the confusion matrix. Write a report to explain the experimental results.
Some practical tips in implementation Feature selection
In the dataset, one can observe that there are many features which values are exactly the same for all data points in the training set. With those features, the classification models cannot gain any more information about the difference (or variation) between data points. Therefore, we can ignore those features in the pre-processing step.
Later on in this course, you will learn more sophisticated models to reduce the dimension of dataset (but not for this assignment).
Note: You will need to save the indices of the features that you use and submit them as part of the submission.
3.2 Neural Network
3.2.1 Neural Network Representation
Neural network can be graphically represented as in Figure 1.
As observed in the Figure 1, there are totally 3 layers in the neural network:
• The first layer comprises of (d + 1) units, each represents a feature of image (there is one extra unit representing the bias).
3

• The second layer in neural network is called the hidden units. In this document, we denote m + 1 as the number of hidden units in hidden layer. There is an additional bias node at the hidden layer as well. Hidden units can be considered as the learned features extracted from the original data set. Since number of hidden units will represent the dimension of learned features in neural network, it’s our choice to choose an appropriate number of hidden units. Too many hidden units may lead to the slow training phase while too few hidden units may cause the the under-fitting problem.
• The third layer is also called the output layer. The value of lth unit in the output layer represents the probability of a certain hand-written image belongs to digit l. Since we have 10 possible digits, there are 10 units in the output layer. In this document, we denote k as the number of output units in output layer.
The parameters in Neural Network model are the weights associated with the hidden layer units and the output layers units. In our standard Neural Network with 3 layers (input, hidden, output), in order to represent the model parameters, we use 2 matrices:
• W(1) ∈ Rm×(d+1) is the weight matrix of connections from input layer to hidden layer. Each row in this matrix corresponds to the weight vector at each hidden layer unit.
• W (2) ∈ Rk×(m+1) is the weight matrix of connections from hidden layer to output layer. Each row in this matrix corresponds to the weight vector at each output layer unit.
We also further assume that there are n training samples when performing learning task of Neural Network. In the next section, we will explain how to perform learning in Neural Network.
3.2.2 Feedforward Propagation
In Feedforward Propagation, given parameters of Neural Network and a feature vector x, we want to compute the probability that this feature vector belongs to a particular digit.
Suppose that we have totally m hidden units. Let aj for 1 ≤ j ≤ m be the linear combination of input data and let zj be the output from the hidden unit j after applying an activation function (in this exercise, we use sigmoid as an activation function). For each hidden unit j (j = 1, 2, · · · , m), we can compute its value as follow:
d+1
aj =Xw(1)xp (1)
jp p=1
zj =σ(aj)= 1 (2) 1 + exp(−aj )
where w(1) = W(1)[j][p] is the weight of connection from the pth input feature to unit j in hidden layer. Note ji
that we do not compute the output for the bias hidden node (m + 1); zm+1 is directly set to 1.
The third layer in neural network is called the output layer where the learned features in hidden units are linearly combined and a sigmoid function is applied to produce the output. Since in this assignment, we want to classify a hand-written digit image to its corresponding class, we can use the one-vs-all binary classification in which each output unit l (l = 1, 2, · · · , 10) in neural network represents the probability of an image belongs to a particular digit. For this reason, the total number of output unit is k = 10. Concretely,
for each output unit l (l = 1, 2, · · · , 10), we can compute its value as follow: m+1
bl = X w(2)zj (3) lj
j=1
ol = σ(bl) = 1 (4) 1 + exp(−bl)
Now we have finished the Feedforward pass.
  4

3.2.3 Error function and Backpropagation
The error function in this case is the negative log-likelihood error function which can be written as follow:
1nk
XX(yillnoil +(1−yil)ln(1−oil)) (5) i=1 l=1
where yil indicates the lth target value in 1-of-K coding scheme of input data i and oil is the output at lth output node for the ith data example (See (4)).
Because of the form of error function in equation (5), we can separate its error function in terms of error
for each input data xi:
where
1 Xn
Ji(W(1),W(2)) (6) Ji(W(1),W(2))=−X(yillnoil +(1−yil)ln(1−oil)) (7)
l=1
where
δl = ∂Ji ∂ol = −(yl − 1 − yl )(1 − ol)ol = ol − yl ∂ol ∂bl ol 1−ol
J(W(1),W(2))=−
n
J(W(1),W(2)) = n k
i=1
One way to learn the model parameters in neural networks is to initialize the weights to some random numbers and compute the output value (feed-forward), then compute the error in prediction, transmits this error backward and update the weights accordingly (error backpropagation).
The feed-forward step can be computed directly using formula (1), (2), (3) and (4).
On the other hand, the error backpropagation step requires computing the derivative of error function with respect to the weight.
Consider the derivative of error function with respect to the weight from the hidden unit j to output unit l where j = 1,2,··· ,m+1 and l = 1,··· ,10:
∂Ji ∂w(2)
lj
∂Ji ∂ol ∂bl (8) ∂ol ∂bl ∂w(2)
=
= δlzj
lj
  (9)
 Note that we are dropping the subscript i for simplicity. The error function (log loss) that we are using in (5) is different from the the squared loss error function that we have discussed in class. Note that the choice of the error function has “simplified” the expressions for the error!
On the other hand, the derivative of error function with respect to the weight from the pth input feature to hidden unit j where p = 1, 2, · · · , d + 1 and j = 1, · · · , m can be computed as follow:
∂Ji ∂w(1)
jp
= Xk ∂Ji ∂ol ∂bl ∂zj ∂aj (10)
  l=1 k
∂ol ∂bl ∂zj ∂aj ∂w(1) jp
= Xδlw(2)(1−zj)zjxp lj
(11) = (1−zj)zj(Xδlw(2))xp (12)
l=1
Note that we do not compute the gradient for the weights at the bias hidden node.
After finish computing the derivative of error function with respect to weight of each connection in neural
network, we now can write the formula for the gradient of error function:
1 Xn
∇J(W(1),W(2)) = n 5
i=1
k
lj
l=1
∇Ji(W(1),W(2)) (13)

We again can use the gradient descent to update each weight (denoted in general as w) with the following rule:
wnew = wold − γ∇J(wold) (14) 3.2.4 Regularization in Neural Network
In order to avoid overfitting problem (the learning model is best fit with the training data but give poor generalization when test with validation data), we can add a regularization term into our error function to control the magnitude of parameters in Neural Network. Therefore, our objective function can be rewritten as follow:
m d+1 k m+1  J(W(1),W(2))=J(W(1),W(2))+ λ XX(w(1))2 +XX(w(2))2
e 2njp lj j=1 p=1 l=1 j=1
(15)
where λ is the regularization coefficient.
With this new objective function, the partial derivative of new objective function with respect to weight
from hidden layer to output layer can be calculated as follow: ∂J1n∂J !
e= X i+λw(2) (16)
  ∂w(2) lj i=1 lj
Similarly, the partial derivative of new objective function with respect to weight from input layer to hidden layer can be calculated as follow:
∂J1n∂J !
e= X i+λw(1) (17)
∂w(1) n jp
With this new formulas for computing objective function (15) and its partial derivative with respect to weights (16) (17) , we can again use gradient descent to find the minimum of objective function.
3.2.5 Python implementation of Neural Network
In the supporting files, we have provided the base code for you to complete. In particular, you have to complete the following functions in Python:
• sigmoid: compute sigmoid function. The input can be a scalar value, a vector or a matrix.
• nnObjFunction: compute the objective function of Neural Network with regularization and the gradient
of objective function.
• nnPredict: predicts the label of data given the parameters of Neural Network.
Details of how to implement the required functions is explained in Python code.
∂w(2) n lj
  ∂w(1) jp i=1 jp
   Optimization: In general, the learning phase of Neural Network consists of 2 tasks. First task is to compute the value and gradient of error function given Neural Network parameters. Second task is to optimize the error function given the value and gradient of that error function. As explained earlier, we can use gradient descent to perform the optimization problem. In this assignment, you have to use the Python scipy function: scipy.optimize.minimize (using the option method=’CG’ for conjugate gradient descent), which performs the conjugate gradient descent algorithm to perform optimization task. In principle, conjugate gradient descent is similar to gradient descent but it chooses a more sophisticated learning rate γ in each iteration so that it will converge faster than gradient descent. Details of how to use minimize are provided here: http://docs.scipy.org/doc/scipy-0. 14.0/reference/generated/scipy.optimize.minimize.html.
 6

We use regularization in Neural Network to avoid overfitting problem (more about this will be discussed in class). You are expected to change different value of λ to see its effect in prediction accuracy in validation set. Your report should include diagrams to explain the relation between λ and performance of Neural Network. Moreover, by plotting the value of λ with respect to the accuracy of Neural Network, you should explain in your report how to choose an appropriate hyper-parameter λ to avoid both underfitting and overfitting problem. You can vary λ from 0 (no regularization) to 60 in increments of 5 or 10.
You are also expected to try different number hidden units to see its effect to the performance of Neural Network. Since training Neural Network is very slow, especially when the number hidden units in Neural Network is large. You should try with small hidden units and gradually increase the size and see how it effects the training time. Your report should include some diagrams to explain relation between number of hidden units and training time. Recommended values: 4, 8, 12, 16, 20.
4 PyTorch Library
In this assignment you will only implement a single layer Neural Network. You will realize that implementing multiple layers can be a very cumbersome coding task. However, additional layers can provide a better modeling of the data set. The analysis of the challenging CelebA data set will show how adding more layers can improve the performance of the Neural Network. To experiment with Neural Networks with multiple layers, we will use PyTorch library (https://pytorch.org/). Please install PyTorch on personal machines. The code provided has been tested on PyTorch Version 1.12.1.
Your experiments should include the following:
• Evaluate the accuracy of single hidden layer Neural Network on CelebA data set (test data only), to distinguish between two classes - wearing glasses and not wearing glasses. Use facennScript.py to obtain these results.
• Evaluate the accuracy of deep Neural Network (try 3, 5, and 7 hidden layers) on CelebA data set (test data only). Use deepnnScript.py to obtain these results.
• Compare the performance of single vs. deep Neural Networks in terms of accuracy on test data and training time.
• Compare the performance of the deep Neural Networks vs. Convolutional Neural Networks in terms of accuracy on test data and training time. Use cnnScript.py to obtain these results (extra points)

References
[1] LeCun, Yann; Corinna Cortes, Christopher J.C. Burges. “MNIST handwritten digit database”.
[2] Bishop, Christopher M. “Pattern recognition and machine learning (information science and statistics)” (2007).
[3] Liu, Ziwei; Luo, Ping; Wang, Xiaogang; Tang, Xiaoou. “Deep Learning Face Attributes in the Wild”, Proceedings of International Conference on Computer Vision (ICCV) (2015).
