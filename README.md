In this project I have taken the implementation of a 2D convolutional layer in Keras (Conv2D class) and modified it to create a new version (Conv2D_HF). In this new version half of the kernels are a horizontal flipped version of the other half. This is motivated by the belief that for natural images, if there is one filter that detects one feature (e.g. a '<' shape) there should be another filter that detects the same shape but horizontally flipped ('>').

# EQUIVARIANCE

Equivariance is an active area of research in Machine Learning. A function F is equivariant to a transformation T if for an input x, the output of F(x) is a transformation of the output of the transformed input F(T(x)). The symmetric convolutional layer implemented in this project is equivariant to horizontal flipping. By running the symmetric_convolutional_layer.py script you can verify this.

# EXPERIMENTS

In order to test the usefulness of this modified convolutional layer I have taken a couple of models trained in cifar10 from the keras github website (https://github.com/keras-team/keras/tree/master/examples) and replaced the convolutional layers with this new implemention. The scripts that run these experiments are the following ones:
1) cifar10_resnet.py
2) cifar10_CNN.py
3) cifar10_resnet_modified.py
4) cifar10_CNN_modified.py

In the symmetric convolutional layer there is some weight sharing between the filters of the same layer. I have increased the number of filters in the symmetric version to have approximately the same number of trainable parameters.

For the resnet model (cifar10_resnet.py), the results are:

Test loss: 0.432

Test accuracy: 0.918

and with the new implementation (cifar10_resnet_modified.py):

Test loss: 0.456

Test accuracy: 0.908

so there is a small decrease in performance.

For the convolutional model (cifar10_CNN.py), the results are:

Test loss: 0.845

Test accuracy: 0.731

and with the new implementation (cifar10_CNN_modified.py):

Test loss: 0.733

Test accuracy: 0.765

which shows a considerable increase in performance.

Further experiments are needed to understand why it is helpful in one model but not the other. Maybe in the resnet model, the filters do not actually learn the same features but flipped. Also the optimization may be an issue since now the number of paths through the same weights has been duplicated. This modifies the variance of the gradiant updates and tuning the learning_rate/batch_size may be beneficial.

# RUNNING THE CODE

The python version used is:

Python 3.5.2

Run the following command to install the dependencies

pip install -r requirements.txt
