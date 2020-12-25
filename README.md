In this repository I have taken the implementation of a 2D convolutional layer in Keras (Conv2D class) and modified it to create two new versions:
1) Conv2D_HF: CNN layer that is equivariant to the horizontal flip transformation in the input space.
2) Conv2D_HF: CNN layer that is equivariant to transformation 'g' in the input space. 'g' is the transformation in the output space of function Conv2D_HF when applying horizontal flip at its input. Therefore, stacking first a Conv2D_HF layer and then a Conv2D_HF2 layers makes the two layers together equivariant to horizontal flip in input space.
If more Conv2D_HF2 layers are stacked on top of the two, the whole network is still equivariant to horizontal flip in input space.

# EQUIVARIANCE

Equivariance is an active area of research in Machine Learning. A function f is equivariant to transformation g_0 if there exist g_1 such that f(g_0(x)) = g_1(f(x)).

Equivariance is a desired property to have in neural networks. It is a good inductive bias to put in the network. Some works show that neural nets learn functions that are equivariant. This is motivated by the fact that similar inputs should be processed in a similar manner. Enforcing this property makes the network more efficient and less prune to overfitting due to weight sharing.

By running the scripts keras/Horflip_equivariant_CNN_firstlayer.py and keras/Horflip_equivariant_CNN_secondlayer.py you can validate these layer are equivariant to the previously described transformations.

# EXPERIMENTS

In order to test the usefullness of this property I have taken a simple CNN model achieving 70% in Cifar 10 and replace some of the convolutional layer with these new CNN layers. Due to weight sharing, for a similar size of the feature vectors in each layer, the number of params for these new CNN layers is smaller.

This experiment has been implemented (and the results shown) in keras/equivariance_experiments_keras.ipynb.

The results show that substituing the first layer for a equivariant one does help by boosting the accuracy of the model around 1%. On the other hand substituing the remaining CNN layers by equivariant ones decreases the accuracy of the model.

# RUNNING THE CODE

The python version used is:

Python 3.7.4

Run the following command to install the dependencies

pip install -r requirements.txt