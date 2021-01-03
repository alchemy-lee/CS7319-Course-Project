# CS7319 Course Project

### Topic A

Studies on bidirectional neural networks can be traced back to auto-association in the 1980s.
One typical example was Least Mean Square Error Reconstruction (Lmser) self-organizing
network that was rst proposed in 1991. Proceeded beyond AE, Lmser is featured
by a bidirectional architecture with several built-in natures, for which readers can refer to
Table I in Ref. One major nature of Lmser is Duality in Connection Weights (DCW).
DCW refers to using the symmetric weights in corresponding layers in encoder and decoder
such that the direct cascading by AE is further enhanced. The purpose of designing such
a symmetrical structure in Lmser is to approximate identity mapping per two consecutive
layers simply through $W^TW \approx I$, where $W^TW = I$ holds only if the weight matrix $W$ is
orthogonal. However, when $W$ is not orthogonal even not a square matrix, what should we
do to minimize the discrepancy between $x$ and $W^dW^ex$? One way is to constraint $W^d$ to be
the pseudo inverse of We. You are required to

1. Build a reconstruction network with the constraint that the weight matrix in the decoder
   to be the pseudo inverse of the encoder in the corresponding layers
2. Compare the reconstruction performance of the models with and without the constraint
   in 1) on the given data sets. 

In convolutional networks, implementing the constraint in 1) need you to rewrite the deconvolution
operation in some deep learning framework (like PyTorch, TensorFlow), thus
only the CNN version is encouraged but not required. We would also like to see some other
alternative constraints for CNN as long as they are designed to approximate the inverse
mapping.



### Dataset

For fully connected networks: MNIST, F-MNIST2
For CNN: STL10



### Structure Description

`Simple_AE.py` contains the training and testing code of auto-encoder.

`Convolution_AE.py` contains the training and testing code of convolution aotu-encoder.

`models` folder contains the code of corresponding AE models and the trained models in `*.pkl`.

`img` folder contains the images in training phase and testing phase of corresponding AE models.