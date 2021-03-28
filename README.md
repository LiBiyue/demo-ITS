
# demo-for-ITS-paper
This demo is used for illustrating the computing procedure of DUACE model submitted for IEEE ITS.

## the demonstration of algorithm is :
![fomula1](https://github.com/LiBiyue/demo-ITS/blob/main/image/algorithm%20demonstration.png)

## the illustration of result is :
![fomula1](https://github.com/LiBiyue/demo-ITS/blob/main/image/result%20illustration.png)
The original data distribute loosely in high-dimensional space. There exits a higher probability of misclassification if clustering is conducted on high-dimensional data. While the latent representations, which are extracted from original data through the trained Autoencoder, have a dense distribution. Therefore, clustering on latent representations has a promising accuracy.

## the input data is : 
[[0.78637992 0.82812079]
 [0.11459974 0.61251262]
 [0.73809806 0.2229145 ]
 [0.57681002 0.29892906]]

## the data after SMOTE is : 
[[0.78637992 0.82812079]
 [0.73809806 0.2229145 ]
 [0.11459974 0.61251262]
 [0.57681002 0.29892906]
 [0.77189536 0.6465589 ]]
 
## weight of encoder is : 
[[0.9 ]
 [0.15]]
 
## bias of encoder is :
[[0.19]
 [0.08]
 [0.62]
 [0.86]
 [0.2 ]]
 
## weight of dencoder is :  
[[0.33 0.77]]

## bias of dencoder is : 
[[0.64 0.17]
 [0.73 0.19]
 [0.82 0.13]
 [0.28 0.54]
 [0.09 0.45]]
 
## the output of AE is : 
[[0.98329987 0.9618459 ]
 [0.99493883 0.79178267]
 [1.08947867 0.7594004 ]
 [0.75555884 1.63475339]
 [0.4162517  1.21403789]]
 
## P matrix is : 
[[1.   0.   0.11 0.89]
 [0.19 1.   0.14 0.68]
 [0.1  0.03 1.   0.87]
 [0.01 0.01 0.98 1.  ]]
 
## Q matrix is :  
[[0.28 0.28 0.27 0.22 0.23]
 [0.29 0.29 0.29 0.2  0.23]
 [0.29 0.3  0.3  0.19 0.22]
 [0.26 0.23 0.22 0.34 0.29]
 [0.26 0.24 0.23 0.27 0.31]]
 
## U matrix is : 
[[ 0.25  0.13 -0.34]
 [-0.4   0.16  0.27]
 [ 0.15 -0.28  0.07]]
 
## V matrix is :  
[[ 0.06 -0.08 -0.08]
 [-0.04  0.05  0.04]
 [-0.03  0.03  0.04]]

## centroids are : 
{0: array([1.14859294]), 1: array([0.79839951])}

## latent representations are : 
[[1.02571249]
 [0.78272138]
 [0.81407764]
 [1.42918604]
 [0.99088027]]
 
## W matrix is : 
[[0.12 0.23]
 [0.37 0.02]
 [0.33 0.02]
 [0.28 0.63]
 [0.16 0.19]]
 
## G matrix is : 
[[0.51 0.46 0.46 0.54 0.5 ]
 [0.49 0.54 0.54 0.46 0.5 ]]
 
