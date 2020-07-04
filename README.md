# Multicalss-TB
Integrating unsupervised and supervised learning for automated diagnosis of tuberculosis.

This was part of my master's research project at Indian Institute of Technology(BHU), varanasi, in which I aimed to compare the performance of Multi-class classifier over binary classifier on MODS dataset of TB disease on a seven layer deep CNN.The binary data of positive and negative images was clustered into optimum number of classes using K-means clustering algorithm.Google colab's GPU was used to train the models for comparison and manual hypertuning of the multi-class model.

The dataset can be downloaded from https://github.com/santiagolopezg/MODS_ConvNet.

The dataset consists of 12510 grayscale images of resolution 224*224 of MODDS culture of Tuberculosis. Out of these, 4849 images were positive and 7661 were negative. The base model was a hypertuned VGG16 convolutional neural network architecture on binary classification with an accuracy of 92 +- 0.35%. Thus, there is a tendency of failure of detection on 8% of cases which puts a constraint on it's practical application.  

The propsed model uses an integrated approch of unsupervised learning by clustering the positive and negative images into an optimal number of clusters by extracting the features out of images using weights of a pretrained VGG16 network on CIFAR10 dataset. Each cluster thus reprsenting a new class. The optimal number of clusters were evaluated using elbow diagram and K-means clustering algorithm.

To run the code, the following python files need to be run in order:
1. data_seperation.py
2. clustering.py
3. data_balancing.py
4. models.ipynb
5. hyper_tun.ipynb

Due to limitation of GPU run time on colab, the hypertuning was done manually for the multi class model as in 'hyper_par.xlsx' file but the same can be done effectively using 'hyperas' library as coded in 'hyper_tun.ipynb'

To check the performance of multi-class classifier over binary classifier, performance parameters of accuracy, precision and recall were evaluated on a seven layer deep network without hypertuning but with model structure similar to VGG16 architecture for both binary and multi-class cases.The network was trained on google colab's GPU.

The performance parameters after hypertuning were as under:
For binary classifier:
Accuracy  : 64.3 %
Precision : 47.33 %
Recall    : 87.42 %

For Multi-class classifier:
Accuracy  : 68.8 %
Precision : 66.63 %
Recall    : 69.76 %

Performance of Hyper-tuned seven layer deep CNN:
Accuracy  : 67.3 +- 2.3 %
Precision : 66.0 +- 1.87 % 
Recall    : 68.5 +- 2.2 %

Scope of future work:
Hypertuning multi-class classifier on a VGG16 network to test for performance over previously huper-tuned binary classifier.
