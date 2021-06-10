# Aspect Level Sentiment Classification
This is a pytorch implementation for [Interactive Attention Networks for Aspect-Level Sentiment Classification](https://arxiv.org/pdf/1709.00893.pdf) (Dehong Ma et al, IJCAI 2017).

Here we have used the Aspect Based Sentiment Analysis(ABSA) dataset. There are two domain specific dataset for laptops and restaurants consisting of over 6K sentences with fine grained aspect-level human annotations have been provided for training.
One can read more about the dataset [here](https://alt.qcri.org/semeval2014/task4/)

The code files contain the pytorch implementation of IAN for both the dataset and the polarity of the specific term in a particular review is also demostrated using the confusion matrix to understand the IAN architechture better.

The models folder contains the models saved after training and can be used for making predictions.
