# CS6910-Shashank MK CH23S019

## Assignment 3 - Foundations of Deep Learning

This repository contains implementation of Seq2Seq Transliteration task using PyTorch as part of assignment in Foundations of Deep Learning coursework. 


In this assignment,  Aksharantar dataset was used which was released by AI4Bharat. This dataset contains pairs of the following form:

xxx,yyy

ajanabee,अजनबी

i.e., a word in the native script and its corresponding transliteration in the Latin script (how we type while chatting with our friends on WhatsApp etc). Given many such (xi,yi)i=1n(x_i, y_i)_{i=1}^n(xi​,yi​)i=1n​ pairs our goal is to train a model y=f^(x)y = \hat{f}(x)y=f^​(x) which takes as input a romanized string (ghar) and produces the corresponding word in Devanagari (घर). 

Wandb data visualisation was used to log one sample from each class at the start of the training, and the metrics at the end of each epoch.

## Libraries used:

    numpy
    python
    pytorch
    Matplotlib
    Pandas
    Wandb

## Running Instructions:




## Code Flexibility:

  The code is flexible to add :
  
       * encoder_embedding_size
       * ecoder_embedding_size
       * encoder_hidden_size
       * decoder_hidden_size
       * cell_type
       * Attention
       * variable number of layers in encoder and decoder

    
    
## Link to Report:
 
 https://wandb.ai/ch23s019/Assignment_3_CS6910/reports/CS6910-Assignment-3--Vmlldzo3Njg5MjQw

## Acknowledgements:

* Lecture and slides by Prof. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule were referred mostly for creating the feed forward neural network.

* Tutorials by Aladdin Persson.
* https://www.deeplearningbook.org/
* https://github.com/bentrevett/pytorch-seq2seq/blob/main/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

  


