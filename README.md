# Neural-Image-Caption-Generator

This project deals with generation of captions that explain the images using a Neural Model that is trained on Flickr8K dataset. An Encoder-Decoder model, in which the Decoder uses LSTM layers combined with Bahdanau Attention Mechanism, will be implemented for training (along with Teacher Forcing - feeding the caption in a sequential manner to the decoder). 

Teacher Forcing can be compared to spoon-feeding wherein we ask the decoder model to spit out the next word when fed with the previously generated/predicted word(s). The Encoder uses a CNN to create deep-level representations of images that are flattened, attended upon and then fed to the (LSTM based) decoder to create captions.

After the model gets trained, we will look at two inference algorithms viz. Greedy Search and Beam Search and discuss about each. For model's evaluation, we propose the BLEU metric to know how good the model can generate captions (albeit the metric having its own limitations)

This work is inspired from the first work in Attention Based Neural Image Captioning - https://arxiv.org/abs/1502.03044 (Show, Attend & Tell). Though variants and much more complex models exist, this is one of the first SOTA work in this direction and it is imperative for those starting in Deep Learning Technologies to cover these researches thoroughly and comprehensively. 

Encoder-Decoder models have been in existence for a long time. Encoder model's primary objective is to learn the sequences of the source input while the Decoder Model's objective is to interpret/transform it into some output. While the Encoder-Decoder models have worked good, Bahdanau et. al. in circa 2014 came up with an Attention Mechanism that helps the decoder to focus on some representations of the input while interpret/transform and spitting out the output. This led to remarkable improvements in the models' performance and now Attention is being implemented to several applications and even in other domains such as Computer Vision, etc.

In this notebook, we need to have tensorflow (2.4) installed in order to run it. Some knowledge on Deep NN Models is assumed as well as knowledge on RNN-LSTM networks.

The dataset has been taken from - https://www.kaggle.com/adityajn105/flickr8k

The notebook is divided into sections as outlined below -

Section 1: Data Processing
Section 2: Data Tokenization
Section 3: Defining the Model
Section 4: Training the Model
Section 5: Inference from the Model (Greedy & Beam Search)
Section 6: Evaluation of the Model
