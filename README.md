# Neural-Image-Caption-Generator

This project deals with generation of captions that explain the images using a Neural Model that is trained on Flickr8K dataset. An Encoder-Decoder model, in which the Decoder uses LSTM layers combined with Bahdanau Attention Mechanism, will be implemented for training (along with Teacher Forcing - feeding the caption in a sequential manner to the decoder). 

Teacher Forcing can be compared to spoon-feeding wherein we ask the decoder model to spit out the next word when fed with the previously generated/predicted word(s). The Encoder uses a CNN to create deep-level representations of images that are flattened, attended upon and then fed to the (LSTM based) decoder to create captions.

After the model gets trained, we will look at two inference algorithms viz. Greedy Search and Beam Search and discuss about each. For model's evaluation, we propose the BLEU metric to know how good the model can generate captions (albeit the metric having its own limitations)

This work is inspired from the first work in Attention Based Neural Image Captioning - https://arxiv.org/abs/1502.03044 (Show, Attend & Tell). Though variants and much more complex models exist, this is one of the first SOTA work in this direction and it is imperative for those starting in Deep Learning Technologies to cover these researches thoroughly and comprehensively. 

Encoder-Decoder models have been in existence for a long time. Encoder model's primary objective is to learn the sequences of the source input while the Decoder Model's objective is to interpret/transform it into some output. While the Encoder-Decoder models have worked good, Bahdanau et. al. in circa 2014 came up with an Attention Mechanism that helps the decoder to focus on some representations of the input while interpret/transform and spitting out the output. This led to remarkable improvements in the models' performance and now Attention is being implemented to several applications and even in other domains such as Computer Vision, etc.

In this notebook, we need to have tensorflow (2.4) installed in order to run it. Some knowledge on Deep NN Models is assumed as well as knowledge on RNN-LSTM networks.

The dataset has been taken from - https://www.kaggle.com/adityajn105/flickr8k

The notebook is divided into sections as outlined below -

Section 1: Data Processing <br>
Section 2: Data Tokenization <br>
Section 3: Defining the Model <br>
Section 4: Training the Model <br>
Section 5: Inference from the Model (Greedy & Beam Search) <br>
Section 6: Evaluation of the Model

## Section 1: Data Processing

After downloading the dataset, the images are places in 'Images' directory while 'captions.txt' is placed in the root directory.

Since, this is notebook is meant to be descriptive, the number of captions are limited to 30,000 (out of 40,455)

The first task in processing is to resize all images to a fixed size (chosen size is 250,250). The text in captions is cleaned and processed to include a 'start' and 'end' signal to indicate the beginning and ending of sentence respectively. 

(I had thought of padding the image when resizing it to maintatin aspect ratio but since the model will have to attend on specific portions of the image, this was completely wasteful for the model)

The parsed images are stored in a dictionary wherein the key is the image name and value is the parsed image. A dataframe of captions is also generated that has the image name in one column, the original caption in the next column and the processed caption in the last column.

| image | caption | caption_processed |
| ------------- | ------------- | ------------- |
| 3508882611_3947c0dbf5.jpg | A large grey dog is jumping over a white hurdle .	 | 	<start> a large grey dog is jumping over a whi... |
| 1000268201_693b08cb0e.jpg | A little girl in a pink dress going into a woo... | <start> a little girl in a pink dress going in... |

and so on...

## Section 2: Data Tokenization

The captions need to be tokenized in order to be fed to the Decoder inside the model. All empty fields are padded with 0s. The tensorflow's tokenizer and padding functionalities will tokenize the processed text.

This means that for two texts in the captions corpus -

S1:- I am here
S2:- He is here right now

The tokenized form would be -

S1:- [1, 2, 3, 0, 0]
S2:- [4, 5, 3, 6, 7]

Basically, the tokenized form would replace the word with a unique number that would represent that word.

* Notice the padding done at the end of sentence 1 (two 0's added). This will be done based on the maximum length of a sentence in the corpus
* Notice the repetiton of 3 in both tokenized forms which represent the word 'here' being repeated

