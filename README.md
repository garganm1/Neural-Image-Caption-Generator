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
| 3508882611_3947c0dbf5.jpg | A large grey dog is jumping over a white hurdle .	 | 	\<start\> a large grey dog is jumping over a whi... |
| 1000268201_693b08cb0e.jpg | A little girl in a pink dress going into a woo... | \<start\> a little girl in a pink dress going in... |

and so on...

## Section 2: Data Tokenization

The captions need to be tokenized in order to be fed to the Decoder inside the model. All empty fields are padded with 0s. The tensorflow's tokenizer and padding functionalities will tokenize the processed text.

This means that for two texts in the captions corpus -

S1:- I am here <br>
S2:- He is here right now

The tokenized form would be -

S1:- [1, 2, 3, 0, 0] <br>
S2:- [4, 5, 3, 6, 7]

Basically, the tokenized form would replace the word with a unique number that would represent that word.

* Notice the padding done at the end of sentence 1 (two 0's added). This will be done based on the maximum length of a sentence in the corpus
* Notice the repetiton of 3 in both tokenized forms which represent the word 'here' being repeated

All processed data is then stored as pickle files.

## Section 3: Defining the Model

**Let us build up the concept slowly. Starting discussion with an Encoder-Decoder model without any attention mechanism**

![fig2](https://user-images.githubusercontent.com/55252306/117485439-5f7c2b80-af36-11eb-8fb0-0e11ee12596d.PNG)

The blue cells denote the encoder and the red cells denote the decoder layers.

After the CNN model extracts features from the image, these are fed to the decoder (red hidden) layer that learns the sequences of caption along with the source features from the encoder.

The decoder layer then makes a projection layer which spits out a prediction vector of size V (vocabulary size of target corpus). The maximum probability value of this vector denotes the word that the model is predicting which is judged against what should be produced as a loss function.

Notice the \<start> at the start of target input words which is the first word fed to the decoder model (representing the start of decoding) and the prediction at this point is the first word of caption. The last word of target input spits out \</end> that would denote the end of prediction.

We have employed only one hidden layer in this notebook in the decoder.

**Teacher Forcing:**

We will implement teacher forcing during training. This means that the model is fed with the next word in the caption as an input to the decoder and that too in a sequential manner. In summary, it is the technique where the target word is passed as the next input to the decoder. Note that this won't and can not be implemented during inference

The inference, i.e. generating image caption once the model has been trained, would be a little different. Let's see how below-

Everything is the same except we don't know the target input to be fed to the model when you would be inferring (teacher forcing above).

In this case, the first prediction of \<start> (Giraffes) is fed as an input of next target word to the model the produce the next word in the caption. The sequence continues until we hit \</end> where the captioning stops.

Above figure is a type of greedy decoding since we are only looking at the word with the highest probability in the prediction vector. This is very basic seq2seq model. Adding the attention mechanism to it greatly enhances its performance. If you have understood the above architecture, move below to understand **Attention**

**Let's now start with an Encoder-Decoder model with Bahdanau attention mechanism**

![fig3](https://user-images.githubusercontent.com/55252306/117485738-c26dc280-af36-11eb-84c3-c772ec13a1e9.PNG)

You will notice the addition of 'attention' to the above discussed model. Also, the picture has VGG-16 as the feature extractor but we are using Inception-V3 which is a pre-trained model

The calculation of the attention process hinges on the below formulae (note that soft attention is implemented) :-

![fig7](https://user-images.githubusercontent.com/55252306/117485771-cd285780-af36-11eb-83c6-17944bccf8eb.PNG)

**Above Figure:- Formula 4 and 1 respectively (for deriving attention weights (α<sub>ti</sub>) )**

![fig4](https://user-images.githubusercontent.com/55252306/117485829-dd403700-af36-11eb-8b8d-584b16d3a1ed.PNG)

**Above Figure:- Formula 2 (for computing context vector (ẑ<sub>t</sub>) )**

![fig5](https://user-images.githubusercontent.com/55252306/117485864-e6c99f00-af36-11eb-9fda-9e1d203352f7.PNG)

**Above Figure:- Formula 3 (for computing the attention vector (α<sub>t</sub>) )**

![fig6](https://user-images.githubusercontent.com/55252306/117485894-ee894380-af36-11eb-9bac-d02f67f9a4a7.PNG)

**Above Figure:- Formula for computing attention score (Bahdanau Attention)**

**Steps of Computation:-**

The attention computation happens at every decoder time step. It consists of the following stages:

1. The extracted features (a<sub>i</sub>) from the encoder are compared with each target hidden state (h<sub>t-1</sub>) to derive attention weights (α<sub>ti</sub>).
2. Based on the attention weights (α<sub>ti</sub>), we compute a context vector (ẑ<sub>t</sub>)) as the weighted average of the source image features.
3. Then we combine the context vector (ẑ<sub>t</sub>) with the current target hidden state (h<sub>t-1</sub>) to yield the final attention vector (α<sub>t</sub>)
4. The attention vector (α<sub>t</sub>) is then fed as an input to the next time step (input feeding)

The way of comparing the input features with the current target hidden state has been researched and Bahdanau' additive style has been employed in this notebook (**Formula 4** and last formula in above). There are other comparative measures such as Luong's multiplicative style as well as their variations and combinations. The paper 'Show, Attend and Tell' also talks about a Hard Attention where the decoder focuses on a sinlge point as compared to an area of an image in soft attention.

The comparison gives out a score when all input features are compared with the current decoder hidden state. This score is fed to a softmax layer (**Formula 1**) that measures the score of the current hidden state against the input features (which are the attention weights).

The weights are then assessed with the input features so that the model focuses on where it should focus in identifying objects/actions in the input image (**Formula 2**). This produces a context vector that contains information where the context lies in the input features to generate the current word in the caption

The context vector is concatenated with the current target hidden state and then activated to get an attention vector (**Formula 3**) that encompasses all information of source input and for target input - everything upto the current decoder state.

Finally, the hidden state obtained during computation of attention vector is added as an input to the next word in decoder input so that the prior information is passed on in a sequential manner to its embedding and subsequent learning.

Notice the **teacher forcing** here during training when the target word is passed as the next input to the decoder. Again, as told earlier, it is an aspect only built-in during the training and the inference will act without it.

During the inference, everything is same except that teacher-forcing isn't implemented and the 'prediction' word from the model itself is fed as the next input to the decoder (along with the attention vector)

I believe that was a lot to take in. Here are some articles I referred when I tried to understand it all and I hope that these will be able to get you some more in-depth knowledge and insights -

1. Base Article - https://www.tensorflow.org/tutorials/text/image_captioning
2. Understanding 'Show, Attend & Tell' - https://arxiv.org/pdf/1502.03044.pdf
3. Understanding Pre-trained CNN - https://towardsdatascience.com/4-pre-trained-cnn-models-to-use-for-computer-vision-with-transfer-learning-885cb1b2dfc

## Section 4: Training the Model

An Object-oriented approach is applied as the Tensorflow-Keras libaries don't have predefined layers that can incorporate this architecture.

Once the classes are formulated and model has been built (along with loss calculation and optimizer defined), the tf.GradientTape() function will be implemented to train the model on each batch and update the gradients of the trainable parameters. The model is trained with epochs set as 100 (with 3 patience), with shuffling of data in each epoch. For more information on this, please refer the notebook on how everything is defined and formulated.

## Section 5: Inference from the Model

**1. Greedy Search**

Greedy Search is the most basic inference algorithm. It takes the word with the highest probability at each output from the decoder input. This word is then fed to the next time step of the decoder to predict the next word until we hit the 'end' signal

Some outputs from Greedy Search:-

![fig8](https://user-images.githubusercontent.com/55252306/117554249-739e5680-b024-11eb-8a79-1f8264ef3189.PNG)

![fig9](https://user-images.githubusercontent.com/55252306/117554254-7731dd80-b024-11eb-97f5-56424895d06e.PNG)


**2. Beam Search**

Beam Search is slightly complicated. It produces K (which is user-defined) number of translations based on highest conditional probabilities of the words. It has been explained in detail in one of my other projects (link - https://github.com/garganm1/Neural-Machine-Translation-with-Bahdanau-Attention). Please see that to understand how the algorithm works.

Some outputs from Beam Search:-

![fig10](https://user-images.githubusercontent.com/55252306/117554260-7d27be80-b024-11eb-9c6f-d854cdaf17bb.PNG)

![fig11](https://user-images.githubusercontent.com/55252306/117554263-7f8a1880-b024-11eb-9403-93cc5f9cd83d.PNG)


## Section 6: Evaluation of the Model

Any model has to be evaluated to know how good it is or to compare it with other models.

Captions are textual data and BLEU is a metric that can help to evaluate the translations with the correct translations that should be given. It is based on an n-gram model where it looks at the words appearing in the candidate translation with the reference translation (with combinations). There are obvious limitations to this evaluation such as it will look at the word's and nearby words' positions only.

NLTK's bleu_score library gives this functionality. It looks at 1-gram to 4-gram and gives an average value (not exactly average) for how good the translations match with each other. You can evaluate the model on a test set using corplus_bleu. For more information, do refer nltk library and this article - https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

On testing the model with BLEU, we find that even though the generated captions are quite close to what is there in the image but the BLEU score is quite low. BLEU ranges from 0(denoting no match) to 1(denoting perfect match). As stated earlier, this is a limitation of BLEU but it is still widely practiced as a metric to measure textual models' performances.
