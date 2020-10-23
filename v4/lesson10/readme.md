# Natural Language Processing

Uses pretrained WIkitext model and then IMDb to fine tune the model

It teaches two things :
	1. creating a dataloader for language model : this will generate the next word
	2. creating a dataloader for classifier : this is an external label like sentiment


## NLP introduction

nlp is about guessing the next word. for that the model needs to create its own labels. It uses self supervised learning.

The language model used to classify IMDb was pretrained on Wikipedia. 

why learn in detail ?

One reason, of course, is that it is helpful to understand the foundations of the models that you are using. But there is another very practical reason, which is that you get even better results if you fine-tune the (sequence-based) language model prior to fine-tuning the classification model

We will be finetuning the pretrained language model which was trained on wikipedia articles.

This is called ULMFit approach.

### text preprocessing

we already know how categorical variables can be used as independent variables for neural network

make a list of all possible levels of the variable (vocab)

Replace each level with its index in the vocab

create an embedding matrix for this contatining a row for each level i.e. for each item in the vocab

Use this embedding matrix as the first layer of a neural network.

we do the same thing with text. 

whats new is idea of sequence. first we concatenate all of the document in our dataset into onw big long string and split it into words giving us very long words or tokens. 

Our independent variable will be the entire string exect for the second last and last will be labe;

Our vocab would be mix of common words from wikipedia and new words specific to our corpus would be movie actors

for building embedding matrix: for words in vocabualary of pre trained modelwe will take corresponding row in the embedding matrix of the pretrained model but for new words we won't have anythong, we willjust initialize the corresponding row with a random vector

## jargon

tokenisation

Numericalisations - making list of unique words -vocab and convert each word to index to look up in vocab

![tokenisation](./img/tokenisation_initial.png)

Language model data loader creation -  LMDDataLoader class for seperating the last token as label

Language model creation - creating a model that handles the input list that are arbitaryily small or big.

![preprocessing](./img/preprocessing.png)
