# NLP deep dive


## Objective 

1. creating our own model of human numbers
2. creating an RNN
3. it helps us to understand the RNN in detail
4. best explanationfor LSTM

## rnn

Looping language model for creating rnn.

RNN architechture:

![rnn_flowchart](./img/rnn_flowchart.png)

the loop :

![rnn_loop](./img/rnn_loop.png)

Multi layer rnn:

![multi_rnn](./img/multirnn.png)

Multi layer RNN unrolled:

![multi_rnn_unrolled](./img/multirnn_unrolled.png)

LSTM:

LSTM is an architecture that was introduced back in 1997 by Jürgen Schmidhuber and Sepp Hochreiter. In this architecture, there are not one but two hidden states. In our base RNN, the hidden state is the output of the RNN at the previous time step. That hidden state is then responsible for two things:

Having the right information for the output layer to predict the correct next token
Retaining memory of everything that happened in the sentence
Consider, for example, the sentences "Henry has a dog and he likes his dog very much" and "Sophie has a dog and she likes her dog very much." It's very clear that the RNN needs to remember the name at the beginning of the sentence to be able to predict he/she or his/her.

In practice, RNNs are really bad at retaining memory of what happened much earlier in the sentence, which is the motivation to have another hidden state (called cell state) in the LSTM. The cell state will be responsible for keeping long short-term memory, while the hidden state will focus on the next token to predict. Let's take a closer look and how this is achieved and build an LSTM from scratch.

![lstm](lstm.png)