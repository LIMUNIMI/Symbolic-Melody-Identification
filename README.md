# A Convolutional Approach to Melody Line Identification in Symbolic Scores
This repository contains the code, the results and the datasets used for the work
submitted to ISMIR 2019 about an approach for using Convolutional Neural Networks for
melody identification in symbolic scores. You can surf this website, view and clone it on
[github](https://github.com/2019paper/Symbolic-Melody-Identification) or download the entire
[archive](https://github.com/2019paper/Symbolic-Melody-Identification/tarball/master)

## Abstract
In many musical traditions, the main melody line is of primary
significance in a piece. Human listeners are able to readily distinguish
melodies from accompaniment; however, making this distinction
given only the written score – i.e. without listening
to the music performed – is a task of high complexity,
which requires knowledge of music theory. Solving this task
is of great importance for both Music Information Retrieval and
musicological applications.

In this paper, we propose an automated approach to
identifying the most salient melody line in a symbolic score.
The backbone of the method consists in a convolutional
neural network (CNN) estimating the probability that each
note in the score (more precisely: each pixel in a piano roll
encoding of the score) belongs to the main melody line.

We train and evaluate the proposed method on various
datasets, using manual annotations where available and
solo instrument parts where not. We also propose a method
to inspect the CNN and to analyze the influence exerted by
notes on the prediction of other notes; this method
can be applied whenever the output of a neural network
has the same size as the input. 

Full paper: [link]

## Summary
The method starts from a score and tries to identify the notes belonging to the melody.

Starting from here 
![Sor](./sor-op35n22.png)

it attempts to output this
![Sor Melody](./sor-op35n22-melody.png)

To achieve that, we use a Convolutional Neural Network and a graph approach. First, the score is converted into a Boolean pianoroll; then, the pianoroll is processed by the CNN whose output is a new pianoroll where each pixel has a probability value of being a part of the melody. Based on this probabilities, we can reconstruct the melody line.

We show that our method is better than sate-of-art, but that further effort is needed.

---

## References
Please, cite us as [...]
