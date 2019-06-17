# A Convolutional Approach to Melody Line Identification in Symbolic Scores
This repository contains the code, the results and the datasets used for the work
presented at the ISMIR 2019 about an approach for using Convolutional Neural Networks for
melody identification in symbolic scores. You can surf this website, view and clone it on
[github](https://github.com/ofai/Symbolic-Melody-Identification) or download the entire
[archive](https://github.com/ofai/Symbolic-Melody-Identification/tarball/master)

A work by [OFAI - Austrian Research Institute for Artificial Intelligence](http://ofai.at/) & [LIM - Music Informatics Laboratory](https://www.lim.di.unimi.it/)

## Abstract
In many musical traditions, the melody line is of primary
significance in a piece. Human listeners can readily distinguish
melodies from accompaniment; however, making this distinction
given only the written score – i.e. without listening
to the music performed – can be a difficult task.

Solving this task
is of great importance for both Music Information Retrieval and
musicological applications.
In this paper, we propose an automated approach to
identifying the most salient melody line in a symbolic score.

The backbone of the method consists of a convolutional
neural network (CNN) estimating the probability that each
note in the score (more precisely: each pixel in a piano roll
encoding of the score) belongs to the  melody line.
We train and evaluate the method on various
datasets, using manual annotations where available and
solo instrument parts where not. We also propose a method
to inspect the CNN and to analyze the influence exerted by
notes on the prediction of other notes; this method
can be applied whenever the output of a neural network
has the same size as the input.

Full paper: [link](./paper.pdf)

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
Please, cite us as:

Federico Simonetta, Carlos Cancino-Chacón, Stavros Ntalampiras & Gerhard Widmer. (2019). "A Convolutional Approach to Melody Line Identification in Symbolic Scores". In _Proceedings of the 20th International Society for Music Information Retrieval Conference_. Delft, The Netherlands.

## Acknowledgments

_This research has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme under grant agreement No. 670035 (project "Con Espressione")._

![ERC logo](./1.jpg)
_We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan V GPU used for this research._

_We thank Elaine Chew for sharing the code for VoSA. We thank Laura Bishop for proofreading an earlier version of this manuscript._
