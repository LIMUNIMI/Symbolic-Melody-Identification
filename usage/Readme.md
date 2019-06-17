# USAGE
This is the description of all the options. 
The entry point is the Python 2 script `terminal_client.py` in the root directory.
Simply launch it without arguments to get this help.

A few examples:
* Train over all files with extension `.ext` in `mydirectory` using hyper-parameters
  contained in `parameters.json` (saves kernels to a pickle object):
> `./terminal_client.py --train mydirectory .ext parameters.json`

* Use the kernels saved in `kernels.pkl` to compile a model for your architecture:
> `./terminal_client.py --rebuild kernels.pck parameters.json model.pkl`

* Use the model `model.pkl` to find the melody in `file.mxl` (compressed MusicXML) and
  creates a MIDI file with separated tracks `output.mid`:
> `./terminal_client.py --model model.pkl --extract file.mxl output.mid`

* Inspect the 10th window of file `input.mid` and saves all the 30.000 thousands
  inspection windows:
> `./terminal_client.py --inspect input.mid 10`

```
usage: ./terminal_client.py [-h] [--extract INPUT OUTPUT]
                         [--inspect-masking INPUT N] [--inspect INPUT]
                         [--model PATH] [--install-deps] [--check-deps]
                         [--rnn] [--time-limit INT] [--epochs INT] [--mono]
                         [--train DIR .EXT FILE]
                         [--crossvalidation DIR .EXT FILE]
                         [--validate DIR .EXT MODEL]
                         [--rebuild KERNELS PARAMETERS OUTPUT]
                         [--hyper-opt DIR .EXT FILE]

Takes in input a model and a symbolic music file and
outputs a new symbolic music file containing solo part and accompaniment
separation. At least one and no more than one option must be provided
between `--extract`, `--train`, `--crossvalidation`, `--hyperopt`, `--rebuild`.
If more than one is provided the first one in the previous list will be used.

optional arguments:
-h, --help            show this help message and exit
--extract INPUT OUTPUT
                     Extract a solo part from INPUT and create a new
                         OUTPUT file containing the accompaniment and the solo
                         part in two separeted tracks (if midi) or parts (if musicxml)
                     The file types are inferred from the extensions.
                     Supported input files: MIDI (.mid, .midi), MusicXML (.xml, .mxl),
                         ABC (.abc), Kern (.krn), and anything supported by music21 library
                         In addition, it supports the loading of pickled objects(.pyc.bz)
                         containing structered arrays:
                         * Each row corresponds to a note.
                         * The fields must be(`pitch`, `onset`, `duration`, `soprano`).
                                 `pitch` is the MIDI pitch of the note.
                                 `onset` is the score position(in beats) of the note.
                                 `duration` is the duration of a note(in beats).
                                 `soprano` is a boolean indicating whether the current
                                     note is part of the main melody.
                     Supported output files: MIDI(.mid, .midi) type 1; ".mid" extension
                         will be added if not provided.
--inspect-masking INPUT N
                     Perform inspection by using masking windows
                     on the window number N of song INPUT.
                     Create numpy compressed file containing the input pianoroll (`in_pianoroll`),
                     the output pianoroll (`out_pianoroll`), the ground truth melody (`melody`),
                     the masked saliency map (`saliency`) and the whole masked outputs (`masked`)
--inspect INPUT       Perform standard inspection on the INPUT song.
                     Create numpy compressed file containing the input pianoroll (`in_pianoroll`),
                     the output pianoroll (`out_pianoroll`), the ground truth melody (`melody`)
                     and the salience map.
                     The file types are inferred from the extensions, as in `--extract`.
--model PATH          A pickled object containing a `predict` method. Usually it's
                         a CNN or RNN object. By default the trained models provided with
                         the software will be used. If a custom network is provided, you
                         should take care that it comes with a `win_width` field, as provided
                         by `melody_extractor.trainer.build_?NN_model`
--install-deps        Automatically install missing libraries in the user home
--check-deps          Automatically check needed libraries on startup. If
                         `--install-deps` is not used, a command to install the missing
                         dependencies will be printed
--rnn                 Use an RNN and non-overlapping windows instead of a CNN
                         with overlapping windows
--time-limit INT      Break the training if the time exceeds the specified
                         limit in seconds (default 120 sec)
--epochs INT          Set the maximum number of epochs. Default 15000.
                         Note that the training already uses early stop algorithm. This option is
                         particularly useful for RCNNs.
--mono                Find a strictly monophonic solo part.
--train DIR .EXT FILE
                     Train the model on files in DIR (and subdirectories)
                         having extension .EXT. Write the trained model to a pickled
                         object in this directory. Use parameters contained in FILE.
                         Wite also parameters to be used with `--rebuild` option for
                         rebuilding the network on a different architecture
--crossvalidation DIR .EXT FILE
                     Perform a 10-fold crossvalidation on files in DIR
                         (and subdirectories) having extension .EXT. Write results to
                         files in this directory. Use parameters contained in FILE as
                         exported with `--hyper-opt`. At each fold, it save a pickled object
                         containing the kernels of the network; you can use these to rebuild
                         the network on a different architecture (`--rebuild` option)
--validate DIR .EXT MODEL
                     Validate the MODEL on files in DIR and sub-dir
                     of type .EXT. Writes the results in a file in the current directory
                     called `results.txt`. This only works with CNN.
--rebuild KERNELS PARAMETERS OUTPUT
                     This is provided for convenience: after
                         having trained a model (`--train`), use the output kernels
                         parameters and the parameters found with `--hyper-opt` to
                         build a new model on a different architecture without retrain
                         retrain it. This is useful if you train on GPU and then want
                         the model exported for a CPU. Write the model in OUTPUT.
--hyper-opt DIR .EXT FILE
                     Perform hyper-parameter optimization on files in DIR
                         (and subdirectories) having extension .EXT. This write the best parameters
                         FILE at each new evaluation. If for any reason the hyper-optimization
                         should stop, then you should take care that `trials.hyperopt` is still
                         in the working directory, so that the already performed evaluations will
                         not be lost.

                         N.B. Be careful about the output parameters because something seems to be
                         written wrong (maybe an hyper-opt bug?)
```
