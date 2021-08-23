# ConditianalTextGeneration

This work is based on Conditional Text Generation proposed by ctrl. It can be executed in
two simple steps:

- Creation of a python environment using conda
- Execution of the Main.py script

## Creation of environment
This step is necessary to create and activate the python environment in order to manage to install
the necessary libraries
The environment must have the name "py36", otherwise it cannot be recognized when running the
[Main.py](https://github.com/Carlo13gen/ConditionalTextGeneration/blob/main/Main.py) script. 

The Linux command to run is:

```buildoutcfg
yes | conda create -n py36 python=3.6
```

Activating the environment can be performed by running the command:

```buildoutcfg
conda activate py36
```

If everything worked properly, it should be visible the string "(py36)" at the beginning of terminal
command line, like in the example:

```buildoutcfg
(py36) user@example_machine:~$
```

## [Main.py](https://github.com/Carlo13gen/ConditionalTextGeneration/blob/main/Main.py) script execution
The Main.py script can be divided in 5 parts:

1. Libraries Installation and Model Download
2. Dataset creation
3. Training phase
4. Generation phase
5. Evaluation phase

By running the script those phases are all executed automatically.

#### Libraries Installation and Model Download
The first phase install inside the python environment, created in the previous section, the necessary libraries and it patches keras too. Furthermore
in this step the [salesforce ctrl repository](https://github.com/salesforce/ctrl) is cloned and the model "seqlen256_v1.ckpt" is downloaded into the 
"training_utils" directory inside the "ctrl" directory just cloned.

#### Dataset Creation
The Dataset Creation phase consists in running the script [Create_dataset.py](https://github.com/Carlo13gen/ConditionalTextGeneration/blob/main/Create_dataset.py). It extracts the dataset from the 
"coco_annotation.zip" archive and creates 4 textual files containing 10'000 entries each one. The training is performed on the first 40'000 captions, while the test on the captions from 40'000 to 41'000.

#### Training Phase
Training has two inner scripts:

- [make_tf_records.py](https://github.com/salesforce/ctrl/blob/master/training_utils/make_tf_records.py) 
- [Training_mod.py](https://github.com/Carlo13gen/ConditionalTextGeneration/blob/main/Training_mod.py)

The first one creates the Tensorflow records to be used for training. 
It takes in input:

- The text file from which the text is taken to make tensorflow records (--text_file)
- The control code (--control_code)
- The sequence length (--seq_len)

The second script performs the real training of the model. It takes in input:

- The model to be trained directory (--model_dir)
- The number of iterations for training (--iterations)

#### Generation Phase
The generation of text is performed by the script [Generate_captions.py](https://github.com/Carlo13gen/ConditionalTextGeneration/blob/main/Generate_captions.py), it takes in input

- The directory of the model (--model dir)
- A parameter called temperature (--temperature)
- A parameter called topk (--topk)
- The command to print only once (--print_once)
- The input file which contains the beginning of the sentences to generate (--input_file)
- A parameter called nucleus (--nucleus)

#### Evaluation Phase
After the generation phase, by running the [Evaluation.py](https://github.com/Carlo13gen/ConditionalTextGeneration/blob/main/Evaluation.py) a score is 
assigned to the generation performed.
This script takes in input three parameters:

- The file with the reference captions, which are the ideal sentences to be generated
- The file with the captions generated in the previous phase
- A file on which the scores are written

[Evaluation.py](https://github.com/Carlo13gen/ConditionalTextGeneration/blob/main/Evaluation.py) computes three different scores:

- BLEU
- SELF-BLEU
- POS-BLEU