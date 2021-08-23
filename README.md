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
"coco_annotation.zip" archive and creates 4 textual files containing 10000 entries each one