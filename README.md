# Image Captioning in Ambiguous Scenes

See Report.pdf for a detailed description of this project.

Image captioning is one of the hottest tasks in deep learning as it allows to study of hybrid/cross-modal approaches between computer vision and natural language processing. Consequently, it is a heavily researched field with a good number of existing models for different aims (i.e. generating images or predicting a description). There is a gap in studying non-standard object images, such as from ambiguous scenes.

## Getting started

- Download Conda and run the following command:

```bash
conda env create -f ./env.yml
```

- Download the [VDQG dataset](https://mmlab.ie.cuhk.edu.hk/projects/vdqg/#dataset) and unzip in the [data](./data) directory.

- Then navigate to [Run Commands](#run-commands) to run the pipeline.

## Project setup

The index.py file is used to run the whole pipeline. For configuration examples see [section](#run-commands). For further information on the configuration see file [src/config.py](./src/config.py).

### Results

Results are collected in the [metrics](./metrics/) folder. This includes:

- Loss plots

- Evaluation metrics of model

Generated questions are stored in [predictions](./predictions/) folder.

## Exploring the results

The note notebook [presentation.ipynb](./presentation.ipynb) contains examples of generated questions.

The two model weights are:

- Approach random trained on image captions from GIT-BASE.

- Approach random trained on image captions from BLIP.

Each generated question is accompanied an image pair, the ground truth questions, the positive ground truth questions, BLEU score, METEOR score and &Delta;BLEU score.

## Exploring the dataset

The note notebook [stats.ipynb](./stats.ipynb) contains general stats of the VDQG dataset such as:

- Occurrences of positive questions.

- Occurrences of color in positive questions.

- The top 20 most used positive question.

## Run commands

- Run the whole pipeline:

```bash
python index.py --mode=all --num_train_epochs=20
```

- To only fine tune the model given train and test data:

```bash
python index.py --mode=fine_tune --num_train_epochs=20 --test_data_path=./bert_data/test.json --train_data_path=./bert_data/train.json
```

- To only run test with a pre-trained model:

```bash
python index.py --mode=test --timestamp=11-29-20H-49M --num_train_epochs=3
```

- Run the whole pipeline with processing two images:

```bash
python index.py --mode=all --num_train_epochs=10 --process_pair=true
```

- Fine tune with small example:

```bash
python index.py --mode=fine_tune --test_data_path=./bert_data/test.json --train_data_path=./bert_data/train.json --num_train_epochs=1 --data_size=10 --approach=train_rand
```

## Conda commands

### Updating env

```bash
conda env update --file env.yml --prune
```

### Creating env

```bash
conda env create -f ./env.yml
```

### Activating/deactivating

```bash
# To activate this environment, use
conda activate image-captioning

# To deactivate an active environment, use
conda deactivate
```

### Running through CUDA

#### Install CUDA Toolkit

Install the latest version of [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) that is compatible with packages from env. In my case it was CUDA Toolkit 11.8.0, due to PyTorch not supporting newer. See [this](https://discuss.pytorch.org/t/cuda-versioning-and-pytorch-compatibility/189777)

Install cuda in the conda env

```bash
conda install cuda -c nvidia
```

#### PyTorch version

Ensure that the pytorch version in the env is compatible with cuda by running the following.

```bash
python

>>> import torch
>>> torch.cuda.is_available()
```

If it returns false, see links below:

[Similar issue on StackOverflow](https://stackoverflow.com/questions/57238344/i-have-a-gpu-and-cuda-installed-in-windows-10-but-pytorchs-torch-cuda-is-availa)

[Pytorch installer helper](https://pytorch.org/get-started/locally/)

## Common errors

```bash
ModuleNotFoundError: No module named '_distutils_hack'

# Fix
pip install -U pip setuptools
```

## Articles used in project

[Transfer learning in Huggins](https://huggingface.co/docs/transformers/training)

[Project formalities](https://github.itu.dk/AML4KCS/AML4KCS2023/blob/main/project/description.md)

[Project image captioning](https://github.itu.dk/AML4KCS/AML4KCS2023/blob/main/project/p_image_captioning.md)

[GitBase model description](https://huggingface.co/microsoft/git-base)

## Papers used in project

[Generative Image-to-text Transformer](https://arxiv.org/pdf/2205.14100.pdf)

[VDQG paper](https://personal.ie.cuhk.edu.hk/~ccloy/files/iccv_2017_learning.pdf)

[BERT for Question Generation paper](https://aclanthology.org/W19-8624.pdf?fbclid=IwAR0RIqxGDKTuD34-ucyPfJPI1io9JN1y4nA4WIYfPJAN1jiN7jrmHVYG5Qc)

[A Recurrent BERT-based Model for Question Generation](https://aclanthology.org/D19-5821.pdf)
