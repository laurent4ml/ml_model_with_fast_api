# Building a MLOps Pipeline with Github, Github Actions and a Cloud Service Provider

## Preparing Data
to load the data set just run
```
python src/model_build/prepare_data/prepare.py
```

This will create a folder 'data' and store the dataset in a file census.csv

## Github Actions

We will use Github Actions to:
- Automate Data Download
- Check whether the model should be retrained or not
- Compare models
- Put a model into production by uploading the new model to a Cloud Provider and deploying it
- Run tests

To add a github action create a yaml file in .github/workflow

The first Github Action load the census file to WandB artifacts. This is manaual action that should performed first so that the data is available for the next actions.
Filename: data_download.yaml

## Install 

First you need to install dependencies in conda
- pip install wandb
