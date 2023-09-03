# Building a MLOps Pipeline with Github, Github Actions and a Cloud Service Provider

## Project Setup
This multiple step project is using python version 3. Setup is done through command line instructions.

## Step 1: Preparing Data
to load the data set:
```
python src/model_build/prepare_data/prepare.py
```

This will create a folder 'data' and store the dataset in a file census.csv

## Step 2: Uploading data set to WandB
to store a version of the dataset to WandB:
```
pip install wandb

export WANDB_API_KEY=<api_key>

python src/model_build/data_preparation/wandb_upload.py --file="./data/census.csv" --artifact_name census_project --artifact_type=dataset --artifact_description="census dataset for ml project"
```

This will create anew project in WandB and store census.csv as an artifact

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
