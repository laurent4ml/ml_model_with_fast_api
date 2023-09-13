# Census Classifier

## Goal
Build a MLOps Pipeline using Github Actions and deploy it to a Cloud Service Provider

## Project Setup
This multiple step project is using python version 3.

## Using CLI
Running all steps through CLI
```
bash deploy/jobs/workloads.sh
```
workloads.sh file content:
```
# All Steps are defined below
python src/model_build/prepare_data/prepare.py            # prepare dataset
python src/model_build/data_preparation/wandb_upload.py   # track dataset
python src/model_build/data_split/train_test_split.py     # split dataset
pytest --dataset-loc=$DATASET_LOC tests/data ...          # test data
python -m pytest tests/code --verbose --disable-warnings  # test code
python src/train_model.py --project-name "census-cl" ...  # train model
python src/model_deploy/evaluate.py --run-id $RUN_ID ...  # evaluate model
pytest --run-id=$RUN_ID tests/model ...                   # test model
python src/model_deploy/serve.py --run_id $RUN_ID         # serve model
```

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

python src/model_build/data_preparation/wandb_upload.py --file="./data/census.csv" --artifact_name census --artifact_type=dataset --artifact_description="census dataset for ml project"
```

This will create a new project in WandB and store census.csv as an artifact

## Step 3: Train Test Split
to create the train and test data
```
python src/model_build/data_split/train_test_split.py --input_artifact="laurent4ml/census-classification/census_project:v0" --artifact_root="census" --artifact_type=dataset --test_size=0.20
```

This will store two datasets in WandB and in the "data" local directory. One for trainning and one for validation.

## Step 4: Test Data
Data testing of 3 csv files
```
pytest src/test/data -vv --input_artifact laurent4ml/census-classification/census_project:latest
pytest src/test/data -vv --input_artifact laurent4ml/census-classification/census_train.csv:v0
pytest src/test/data -vv --input_artifact laurent4ml/census-classification/census_test.csv:v0
```

## Step 5: Test Model
```
pytest src/ml -vv --input_artifact laurent4ml/census-classification/census_train.csv:latest
```

## Step 6: Model Training
train model
```
python src/train_model.py --artifact_root="census" --project_name="census-classification" --model_file="lr_model_5.joblib"
```
This step trains the model and store the models on local

## Step 7: Evaluate Model
evaluate model
```
python src/evaluate_model.py --artifact_root="census" --project_name="census-classification" --model_file="lr_model_5.joblib"
```
This step returns metrics about the model performances like
```
auc: 0.7760486473070242
precision: 0.7571884984025559
recall: 0.6196078431372549
fbeta: 0.6815240833932422
```
but also performance on specific data slice like "edication"
```
education - Prof-school - {'precision': 0.8285714285714286, 'recall': 0.8285714285714286, 'f1': 0.8285714285714286}
education - HS-grad - {'precision': 0.8655086848635236, 'recall': 0.8655086848635236, 'f1': 0.8655086848635236}
education - Bachelors - {'precision': 0.8158415841584158, 'recall': 0.8158415841584158, 'f1': 0.8158415841584158}
education - Some-college - {'precision': 0.8372268274302939, 'recall': 0.8372268274302939, 'f1': 0.8372268274302939}
education - 7th-8th - {'precision': 0.9345794392523364, 'recall': 0.9345794392523364, 'f1': 0.9345794392523366}
education - Masters - {'precision': 0.8372781065088757, 'recall': 0.8372781065088757, 'f1': 0.8372781065088757}
education - 11th - {'precision': 0.9514563106796117, 'recall': 0.9514563106796117, 'f1': 0.9514563106796117}
education - Assoc-acdm - {'precision': 0.8117647058823529, 'recall': 0.8117647058823529, 'f1': 0.8117647058823529}
education - Assoc-voc - {'precision': 0.7932489451476793, 'recall': 0.7932489451476793, 'f1': 0.7932489451476793}
education - 5th-6th - {'precision': 0.9859154929577465, 'recall': 0.9859154929577465, 'f1': 0.9859154929577465}
education - 10th - {'precision': 0.9408284023668639, 'recall': 0.9408284023668639, 'f1': 0.9408284023668639}
education - 12th - {'precision': 0.925, 'recall': 0.925, 'f1': 0.925}
education - Doctorate - {'precision': 0.7222222222222222, 'recall': 0.7222222222222222, 'f1': 0.7222222222222222}
education - 1st-4th - {'precision': 0.9375, 'recall': 0.9375, 'f1': 0.9375}
education - 9th - {'precision': 0.9647058823529412, 'recall': 0.9647058823529412, 'f1': 0.9647058823529412}
education - Preschool - {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
```
This information is also available in Weight And Biases dashboard.

## Step 8: Rest API using FastAPI
Install app on local machine
```
pip install fastapi
pip install "uvicorn[standard]"
uvicorn main:app --reload
```

call the inference endpoint from Postman
url: POST http://127.0.0.1:8000/predict using a body query like in the tests below

## Step 9: Test Rest API
Test #1:
body:
```
{
    "workclass": "Never-worked",
    "education": "Preschool",
    "maritalstatus": "Divorced",
    "occupation": "Priv-house-serv",
    "relationship": "Wife",
    "race": "White",
    "sex": "Female",
    "nativecountry": "United-States",
    "capitalgain": 0,
    "capitalloss": 10000,
    "fnlwgt": 0,
    "age": 22,
    "educationnum": 1,
    "hoursperweek": 0
}
```
response:
```
{
    "results": "<50K"
}
```

Test #2:
```
{
    "workclass": "Private",
    "education": "Doctorate",
    "maritalstatus": "Divorced",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "nativecountry": "United-States",
    "capitalgain": 1000000,
    "capitalloss": 10000,
    "fnlwgt": 0,
    "age": 52,
    "educationnum": 16,
    "hoursperweek": 70
}
```
response:
```
{
    "results": ">50K"
}
```

```
pytest test_main.py -vv
```

You should get something like
```
test_main.py::test_api_locally_get_root PASSED [ 33%]
test_main.py::test_post_result_over_50k PASSED [ 66%]
test_main.py::test_post_result_under_50k PASSED [ 100%]
```

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
