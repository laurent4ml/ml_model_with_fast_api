# Census Classifier

## Project
The project is training a logistic regression model which aims to classifiy wether or not an individual has an income over $50,000 based on various demographics features. Check the MODEL_CARD.md for more details.

The model is trained on the UCI Census Income Dataset. (https://archive.ics.uci.edu/dataset/20/census+income).

This project is live at this url: https://census-classification.onrender.com
The main endpoint is /predict which is a POST, check in the test section for examples.

## Goal
Build a MLOps Pipeline using Github Actions for automation, Weight and Biases for model registry and deploy the app to Renders.com a Cloud Service Provider

## Project Setup
This multiple step project is using python version 3.8.17 and the librairies in requirements.txt

## Using CLI
Running all steps using mlflow
```
# prepare dataset, track and split dataset, test data and model,
# train and evaluate model
mlflow run src

# install app
uvicorn main:app --reload

# test api
pytest test_main.py -vv

# run inference
python scripts/run_inference.py
```

## Step 1: Preparing Data
to load the data set into "data" directory
```
mlflow run src -P steps=download
```
This will create a folder 'data' at the root of the project, store the data files downloaded from the UCI website and store the cleaned dataset in a file census.csv

## Step 2: Uploading data set to WandB
to store a version of the dataset to WandB:
```
export WANDB_API_KEY=<api_key>

mlflow run src -P steps=upload
```
This will create a new project in WandB and store census.csv as an artifact

## Step 3: Train Test Split
to create the train and test data
```
mlflow run src -P steps=train_test_split
```
This will store two datasets in WandB and in the "data" local directory. One for trainning and one for validation.

## Step 4: Test Data
Data testing of 3 csv files: census.csv, census_train.csv, census_test.csv
```
mlflow run src -P steps=data_check
```

## Step 5: Test Model
```
mlflow run src -P steps=model_check
```
Run 3 test against model training results.

## Step 6: Model Training
This step is training the model based on the training data provided.
```
mlflow run src -P steps=model_training
```
Steps Performed:
- trains model
- store trained model on local
- store one hot encoder on local
- store label binarizer on local
- upload trained model to wand model registry
- upload one hot encoder to wandb model registry

## Step 7: Evaluate Model
evaluate model
```
mlflow run src -P steps=model_evaluate
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

## Step 8: Setup Rest API on local using FastAPI
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

## Step 10: Run inference script
```
python scripts/run_inference.py
```
result:
```
200
{'results': '<50K'}
```

Screenshot showing the result of the inference script
![alt text](https://github.com/laurent4ml/ml_model_with_fast_api/blob/main/images/live_post.png?raw=true)

# API Documentation
The API is built on Render.com and has two endpoints:
- GET /
- POST /predict

The responses are showed in the screenshots below.
![alt text](https://github.com/laurent4ml/ml_model_with_fast_api/blob/main/images/docs_live_get.png?raw=true)

![alt text](https://github.com/laurent4ml/ml_model_with_fast_api/blob/main/images/docs_live_post.png?raw=true)

Screenshot showing the response from the GEt endpoint in a browser window.

![alt text](https://github.com/laurent4ml/ml_model_with_fast_api/blob/main/images/live_get.png?raw=true)

# Continuous Deployment and Github Actions
The CICD system is setup on Render.com and uses github actions.
One action, python-app.yml will run tests using Pytest.
And if successful will deploy the code to Renders using a secret webhook.
```
name: Deploy
      # Only run this step if the branch is main
      if: github.ref == 'refs/heads/main'
      env:
        deploy_url: ${{ secrets.RENDER_DEPLOY_HOOK_URL }}
      run: |
        curl "$deploy_url"
```
This removes the need to set the deployment to be automatic on Render.com as showed in the screenshots below. This is a more flexible continuous deployment solution since only the action we want to succeed actually deploys the code if all steps are successfull.

![alt text](https://github.com/laurent4ml/ml_model_with_fast_api/blob/main/images/continuous_deployment.png?raw=true)

![alt text](https://github.com/laurent4ml/ml_model_with_fast_api/blob/main/images/continuous_deployment_2.png?raw=true)
