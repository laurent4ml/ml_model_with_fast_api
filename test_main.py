from fastapi.testclient import TestClient
import json

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_post_result_over_50k():

    data = json.dumps(
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
            "hoursperweek": 70,
        }
    )
    r = client.post("/predict", data=data)
    assert r.json() == {"results": ">50K"}


def test_post_result_under_50k():

    data = json.dumps(
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
            "hoursperweek": 0,
        }
    )
    r = client.post("/predict", data=data)
    assert r.json() == {"results": "<50K"}
