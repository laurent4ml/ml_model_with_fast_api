import requests

url = "https://census-classification.onrender.com/predict"
data = {
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
headers = {"Content-type": "application/json"}

response = requests.post(url, json=data, headers=headers)

print(response.status_code)
print(response.json())
