from enum import Enum, IntEnum
import logging

# import numpy as np
import wandb
from src.config import api_config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import pandas as pd
from joblib import load
from src.ml.model import inference
from src.ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Instantiate the app.
app = FastAPI(
    title="Census Classification",
    description="Classify an individual income",
    version="0.1",
)


class WorkclassEnum(str, Enum):
    private = "Private"
    selfempnotinc = "Self-emp-not-inc"
    selfempinc = "Self-emp-inc"
    federalgov = "Federal-gov"
    localgov = "Local-gov"
    stategov = "State-gov"
    withoutpay = "Without-pay"
    neverworked = "Never-worked"


class EducationnumEnum(IntEnum):
    doctorate = 16
    profschool = 15
    masters = 14
    bachelors = 13
    assocacdm = 12
    assocvoc = 11
    somecollege = 10
    hsgrad = 9
    twelve = 8
    eleventh = 7
    tenth = 6
    nineth = 5
    seventheigth = 4
    fifthtosixth = 3
    firsttofourth = 2
    preschool = 1


class EducationEnum(str, Enum):
    bachelors = "Bachelors"
    somecollege = "Some-college"
    eleventh = "11th"
    hsgrad = "HS-grad"
    profschool = "Prof-school"
    assocacdm = "Assoc-acdm"
    assocvoc = "Assoc-voc"
    nineth = "9th"
    seventhtoeigth = "7th-8th"
    twelve = "12th"
    masters = "Masters"
    firsttofourth = "1st-4th"
    tenth = "10th"
    doctorate = "Doctorate"
    fifthtosixth = "5th-6th"
    preschool = "Preschool"


class MaritalStatusEnum(str, Enum):
    marriedcivspouse = "Married-civ-spouse"
    divorced = "Divorced"
    nevermarried = "Never-married"
    separated = "Separated"
    widowed = "Widowed"
    marriedspouseabsent = "Married-spouse-absent"
    marriedafspouse = "Married-AF-spouse"


class OccupationEnum(str, Enum):
    techsupport = "Tech-support"
    craftrepair = "Craft-repair"
    otherservice = "Other-service"
    sales = "Sales"
    execmanagerial = "Exec-managerial"
    profspecialty = "Prof-specialty"
    handlerscleaners = "Handlers-cleaners"
    machineopinspct = "Machine-op-inspct"
    admclerical = "Adm-clerical"
    farmingfishing = "Farming-fishing"
    transportmoving = "Transport-moving"
    privhouseserv = "Priv-house-serv"
    protectiveserv = "Protective-serv"
    armedforces = "Armed-Forces"


class RelationshipEnum(str, Enum):
    wife = "Wife"
    ownchild = "Own-child"
    husband = "Husband"
    notinfamily = "Not-in-family"
    otherrelative = "Other-relative"
    unmarried = "Unmarried"


class RaceEnum(str, Enum):
    white = "White"
    asianpacislander = "Asian-Pac-Islander"
    amerindianeskimo = "Amer-Indian-Eskimo"
    other = "Other"
    black = "Black"


class SexEnum(str, Enum):
    female = "Female"
    male = "Male"


class NativeCountryEnum(str, Enum):
    unitedstates = "United-States"
    cambodia = "Cambodia"
    england = "England"
    puertorico = "Puerto-Rico"
    canada = "Canada"
    germany = "Germany"
    outlyingus = "Outlying-US(Guam-USVI-etc)"
    india = "India"
    japan = "Japan"
    greece = "Greece"
    south = "South"
    china = "China"
    cuba = "Cuba"
    iran = "Iran"
    honduras = "Honduras"
    philippines = "Philippines"
    italy = "Italy"
    poland = "Poland"
    jamaica = "Jamaica"
    vietnam = "Vietnam"
    mexico = "Mexico"
    portugal = "Portugal"
    ireland = "Ireland"
    france = "France"
    dominicanrepublic = "Dominican-Republic"
    laos = "Laos"
    ecuador = "Ecuador"
    taiwan = "Taiwan"
    haiti = "Haiti"
    columbia = "Columbia"
    hungary = "Hungary"
    guatemala = "Guatemala"
    nicaragua = "Nicaragua"
    scotland = "Scotland"
    thailand = "Thailand"
    yugoslavia = "Yugoslavia"
    elsalvador = "El-Salvador"
    trinadadtobago = "Trinadad&Tobago"
    peru = "Peru"
    hong = "Hong"
    holandnetherlands = "Holand-Netherlands"


# Declare the data object with its components and their type.
class Inference(BaseModel):
    age: int = None
    workclass: str = None
    fnlwgt: int = None
    educationnum: int = None
    education: str = None
    maritalstatus: str = None
    occupation: str = None
    relationship: str = None
    race: str = None
    sex: str = None
    nativecountry: str = None
    capitalgain: int = None
    capitalloss: int = None
    hoursperweek: int = None


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": api_config.app.greet_message}


@app.post("/predict/")
async def _predict(input: Inference):
    project_name = api_config.app.project_name

    run = wandb.init(project=project_name, job_type="inference")

    X = pd.DataFrame(
        {
            "workclass": str(input.workclass),
            "education": str(input.education),
            "marital-status": str(input.maritalstatus),
            "occupation": str(input.occupation),
            "relationship": str(input.relationship),
            "race": str(input.race),
            "sex": str(input.sex),
            "native-country": str(input.nativecountry),
            "capital-gain": input.capitalgain,
            "capital-loss": input.capitalloss,
            "fnlwgt": input.fnlwgt,
            "age": input.age,
            "education-num": input.educationnum,
            "hours-per-week": input.hoursperweek,
        },
        index=[0],
    )

    wandb_encoder = f"laurent4ml/model-registry/{project_name}-encoder:latest"
    artifact_encoder = run.use_artifact(wandb_encoder, type="model")
    encoder_path = artifact_encoder.file()
    logger.info(f"main - encoder_path: {encoder_path}")

    try:
        onehotencoder = load(encoder_path)
        # onehotencoder = load("./model/encoder.joblib")
        # label_binarizer = load("./model/label_binarizer.joblib")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model not found: {e}")

    logger.info("Processing training data")
    try:
        X, _, _, _ = process_data(
            X,  # pd.DataFrame
            categorical_features=cat_features,
            training=False,
            encoder=onehotencoder,
            label_binarizer=None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Process data error: {e}")

    X = X.reshape(1, -1)
    logger.info(f"main - X shape: {X.shape}")
    logger.info(f"main - X: {X}")

    # download best model from Weight and Biases
    wandb_model = f"laurent4ml/model-registry/{project_name}-model:latest"
    artifact = run.use_artifact(wandb_model, type="model")
    model_path = artifact.file()
    logger.info(f"main - model_path: {model_path}")

    logger.info("Start inference")
    try:
        predictions = inference(model_path, X)
    except ValidationError as e:
        raise HTTPException(
            status_code=500, detail=f"Inference Validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    logger.info("End inference")
    logger.info(f"predictions[0] : {predictions[0]}")
    logger.info(f"predictions : {predictions}")
    pred = int(predictions[0])
    if pred == 0:
        return {"results": "<50K"}
    else:
        return {"results": ">50K"}
    wandb.finish()
