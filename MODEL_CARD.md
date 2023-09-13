
# Model Card

## Model Details
This is a logistic regression model which aims to classifiy wether or not an individual has an income over $50,000 based on various demographics features.

## Intended Use
This model can be used in fairness related studies that compare inequalities accross sex and race based on people's annual incomes.

## Metrics
The model was evaluated using Fbeta, precision and recall score.
The precision is 0.7582065652522018
recall: 0.6189542483660131
fbeta: 0.6815401223461677
We also analysed the same metrics on slices of data for each categorical features.

## Training Data
The model is trained on the UCI Census Income Dataset. (https://archive.ics.uci.edu/dataset/20/census+income).

## Dataset Columns
        "age"
        "workclass"
        "fnlwgt"
        "education"
        "education-num"
        "marital-status"
        "occupation"
        "relationship"
        "race"
        "sex"
        "capital-gain"
        "capital-loss"
        "hours-per-week"
        "native-country"
        "income"

## Evaluation Data
20% of the training dataset is used for model testing

## Caveats and Recommendations
The data contains biases and should be considered only for training purposes.
