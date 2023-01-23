# Spaceship Titanic

## Problem description
___
For my capstone project I've choosen a dataset from a Kaggle competitition ([**Spaceship Titanic**](https://www.kaggle.com/competitions/spaceship-titanic/overview/description)). It is a new version of the famous Titanic ML competition, with different variables and a space setting. This competition is currently active.

The goal is **to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly**

To get a score in the competition, you must first built and train a model using a training dataset and then use another testing dataset to calculate and submit a set of predictions

### Used Datasets
___

Description and dowload: https://www.kaggle.com/competitions/spaceship-titanic/data
(also, a copy of the files is available in the repository)

- train.csv: Used to train the model
- test.csv: Used to predict and submit the results


### Variables of the training dataset (according to Kaggle's description)
___

- **PassengerId** - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
- **HomePlanet** - The planet the passenger departed from, typically their planet of permanent residence.
- **CryoSleep** - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
- **Cabin** - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
Destination - The planet the passenger will be debarking to.
- **Age** - The age of the passenger.
- **VIP** - Whether the passenger has paid for special VIP service during the voyage.
- **RoomService**, **FoodCourt**, **ShoppingMall**, **Spa**, **VRDeck** - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
- **Name** - The first and last names of the passenger.
- **Transported** - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

### Evaluation Metric
___

The evaluation metric is [**classification accuracy**](https://developers.google.com/machine-learning/crash-course/classification/accuracy), the percentage of predictions that are correct

___Accuracy = TP + TN / (TP + TN + FP + FN)___ where TP = True Positives, TN = True Negatives, FP = False Positives, and FN = False Negatives

### Additional dificulties
___
1. **The dataset has a good amount of missing values"** both in the training data set and in the test data set

## Approach and solutions
___
- For this classification problem I've trained the different types of algorithms seen in the course so far and I've evaluated them using **accuracy**
- I've selectioned ___XGBoost___ as the best model and tuned its parameters. 
It is worth mentioning that the differences between models have been very small.
- Although the model is very light (around 40k) and has needed very few iterations, it seems to offer quite a good result.


## Repo description
___
- `notebook.ipynb` -> Notebook for EDA, data cleaning, model testing and other preparatory steps
- `train.py` -> Script with the entire process to train the model from the initial loading of the dataset to the generation of the `.bin` file
- `predict.py` -> Creates an application that uses port 9696 and the endpoint `/predict` (`POST` method). The service:
  - Receive data request in json format
  - Validate and adapt the data to the model
  - Predict and return the result in json format
- `predict_test.py` -> Sends a test request and prints the result
- `generate_predictions.py` -> Script to generate the results to be sent to Kaggle. It creates the (`predictions.csv`) file
- `Pipfile`, `Pipfile.lock` -> For dependency management using Pipenv
- `Dockerfile` -> Required for build a docker image of the app with the 
predict service
- `XGB_model.bin` -> DictVectorizer and Model used by predict.py. Another file can be generated using train.py

## Dependencies Management
___
In order to manage the dependencies of the project you could:
- Install Pipenv => `pip install pipenv`
- Use it to install the dependencies => `pipenv install`

## Train the model
___
1. Enter the virtual environment => `pipenv shell`
2. Run script => `train.py`
or
1. Run directly inside the virtual environment => `pipenv run train.py`

Optional parameters:
- `-E --eta` (default 0.3)
- `-D --maxdepth` (default 2)
- `-R --nrounds` (default 40)
- `-S --nsplits` (default 10)
- `-O --output` (default `XGB_model.bin`)
## Run the app locally (on port 9696)
___
1. Enter the virtual environment => `pipenv shell`
2. Run script => `uvicorn predict:app --host 0.0.0.0 --port 9696`

## Using Docker
___
You can also build a Docker image using the provided `Dockerfile` file =>
`docker build . -t ss_titanic`

To run this image locally => `docker run -it --rm -p 9696:9696 ss_titanic`
## Deploying to AWS
___
Assuming you have a AWS account, you can deploy the image to AWS Elastic Beanstalk with the next steps:
- Install AWS Elastic Beanstalk CLI => `pip install awsebcli`
- Init EB configuration => `eb init -p docker -r eu-west-3 ss_titanic` and authenticate with your credentials (If you want you can change eu-west-3 for your local zone)
- Create and deploy the app => `eb create ss_titanic_env`

(**Until the end of the project review period the application will remain deployed on AWS Elastic Beanstalk** and accessible on ss-titanic-env.eba-mbs3xdhm.eu-west-3.elasticbeanstalk.com)

## Using the service
___
For a simple test request you have several options. You could:
1) Run the test script locally => `python predict_test.py` (needs `requests`. To install => `pip install requests`)

  Optional parameter =>` -H --host` (`localhost:9696` by default). For example, if you want test the AWS deployment you can run => `python predict_test.py --host ss-titanic-env.eba-mbs3xdhm.eu-west-3.elasticbeanstalk.com`

2) Or use the browser for access the `/docs` endpoint => `localhost:9696/docs`  or `ss-titanic-env.eba-mbs3xdhm.eu-west-3.elasticbeanstalk.com/docs` and then use the UI to send a request (POST) to the `/predict` endpoint

3) Or use curl, Postman or another similar tool to send the request to the `/predict` endpoint

## Generating a file to submit the predictions to Kaggle
___
To generate a .csv file with the predictions => `pipenv run generate_predictions`. Optional parameters:
- `-M --model` (default `XGB_model.bin`)
- `-O --output` (default `predictions.csv`)












