import requests
import argparse
 
parser = argparse.ArgumentParser(description="Try predict service",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-H","--host", help="host of the service")
args = parser.parse_args()
config = vars(args)

if config["host"] == None:
  host = 'localhost:9696'
else:
  host = config["host"]
# host = 'capstone-serving-env.eba-vbshenz3.eu-west-3.elasticbeanstalk.com'
# host = 'churn-serving-env.eba-vupegvrr.eu-west-3.elasticbeanstalk.com'

url = f'http://{host}/predict'

passenger = {
  "homeplanet": "europa",
  "cryosleep": 0,
  "cabin": "e/608/s",
  "destination": "55_cancri_e",
  "age": 32,
  "vip": 0,
  "roomservice": 0,
  "foodcourt": 1049,
  "shoppingmall": 0,
  "spa": 353,
  "vrdeck": 3235
}

print('Passenger: ', passenger)

response = requests.post(url, json=passenger).json()
print('Prediction: ', response)