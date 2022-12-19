# Madrid house price predictor
## About the project:
Are you trying to figure out your dream house price in Madrid Spain?  
It is complicated to look to multiple sites to find out an approximate house price, with this model, you can provide the features such as neighborhood, bedroom number, etc, and get an approximate price of a house in Madrid with that characteristic.

**This project was made for learning and fun purposes and is not a production service**

## About the data
I scraped the data from a real estate website in October 2022,  the data has not been updated since then, and the last update is relevant because the house  price fluctuation. 

## About the model
 - Model trained with 11404 observations
 - Last Update 2022/11/05
 - The model used is Random Forest
 - Available parameters (all mandatory):  House type, House type 2, Rooms, m2, Elevator, Garage, District & Neighborhood.
 
## How to run the model
 - Download the files
 - Run train.py

**Option 1**
 - Build Docker container: `docker build -t madrid_house_price .`
 - Run Docker container: `docker run -it -p 9696:9696 madrid_house_price:latest`

