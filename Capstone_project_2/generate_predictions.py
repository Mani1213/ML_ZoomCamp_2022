import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import argparse

#Parameters
parser = argparse.ArgumentParser(description="Generate test predictions in a .csv file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-M","--model", help="model file")
parser.add_argument("-O","--output", help="output file")
args = parser.parse_args()
config = vars(args)

model_file = 'XGB_model.bin'
if config["model"] != None:
  model_file = config["model"]
output_file = 'predictions.csv'
if config["output"] != None:
  output_file = config["output"]
  
# Loading and preparing the dataset
print ('Loading and preparing the dataset...')

df_test = pd.read_csv('test.csv')

df_test.columns = df_test.columns.str.lower().str.replace(' ', '_')

df_test.cryosleep = pd.to_numeric(df_test.cryosleep, errors='coerce')
df_test.vip = pd.to_numeric(df_test.vip, errors='coerce')

df_test['groupid'] = df_test['passengerid'].apply(lambda x: int(x[:4]))
gr = pd.DataFrame(df_test['groupid'].value_counts()).reset_index().rename({'index': 'groupid', 
                      'groupid': 'groupsize'}, axis=1)
df_test = df_test.merge(gr, on='groupid', how='left')
splitcabin = df_test['cabin'].str.split('/', expand=True).rename({0: 'deck', 1: 'num', 2: 'side'}, axis=1)
df_test = df_test.merge(splitcabin, left_index=True, right_index=True, how='left')

df_test.cryosleep = df_test.cryosleep.fillna(df_test.cryosleep.median()).astype(int)
df_test.age = df_test.age.fillna(df_test.age.mean()).astype(int)
df_test.vip = df_test.vip.fillna(df_test.vip.median()).astype(int)
df_test.roomservice = df_test.roomservice.fillna(df_test.roomservice.mean()).astype(int)
df_test.foodcourt = df_test.foodcourt.fillna(df_test.foodcourt.mean()).astype(int)
df_test.shoppingmall = df_test.shoppingmall.fillna(df_test.shoppingmall.mean()).astype(int)
df_test.spa = df_test.spa.fillna(df_test.spa.mean()).astype(int)
df_test.vrdeck = df_test.vrdeck.fillna(df_test.spa.mean()).astype(int)
df_test.groupsize = df_test.groupsize.fillna(df_test.groupsize.mean()).astype(int)
df_test.homeplanet = df_test.homeplanet.fillna('earth')
df_test.destination = df_test.destination.fillna('trappist-1e')
df_test.deck = df_test.deck.fillna('f')
df_test.side = df_test.side.fillna('s')

# Doing the predictions
print("Doing the predictions with model %s..." % model_file)
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

features = ['homeplanet', 'cryosleep', 'destination', 'age',
            'vip', 'roomservice', 'foodcourt', 'shoppingmall', 'spa', 'vrdeck', 'groupsize', 'deck', 'side']    

X = dv.transform(df_test[features].to_dict(orient='records'))
features = dv.get_feature_names_out()
dX = xgb.DMatrix(X, feature_names=features)
y_pred = model.predict(dX)    

df_test['PassengerId'] = df_test['passengerid']
df_test['Transported'] = (y_pred >= 0.5)

df_test[['PassengerId','Transported']].to_csv(output_file,index=False)
print("File %s generated" % output_file)
