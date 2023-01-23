import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from sklearn.model_selection import KFold
import pickle
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

#Parameters
parser = argparse.ArgumentParser(description="Train model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-E","--eta", help="eta")
parser.add_argument("-D","--maxdepth", help="maximum depth")
parser.add_argument("-R","--nrounds", help="number of rounds")
parser.add_argument("-S","--nsplits", help="number of splits in cross validation")
parser.add_argument("-O","--output", help="output file")
args = parser.parse_args()
config = vars(args)

eta = 0.3
if config["eta"] != None:
  eta = float(config["eta"])
max_depth = 2
if config["maxdepth"] != None:
  max_depth = int(config["maxdepth"])
xgb_num_rounds = 40
if config["nrounds"] != None:
  xgb_num_rounds = int(config["nrounds"])
n_splits = 10
if config["nsplits"] != None:
  n_splits = int(config["nsplits"])
output_file = 'XGB_model.bin'
if config["output"] != None:
  output_file = config["output"]

xgb_params = {'eta': eta, 'max_depth': max_depth, 'min_child_weight': 1, 'objective': 'reg:squarederror', 'nthread': 8, 'seed': 1, 'verbosity': 1}
features = ['homeplanet', 'cryosleep', 'destination', 'age',
            'vip', 'roomservice', 'foodcourt', 'shoppingmall', 'spa', 'vrdeck', 'groupsize', 'deck', 'side']    

# Function to split dataframes
def split_dataframe(dataframe): 
  df_full_train, df_test = train_test_split(dataframe, test_size=0.2, random_state=1)

  df_full_train = df_full_train.reset_index(drop=True)
  df_test = df_test.reset_index(drop=True)

  return df_full_train, df_test

# Functions to calculate XGBoost model
def XGB_train(df_train, df_val, features, xgb_params, num_rounds):
    train_dicts = df_train[features].to_dict(orient='records')
    val_dicts = df_val[features].to_dict(orient='records')
    dv = DictVectorizer(sparse=False) # OHE
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    #X_train = df_train[features].values
    #X_val = df_val[features].values

    y_train = df_train.transported.values

    features = dv.get_feature_names_out()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, feature_names=features)
    model = xgb.train(xgb_params, dtrain, num_boost_round=num_rounds)

    return(dv,model,dtrain,dval)
  
def XGB_predict(dtrain, dval, model):
    y_pred_train = model.predict(dtrain)
    y_pred_val = model.predict(dval)

    return(y_pred_train, y_pred_val)

def XGB_calculate(features, xgb_params, num_rounds=100, print_kfold=True, print_summary=True,n_splits=5):
    # Cross-Validating
    print('ETA: ', xgb_params['eta'], 'max_depth: ', xgb_params['max_depth'], 'num_rounds: ', num_rounds)
    if print_kfold: print('Doing validation with %s splits' % n_splits)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores_train = []
    scores_val = []
    fold = 0

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        dv, model, dtrain, dval = XGB_train(df_train, df_val, features, xgb_params, num_rounds)
        y_pred_train, y_pred_val = XGB_predict(dtrain, dval, model)

        y_train = df_train.transported.values
        y_val = df_val.transported.values
        scores_train.append(round(roc_auc_score(y_train, y_pred_train),4))
        scores_val.append(round(roc_auc_score(y_val, y_pred_val),4))

        # Calculates and prints metrics
        if print_kfold: print(f'Fold {fold} ROC_AUC train: {round(roc_auc_score(y_train, y_pred_train),4)} ROC_AUC val: {round(roc_auc_score(y_val, y_pred_val),4)}...')
        fold = fold + 1

    if print_kfold: print('Validation results:')
    if print_kfold: print('ROC_AUC_train: %.3f +- %.3f' % (np.mean(scores_train), np.std(scores_train))) 
    if print_kfold: print('ROC_AUC_val: %.3f +- %.3f' % (np.mean(scores_val), np.std(scores_val))) 

    #Final model
    if print_summary: print("Final model...")
    dv, model, dtrain, dtest = XGB_train(df_full_train, df_test, features, xgb_params, num_rounds)
    y_pred_train, y_pred_test = XGB_predict(dtrain, dtest, model)

    y_train = df_full_train.transported.values
    y_test = df_test.transported.values
   
    if print_summary: print('ROC_AUC train: ',round(roc_auc_score(y_train, y_pred_train),4), ' ROC_AUC test: ',round(roc_auc_score(y_test, y_pred_test),4))
      
    return [xgb_params['eta'], xgb_params['max_depth'], num_rounds,
          round(roc_auc_score(y_train, y_pred_train),4), round(roc_auc_score(y_test, y_pred_test),4)
          ], dv, model

# Loading and preparing the dataset
print ('Loading and preparing the dataset...')
print('Doing validation with ETA=%.2f' % eta)

df = pd.read_csv('train.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

df.cryosleep = pd.to_numeric(df.cryosleep, errors='coerce')
df.vip = pd.to_numeric(df.vip, errors='coerce')
df.transported = pd.to_numeric(df.transported, errors='coerce')

df['groupid'] = df['passengerid'].apply(lambda x: int(x[:4]))
gr = pd.DataFrame(df['groupid'].value_counts()).reset_index().rename({'index': 'groupid', 
                      'groupid': 'groupsize'}, axis=1)
df = df.merge(gr, on='groupid', how='left')
splitcabin = df['cabin'].str.split('/', expand=True).rename({0: 'deck', 1: 'num', 2: 'side'}, axis=1)
df = df.merge(splitcabin, left_index=True, right_index=True, how='left')
df = df.drop(columns = ['passengerid', 'cabin', 'name', 'groupid', 'num'])

df.cryosleep = df.cryosleep.fillna(df.cryosleep.median()).astype(int)
df.age = df.age.fillna(df.age.mean()).astype(int)
df.vip = df.vip.fillna(df.vip.median()).astype(int)
df.roomservice = df.roomservice.fillna(df.roomservice.mean()).astype(int)
df.foodcourt = df.foodcourt.fillna(df.foodcourt.mean()).astype(int)
df.shoppingmall = df.shoppingmall.fillna(df.shoppingmall.mean()).astype(int)
df.spa = df.spa.fillna(df.spa.mean()).astype(int)
df.vrdeck = df.vrdeck.fillna(df.spa.mean()).astype(int)
df.groupsize = df.groupsize.fillna(df.groupsize.mean()).astype(int)
df.homeplanet = df.homeplanet.fillna('earth')
df.destination = df.destination.fillna('trappist-1e')
df.deck = df.deck.fillna('f')
df.side = df.side.fillna('s')

# Spliting dataset
df_rest, df_tmp = train_test_split(df, test_size=0.1, random_state=1)
df_full_train, df_test = split_dataframe(df_tmp)

a, dv, model = XGB_calculate(features, xgb_params, xgb_num_rounds, True, True, 10)

# Saving the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to {output_file}')
