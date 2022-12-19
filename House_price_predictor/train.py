import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv('output/preprocessed_house_prices.csv')

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=31)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = np.log1p(df_full_train.price)
y_test = np.log1p(df_test.price)

del df_full_train["price"]
del df_test["price"]

dict_full_train = df_full_train.to_dict(orient="records")
dict_test = df_test.to_dict(orient="records")

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dict_full_train)
X_test = dv.transform(dict_test)


rf = RandomForestRegressor(n_estimators=150, max_depth=30, min_samples_leaf=1, random_state=31)
rf.fit(X_full_train, y_full_train)

y_pred = rf.predict(X_test)
print(f'test rmse:{mean_squared_error(y_test, y_pred, squared=False):.5f}')

with open ('model/house_price_model.bin', 'wb') as f:
    pickle.dump((dv,rf), f)

print("Model training completed")
