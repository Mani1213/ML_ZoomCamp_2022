import pickle
from sys import is_finalizing

input_model = "model1.bin"

with open(input_model,"rb") as f_in:
    model = pickle.load(f_in)

input_dict_vect = "dv.bin"
with open(input_dict_vect,"rb") as f_in:
    dv = pickle.load(f_in)

def predict(data):
    X = dv.transform(data)
    proba = model.predict_proba(X)
    return proba



if __name__=="__main__":
    data = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
    proba = predict(data)
    print(proba[0][1])
