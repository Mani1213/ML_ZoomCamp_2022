from flask import Flask
from flask import request
from flask import jsonify
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load Model
input_model = "model1.bin"
with open(input_model,"rb") as f_in:
    model = pickle.load(f_in)

# Load Dict Vectorizer
input_dict_vect = "dv.bin"
with open(input_dict_vect,"rb") as f_in:
    dv = pickle.load(f_in)

app = Flask("card")

@app.route("/ques6",methods = ["POST"])
def predict():
    data = request.get_json()
    X = dv.transform([data])
    proba = model.predict_proba(X)[0,1]
    card  = proba>=0.5
    result = {"proba":float(proba),
              "get_card":bool(card)
              }
    
    return jsonify(result)


if __name__ =="__main__":
    app.run(debug=True, host="0.0.0.0",port=9696)

