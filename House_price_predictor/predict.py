from flask import Flask, render_template, request
import numpy as np
import pickle


with open('model/house_price_model.bin', 'rb') as f:
    dv, model = pickle.load(f)


def price_prediction(house, dv, model):
    """Function for applying the house price prediction model"""
    X = dv.transform(house)
    y_pred = np.expm1(model.predict(X))
    return int(np.round(y_pred, 0))


app = Flask(__name__)


@app.route("/")
def my_form_home():
    return render_template('home.html')


@app.route("/", methods=['POST', 'GET'])
def predict():

    try:
        m2_data = float(request.form['m2'])
    except ValueError:
        m2_data = 20.0


    house = {
        'house_type': request.form['house_type'],
        'house_type_2': request.form['house_type_2'],
        'rooms': request.form['rooms'],
        'm2': m2_data,
        'elevator': request.form['elevator'],
        'garage': request.form['garage'],
        'neighborhood': request.form['district'].replace(" ", "").split(",")[1],
        'district': request.form['district'].replace(" ", "").split(",")[0]
    }
    pred = price_prediction(house, dv, model)

    return render_template('prediction.html', data=pred, data2=house)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
