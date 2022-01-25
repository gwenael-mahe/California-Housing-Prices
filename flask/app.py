from flask import Flask, render_template, request
import numpy as np
import joblib

path = "../xgb_model.pkl"

model = joblib.load(path)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/result", methods = ['GET', 'POST'])
def result():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)