from flask import Flask, render_template, request,jsonify
import pickle
import pandas as pd
import numpy as np
# from src.exception import CustomException
# import sys
import joblib
from lime.lime_tabular import LimeTabularExplainer

# initialize the flask app
app = Flask(__name__, template_folder="templates")
model = pickle.load(open("final_model_prediction.pkl", "rb"))
# scalar = pickle.load(open("StandardScaler.pkl","rb"))
scalar = joblib.load("StandardScaler.pkl")

X_train = pd.read_csv("D:\STUDY\Hackathon\PROJECT\Rajasthan_hackathon\src\Data\train.csv")
X_test = pd.read_csv("D:\STUDY\Hackathon\PROJECT\Rajasthan_hackathon\src\Data\test.csv")
class_names = ['Fraud', 'NotFraud']
feature_names = list(X_train.columns)
explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names, class_names = class_names, mode = 'classification')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.fit_transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

# Now predict function
@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.fit_transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    if output == 0:
        result = "Current Transaction is not fraud!!"
    else:
        result = "Current Transaction is fraud!!"

    exp = explainer.explain_instance(X_test.iloc[100], model.predict_proba)
    exp = exp.as_html()
    
    return render_template("index.html",prediction_text=result, exp=exp)

if __name__ == '__main__':
    app.run(debug=True)