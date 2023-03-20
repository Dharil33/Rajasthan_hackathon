from flask import Flask, render_template, request
import pickle
import pandas as pd

# initialize the flask app
app = Flask(__name__, template_folder="templates")
model = pickle.load(open("Fraud_detection.pkl", "rb"))


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


# Now predict function
@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on the HTML page
    step = request.form['step']
    type = request.form['type']
    amount = request.form['amount']
    oldbalanceOrg = request.form['oldbalanceOrg']
    newbalanceOrig = request.form['newbalanceOrig']
    oldbalanceDest = request.form['oldbalanceDest']
    newbalanceDest = request.form['newbalanceDest']

    step = int(step)
    type = int(type)
    amount = int(amount)
    oldbalanceOrg = int(oldbalanceOrg)
    newbalanceOrig = int(newbalanceOrig)
    oldbalanceDest = float(oldbalanceDest)
    newbalanceDest = float(newbalanceDest)

    row_df = pd.DataFrame([pd.Series([step,type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest])])
    print(row_df)
    prediction = model.predict_proba(row_df)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    output = str(float(output) * 100) + '%'
    return render_template('index.html', pred=f'Probability of having fraud is {output}')


if __name__ == '__main__':
    app.run(debug=True)