# Rajasthan_hackathon

# AI based Financial Fraud Prediction and Detection

## Introduction
•Financial fraud detection refers to the use of technology and analytical tools to identify and prevent fraudulent activities within financial systems. Financial fraud can take many forms, including credit card fraud, identity theft, money laundering, and insider trading.
•The process of financial fraud detection involves analyzing large amounts of data to identify patterns and anomalies that may indicate fraudulent activity. This is typically done using machine learning algorithms that are trained to recognize specific types of fraudulent behavior based on historical data.

## Input Variables

1. type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
2. amount - amount of the transaction in local currency.
3. oldbalanceOrg - initial balance before the transaction
4. newbalanceOrig - new balance after the transaction
5. oldbalanceDest - initial balance recipient before the transaction. 
6. newbalanceDest - new balance recipient after the transaction. 
7. isFraud - This is the transactions made by the fraudulent agents inside the simulation.

# Approach

1. Data Exploration 
2. Model Selection:
    Tested all base models to check the base accuracy
3. Hyperparameter Tuning
    Performed hyperparameter tuning using RandomsearchCV
4. Pickle File
    Selected model as per best accuracy and created pickle file using Pickle .
5. Webpage & deployment
     1. Created a web form that takes all the necessary inputs from user and shows output.
        The output is accompanied with the output of the Explainable AI
     2. We had deployed project on AWS EC2 Instance


## File Structure

Rajasthan_hackathon
|
|
components
    |
    |
    _init_.py
    data_ingestion.py
    data_transformation.py
    train_mlmodels.py
|
|
src
    |
    |
    Data
        |
        |
        Final_Dataset.csv
        Type_LabelMapping.csv
        test.csv
        train.csv
    |
    |
    Notebook
        |
        |
        EDA.ipynb
        Training_&_Prediction.ipynb
        XAI.ipynb
    _init_.py
    exception.py
    logger.py
    utils.py
|
|
static
    |
    |
    blue.jpg
    style.css
|
|
templates
    |
    |
    index.html
    result.html
|
|
Lime_XAI.pkl
StandardScaler.pkl
app.py
final_model_prediction.pkl
requiremnts.txt
setup.py

## Deployment Link

AWS Link: http://ec2-34-205-54-116.compute-1.amazonaws.com:8080/

## Installation
The Code is written in Python 3.8.0. To install the required packages and libraries, run this command after cloning the repository.

'''rb
#pip install -r requirements.txt
'''

'''rb
Run app.py file
'''

## Technology Used:
1. Github Account [www.Github.com]
2. Visual Studio Code
3. Jupyter Notebook
4. Anaconda
5. Coding Language : Python
6. Markup Language : HTML and CSS 
7. AWS EC2 Instance 
