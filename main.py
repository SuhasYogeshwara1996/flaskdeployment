import numpy as np
from flask import Flask, request
import pickle

app = Flask(__name__)
model=pickle.load(open('Salary_Data','rb'))

@app.route('/predict',methods = ['POST'])

def predict():
    '''
    Predicting the Salary Package based on Number of Years Experiences
    '''

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output=round(prediction[0],2)
    return

if __name__ == '__main__':
    app.run(port=5000,debug=True)



