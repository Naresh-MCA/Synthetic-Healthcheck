from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler once
with open('synthetic_health.pkl', 'rb') as f:
    model = pickle.load(f)

#with open('scaler.pkl', 'rb') as f:
    #scaler = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        try:
            Age = float(request.form['Age'])
            BMI = float(request.form['BMI'])
            Exercise_Frequency = float(request.form['Exercise_Frequency'])
            Diet_Quality = float(request.form['Diet_Quality'])
            Sleep_Hours = float(request.form['Sleep_Hours'])
            Smoking_Status = float(request.form['Smoking_Status'])
            Alcohol_Consumption = float(request.form['Alcohol_Consumption'])

            data = np.array([[Age, BMI, Exercise_Frequency,
                              Diet_Quality, Sleep_Hours,
                              Smoking_Status, Alcohol_Consumption]])

            # 🔥 SCALE BEFORE PREDICTION
            #scaled_data = scaler.transform(data)

            prediction = round(model.predict(data)[0], 2)

        except Exception as e:
            prediction = str(e)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
