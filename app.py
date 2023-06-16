from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        age = float(request.form.get('age'))
        sex = float(request.form.get('sex'))
        chest_pain = float(request.form.get('chest-pain'))
        resting_bp = float(request.form.get('blood-pressure'))
        cholesterol = float(request.form.get('cholesterol'))
        fasting_bs = float(request.form.get('blood-sugar'))
        resting_ecg = float(request.form.get('ecg'))
        max_heart_rate = float(request.form.get('max-heart-rate'))
        exercise_angina = float(request.form.get('exercise-angina'))
        oldpeak = float(request.form.get('st-depression'))
        st_slope = float(request.form.get('st-slope'))

        # Create a dictionary of input features
        input_data = {'age': [age], 'sex': [sex], 'chest pain type': [chest_pain], 'resting bp s': [resting_bp],
                      'cholesterol': [cholesterol], 'fasting blood sugar': [fasting_bs], 'resting ecg': [resting_ecg],
                      'max heart rate': [max_heart_rate], 'exercise angina': [exercise_angina], 'oldpeak': [oldpeak],
                      'ST slope': [st_slope]}

        # Convert the input features into a dataframe
        input_df = pd.DataFrame(data=input_data)

        # Scale the input data (assuming you have a scaler object)
        scaler = RobustScaler()
        scaler.fit(input_df)
        input_scaled = scaler.transform(input_df)

        # Use the trained model to predict the likelihood of a heart attack
        probability = model.predict_proba(input_scaled)[0][1] * 100

        # Make recommendations based on the probability of a heart attack
        recommendations = []
        if probability < 10:
            recommendations.append("increase your exercise level")
            recommendations.append("reduce your cholesterol intake")
        elif probability < 30:
            recommendations.append("maintain a healthy weight")
            recommendations.append("reduce your sodium intake")
        elif probability < 50:
            recommendations.append("quit smoking")
            recommendations.append("reduce your stress levels")
        else:
            recommendations.append("consult with your doctor for personalized recommendations.")

        return render_template('result.html', probability=probability, recommendations=recommendations)

    # Render the initial form
    return render_template('index.html')

@app.route('/aboutus.html')
def about_us():
    return render_template('aboutus.html')

@app.route('/contactus.html')
def contact_us():
    return render_template('contactus.html')

if __name__ == '__main__':
    app.run(debug=True)
