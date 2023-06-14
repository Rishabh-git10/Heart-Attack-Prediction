from django.shortcuts import render
from django.http import HttpResponse
from sklearn.preprocessing import RobustScaler
import pandas as pd
import pickle


# Create your views here.
def predict_and_recommend(request):
    if request.method == 'POST':
        scaler = RobustScaler()
        model = pickle.load(open("model.pkl", "rb"))
    
        # Retrieve the form data
        age = float(request.POST['age'])
        sex = float(request.POST['sex'])
        chest_pain = float(request.POST['chest-pain'])
        resting_bp = float(request.POST['blood-pressure'])
        cholesterol = float(request.POST['cholesterol'])
        fasting_bs = float(request.POST['blood-sugar'])
        resting_ecg = float(request.POST['ecg'])
        max_heart_rate = float(request.POST['max-heart-rate'])
        exercise_angina = float(request.POST['exercise-angina'])
        oldpeak = float(request.POST['st-depression'])
        st_slope = float(request.POST['st-slope'])
        
        
        # Create a dictionary of input features
        input_data = {'age': [age], 'sex': [sex], 'chest pain type': [chest_pain], 'resting bp s': [resting_bp],
                      'cholesterol': [cholesterol], 'fasting blood sugar': [fasting_bs], 'resting ecg': [resting_ecg],
                      'max heart rate': [max_heart_rate], 'exercise angina': [exercise_angina], 'oldpeak': [oldpeak],
                      'ST slope': [st_slope]}
        
        # Convert the input features into a dataframe
        input_df = pd.DataFrame(data=input_data)
        
        # Scale the input data
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
        
        # print("Your probability of a heart attack is {:.2f}%.".format(probability))
        # print("To reduce your probability of a heart attack, consider the following recommendations: {}".format(', '.join(recommendations)))
        
        return render(request, "result.html", {"probability": probability, "recommendations": recommendations})
    
    return render(request, "index.html")