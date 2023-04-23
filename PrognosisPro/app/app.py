from flask import Flask, render_template, request

app = Flask(__name__)

diabetes_model = load_diabetes_model()
heart_disease_model = load_heart_disease_model()
breast_cancer_model = load_breast_cancer_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    dataset = request.form['dataset']
    glucose = request.form['glucose']
    blood_pressure = request.form['blood_pressure']
    skin_thickness = request.form['skin_thickness']
    insulin = request.form['insulin']
    bmi = request.form['bmi']
    
    
    if dataset == 'diabetes':
        prediction = diabetes_model.predict([[glucose, blood_pressure, skin_thickness, insulin, bmi]])[0]
    elif dataset == 'heart_disease':
        prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])[0]
    elif dataset == 'breast_cancer':
        prediction = breast_cancer_model.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]])[0]
    
    
    return render_template('results.html', prediction=prediction)
