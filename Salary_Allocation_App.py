from flask import Flask, request, render_template
import joblib
import numpy as np


# Initialize Flask app
app = Flask(__name__)
# Load the saved model
model = joblib.load('salary_allocation_model.pkl')
# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        salary = float(request.form['salary'])  # Salary (SAR)
        monthly_debt = float(request.form['monthly_debt'])  # Monthly Debt (SAR)
        elementary_expenses = float(request.form['elementary_expenses'])  # Elementary Expenses (SAR)
        number_of_children = int(request.form['number_of_children'])  # Number of Children

        # Encode 'state_Single'
        state = request.form['state']
        state_Single = 1 if state == 'single' else 0

        # Encode 'sex_Male'
        sex = request.form['sex']
        sex_Male = 1 if sex == 'male' else 0

        # Encode 'goal_Savings'
        goal = request.form['goal']
        goal_Savings = 1 if goal == 'savings' else 0

        # Encode 'goal_Spending'
        goal_Spending = 1 if goal == 'spending' else 0

        # Encode employment status
        employment_status = request.form['employment_status']
        employment_status_Student = 1 if employment_status == 'student' else 0
        employment_status_Unemployed = 1 if employment_status == 'unemployed' else 0

        # Encode age group
        age = int(request.form['age'])
        if 18 <= age <= 24:
            age_group_encoded = 0
        elif 25 <= age <= 34:
            age_group_encoded = 1
        elif 35 <= age <= 44:
            age_group_encoded = 2
        elif 45 <= age <= 54:
            age_group_encoded = 3
        else:
            age_group_encoded = 4

        # Calculate derived features
        debt_to_income_ratio = monthly_debt / salary if salary > 0 else 0
        savings_ratio = elementary_expenses / salary if salary > 0 else 0

        # Combine all features into a single input array
        input_features = np.array([[salary, monthly_debt, elementary_expenses, number_of_children,
                                    state_Single, sex_Male, goal_Savings, goal_Spending, employment_status_Student,
                                    employment_status_Unemployed, age_group_encoded,
                                    debt_to_income_ratio, savings_ratio]])

        # Make a prediction
        prediction = model.predict(input_features)

        # Budgeting rule based on the prediction
        rules = {
              0: 
              "rule: 50/30/20\nExplanation: 50% of your income should go to expenses, 30% to saving, and 20% to investment. This is a balanced approach for managing your finances."
              ,
              1: 
                     "rule: 70/20/10\nExplanation: 70% of your income should go to expenses, 20% to saving, and 10% to investment. This rule is ideal for higher earners or those with minimal debt."
              ,
              2: 
                     "rule: 60/20/20\nExplanation: 60% of your income should go to expenses, 20% to saving, and 20% to investment. This is suitable for individuals with higher living costs or larger families."
              
        }

        # Get the rule for the predicted class
        result = rules.get(prediction[0], "Unknown result")

        return render_template('index.html', prediction_text=f"Model Prediction: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)