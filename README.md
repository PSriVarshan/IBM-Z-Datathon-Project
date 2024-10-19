# IBM-Z-Datathon-Project
# Alzheimer's Detection Using Biomarkers

## Dependencies

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

## Data Preprocessing

```python
# Load the dataset
file_path = 'BIOM.csv'
data = pd.read_csv(file_path)

# Select relevant columns
selected_columns = ["AGE", "PTGENDER", "FDG", "PIB", "MMSE", "PTMARRY", "APOE4", "DX"]
data_selected = data[selected_columns]

# Clean and preprocess data
data_cleaned = data_selected.replace("NA", pd.NA)

# Fill missing values
data_cleaned['AGE'] = data_cleaned['AGE'].fillna(data_cleaned['AGE'].mean())
data_cleaned['FDG'] = data_cleaned['FDG'].fillna(data_cleaned['FDG'].mean())
data_cleaned['PIB'] = data_cleaned['PIB'].fillna(data_cleaned['PIB'].mean())
data_cleaned['MMSE'] = data_cleaned['MMSE'].fillna(data_cleaned['MMSE'].mean())
data_cleaned['APOE4'] = data_cleaned['APOE4'].fillna(data_cleaned['APOE4'].mode()[0])
data_cleaned['PTGENDER'] = data_cleaned['PTGENDER'].fillna(data_cleaned['PTGENDER'].mode()[0])
data_cleaned['PTMARRY'] = data_cleaned['PTMARRY'].fillna(data_cleaned['PTMARRY'].mode()[0])

# Convert categorical variables to numerical
data_cleaned['PTGENDER'] = data_cleaned['PTGENDER'].map({'Male': 0, 'Female': 1})
data_cleaned['PTMARRY'] = data_cleaned['PTMARRY'].map({
    'Married': 0, 'Divorced': 1, 'Widowed': 2, 'Never married': 3, 'Unknown': 4
})

# Map DX column to binary classification
dx_mapping = {
    'NL': 0, 'NL to MCI': 0, 'MCI to NL': 0, 'MCI': 0,
    'Dementia': 1, 'MCI to Dementia': 1, 'NL to Dementia': 1
}
data_cleaned['DX'] = data_cleaned['DX'].map(dx_mapping).dropna()

# Separate features and target variable
X = data_cleaned.drop(columns=['DX'])
y = data_cleaned['DX']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
```

## Model Training

```python
# Define base learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='linear', probability=True))
]

# Create the stacking ensemble
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())

# Train the stacking model
stacking_model.fit(X_train, y_train)
```

## Model Evaluation

```python
# Make predictions
y_pred = stacking_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the stacking model: {accuracy * 100:.2f}%")
```

## Making Predictions

```python
def get_user_input():
    age = float(input("Enter Age: "))
    ptgender = int(input("Enter Gender (0 = Male, 1 = Female): "))
    fdg = float(input("Enter FDG value: "))
    pib = float(input("Enter PIB value: "))
    mmse = float(input("Enter MMSE score: "))
    ptmarry = int(input("Enter Marital Status (0=Married, 1=Divorced, 2=Widowed, 3=Never married, 4=Unknown): "))
    apoe4 = int(input("Enter APOE4 allele count (0, 1, or 2): "))

    input_data = np.array([[age, ptgender, fdg, pib, mmse, ptmarry, apoe4]])
    return input_data

# Get user input
user_input = get_user_input()

# Make prediction
prediction = stacking_model.predict(user_input)
probabilities = stacking_model.predict_proba(user_input)

# Display results
print(f"Predicted class (0 = Normal, 1 = Dementia): {prediction[0]}")
print(f"Probability for each class: {probabilities[0]}")
```

## Screenshot

![Alzheimer's Detection Model Output](https://example.com/path/to/your/screenshot.png)
