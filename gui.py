import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')

# Load the Titanic dataset

titanic_df = pd.read_csv("C:\\Users\\mudiy\\OneDrive\\Desktop\\ml\\train.csv")

# Define the features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
target = 'Survived'

# Preprocess the data
titanic_df['Sex'] = titanic_df['Sex'].map({'female': 1, 'male': 0})
titanic_df = titanic_df.dropna()

# Train the model
X = titanic_df[features]
y = titanic_df[target]
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X, y)

# Define the GUI
root = tk.Tk()
root.title('Titanic Survival Predictor')

# Define the widgets
class_label = tk.Label(root, text='Passenger Class (1-3):')
class_entry = tk.Entry(root)

sex_label = tk.Label(root, text='Passenger Sex (0=male, 1=female):')
sex_entry = tk.Entry(root)

age_label = tk.Label(root, text='Passenger Age:')
age_entry = tk.Entry(root)

sibsp_label = tk.Label(root, text='Number of Siblings/Spouses:')
sibsp_entry = tk.Entry(root)

parch_label = tk.Label(root, text='Number of Parents/Children:')
parch_entry = tk.Entry(root)

fare_label = tk.Label(root, text='Passenger Fare:')
fare_entry = tk.Entry(root)

output_label = tk.Label(root, text='')

# Define the predict function
def predict_survival():
    passenger = [[int(class_entry.get()), int(sex_entry.get()), float(age_entry.get()),
                  int(sibsp_entry.get()), int(parch_entry.get()), float(fare_entry.get())]]
    prediction = model.predict(passenger)[0]
    if prediction == 0:
        output_label.config(text='The passenger did not survive.')
    else:
        output_label.config(text='The passenger survived!')

# Define the submit button
submit_button = tk.Button(root, text='Predict Survival', command=predict_survival)

# Pack the widgets
class_label.pack()
class_entry.pack()

sex_label.pack()
sex_entry.pack()

age_label.pack()
age_entry.pack()

sibsp_label.pack()
sibsp_entry.pack()

parch_label.pack()
parch_entry.pack()

fare_label.pack()
fare_entry.pack()

submit_button.pack()

output_label.pack()

root.mainloop()