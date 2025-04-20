#Candidate Selection in Hiring: Application Evaluation with SVM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from faker import Faker
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

n_samples = 200

experience_years = np.round(np.random.uniform(0,10,n_samples),2)
technical_scores = np.round(np.random.uniform(0,100,n_samples),2)

labels = []

#rule for labeling
for experience,score in zip(experience_years,technical_scores):
    if experience < 2 and score < 60:
        labels.append(1) #not hired
    else:
        labels.append(0) #hired

#creating fake names with faker
candidate_names = [fake.name() for _ in range(n_samples)]

df = pd.DataFrame({
    'candidate_name': candidate_names,
    'experience_years': experience_years,
    'technical_score':technical_scores,
    'label':labels
})

#print(df.head())

#features and target
X = df[['experience_years','technical_score']]
y = df['label']

#training & test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#scaling the data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#SVM model training
svc_model = SVC(kernel="linear", class_weight="balanced") #due to imbalanced data the prediction was not accurate so I add class_weight
svc_model.fit(X_train_scaled,y_train)

#visualization
def plot_svm_decision_boundary(model, X, y):
  
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1],
                color='green', label='Hired', edgecolors='k', s=60)
    
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],
                color='red', label='Not Hired', edgecolors='k', s=60)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.7,
               linestyles=['--', '-', '--'])

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k')

    plt.title("SVM Classification with Support Vectors")
    plt.xlabel("Experience (scaled)")
    plt.ylabel("Technical Score (scaled)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.show()


plot_svm_decision_boundary(svc_model, X_train_scaled, y_train)

#model evaluation
y_pred = svc_model.predict(X_test_scaled)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#new candidate prediction from user input
def predict_new_candidate():
    try:
        experience = float(input("Enter candidate's years of experience (e.g. 3.5): "))

        score = float(input("Enter candidate's technical test score (0–100): "))

        input_df = pd.DataFrame([[experience, score]], columns=["experience_years", "technical_score"])

        input_scaled = scaler.transform(input_df)

        prediction = svc_model.predict(input_scaled)

        result_text = "Not Hired" if prediction[0] == 1 else "Hired!"
        print(f"\nPrediction Result: {result_text}")

    except ValueError:
        print("Please enter valid numeric values for experience and score.")

predict_new_candidate()
