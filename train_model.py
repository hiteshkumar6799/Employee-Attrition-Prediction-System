import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_excel(
    r"E:\Employee Attrition Project\employee_performance_attrition.csv.xlsx"
)



df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
df["HighSalary"] = df["HighSalary"].map({"Yes": 1, "No": 0})
df["LongTenure"] = df["LongTenure"].map({"Yes": 1, "No": 0})


df = pd.get_dummies(df, columns=["Department", "JobRole"])


df["PerformanceIndex"] = (
    0.4 * df["JobSatisfaction"] + 0.3 * df["WorkLifeBalance"] + 0.3 * df["PerformanceRating"]
)


features = [
    "JobSatisfaction", "WorkLifeBalance", "PerformanceRating",
    "Salary", "YearsAtCompany", "OverTime", "PerformanceIndex",
    "HighSalary", "LongTenure"
] + [col for col in df.columns if col.startswith("Department_") or col.startswith("JobRole_")]

X = df[features]
y = df["Attrition"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 2))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


model_info = {
    "model": model,
    "columns": X.columns.tolist()
}
pickle.dump(model_info, open("model.pkl", "wb"))

print("\n✅ Model trained and saved successfully with", len(features), "features.")
