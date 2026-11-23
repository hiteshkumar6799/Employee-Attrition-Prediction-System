import pandas as pd

df = pd.read_excel("employee_performance_attrition.csv.xlsx")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())