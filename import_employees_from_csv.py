import pandas as pd
from app import db, Employee, app


df = pd.read_excel(
    r"E:\Python-Data-Analysis-Projects-Package\Python Data Analysis Projects Package\Project 7\employee_performance_attrition.csv.xlsx"
)


print("CSV Columns:", df.columns.tolist())


employees_df = df[["EmployeeID", "Department", "JobRole"]].dropna()


employees_df["EmployeeID"] = employees_df["EmployeeID"].astype(int)

with app.app_context():
    inserted = 0
    for _, row in employees_df.iterrows():
        emp_exists = Employee.query.filter_by(EmployeeID=row["EmployeeID"]).first()
        if not emp_exists:
            emp = Employee(
                EmployeeID=row["EmployeeID"],
                Department=row["Department"],
                JobRole=row["JobRole"]
            )
            db.session.add(emp)
            inserted += 1

    db.session.commit()

print(f"✅ Successfully inserted {inserted} employees into PostgreSQL!")
