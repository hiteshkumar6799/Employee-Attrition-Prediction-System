
import pandas as pd
import psycopg2
import os

from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# PostgreSQL configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "employee_attrition")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")


# Check database password
if not DB_PASS:
    raise RuntimeError("DB_PASSWORD is missing from .env")


# Connect to PostgreSQL
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
    print("Successfully connected to the database.")

except Exception as e:
    print("Error connecting to database:", e)
    raise SystemExit(1)


# Fetch employee prediction records
query = """
SELECT
    ei.id AS "ID",
    ei.employee_id AS "Employee ID",
    e."Department" AS "Department",
    e."JobRole" AS "Job Role",
    ei.jobsatisfaction AS "Job Satisfaction",
    ei.worklifebalance AS "Work Life Balance",
    ei.performancerating AS "Performance Rating",
    ei.salary AS "Salary",
    ei.yearsatcompany AS "Years at Company",
    ei.overtime AS "Overtime",
    ei.overtime_month AS "Overtime Month",
    ei.overtime_count AS "Overtime Count",
    ei.highsalary AS "High Salary",
    ei.longtenure AS "Long Tenure",
    ei.performance_index AS "Performance Index",
    ei.prediction AS "Prediction",
    ei.probability AS "Probability"
FROM employee_inputs ei
JOIN employees e ON ei.employee_id = e."EmployeeID"
ORDER BY ei.employee_id;
"""


try:
    df = pd.read_sql(query, conn)
    print("Data fetched successfully.")

except Exception as e:
    print("Error fetching data:", e)
    conn.close()
    raise SystemExit(1)


# Convert binary columns into Yes/No
bool_columns = ["Overtime", "High Salary", "Long Tenure"]

for col in bool_columns:
    if col in df.columns:
        df[col] = df[col].map({
            1: "Yes",
            0: "No",
            True: "Yes",
            False: "No"
        }).fillna(df[col])


# Handle missing overtime month
if "Overtime Month" in df.columns:
    df["Overtime Month"] = df["Overtime Month"].replace(
        ["", None], "N/A"
    )


# Export CSV inside project folder
csv_filename = BASE_DIR / "latest_predictions.csv"

try:
    df.to_csv(csv_filename, index=False)
    print(f"Data exported to {csv_filename}")

except Exception as e:
    print("Error exporting CSV:", e)

finally:
    conn.close()
    print("Database connection closed.")
