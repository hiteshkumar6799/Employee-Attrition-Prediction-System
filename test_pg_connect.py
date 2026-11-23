import psycopg2

try:
    conn = psycopg2.connect(
        dbname="employee_attrition",
        user="postgres",
        password="your_postgres_password",  
        host="localhost"
    )
    print("✅ PostgreSQL connection successful!")
    conn.close()
except Exception as e:
    print("❌ Connection failed:", e)
