# Employee Attrition Prediction System

## Project Overview

The **Employee Attrition Prediction System** is a machine learning-based web application designed to predict and analyze employee attrition.

The project uses the IBM HR Analytics Employee Attrition dataset to identify patterns associated with employee turnover. It combines machine learning, web development, database management, and business intelligence to help HR professionals understand attrition trends and support data-driven decision-making.

## Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn, Random Forest
- **Web Framework:** Flask
- **Database:** PostgreSQL
- **Database ORM:** SQLAlchemy
- **Authentication:** Flask-Bcrypt and Session Management
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Power BI
- **Environment Management:** Python dotenv

## Dataset

- **Dataset:** IBM HR Analytics Employee Attrition
- **Source:** IBM HR Analytics Employee Attrition dataset
- **Purpose:** Analyze employee characteristics and identify factors associated with employee attrition.

## Key Features

- Employee attrition prediction using a Random Forest classifier.
- Flask-based web application.
- User authentication and session management.
- Role-based access for HR and CEO users.
- PostgreSQL integration for employee information and prediction records.
- Employee data import and database management.
- Prediction results exported for Power BI visualization.
- Interactive dashboard for analyzing employee attrition trends.

## Project Workflow

1. Load and preprocess the employee dataset.
2. Perform feature engineering and encode categorical variables.
3. Train the Random Forest classification model.
4. Evaluate model performance.
5. Save the trained model using Pickle.
6. Integrate the model into the Flask application.
7. Store employee information and prediction records in PostgreSQL.
8. Export prediction results for Power BI analysis.

## Project Structure

```text
Employee Attrition Project/
│
├── app.py
├── train_model.py
├── model.pkl
├── employee_performance_attrition.csv.xlsx
│
├── templates/
│
├── employee_data.db
├── latest_predictions.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```



## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Employee Attrition Project"
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file and configure the required database connection and application secret key.

Never upload your `.env` file or expose database credentials.

### 5. Run the application

```bash
python app.py
```
## Project Screenshots

### LOGIN Page(ROLE-BASED)
![login Page](Screenshots/login page.png)


### HR Page
![Home Page for HR](Screenshots/HR accessed page.png)

### HR DB Page
![HR DATABASE VIEW Page](Screenshots/DB page view for HR.png)


### ceo Page
![Home Page for CEO](Screenshots/CEO accessed page.png)


### Attrition Prediction
![Prediction Result](Screenshots/Prediction.png)



## Dashboard

Power BI is used to visualize employee attrition patterns and analyze prediction results.

## Future Improvements

- Improve model performance through hyperparameter optimization.
- Add more advanced employee analytics.
- Improve dashboard reporting and visualization.
- Enhance application security and scalability.

### Power BI Dashboard
![Dashboard](Screenshots/DASHBOARD-1.png)
![Dashboard](Screenshots/DASHBOARD-2.png)



## Disclaimer

This project is developed for educational and analytical purposes. Machine learning predictions should support, not replace, human decision-making.

## Author

Developed as an Employee Attrition Prediction and Analysis project using Python, Machine Learning, Flask, PostgreSQL, and Power BI.
