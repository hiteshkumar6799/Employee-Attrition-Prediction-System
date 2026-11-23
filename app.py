
import os
import pickle
import subprocess
import sys
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY")

if not app.config["SECRET_KEY"]:
    raise RuntimeError("FLASK_SECRET_KEY is missing from .env")

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get(
        "COOKIE_SECURE", "false"
    ).lower() == "true",
)

bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)


# ============================================================
# PostgreSQL Configuration
# ============================================================

DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME", "employee_attrition")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")

if not DB_PASSWORD:
    raise RuntimeError("DB_PASSWORD is missing from .env")

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"postgresql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ============================================================
# Database Models
# ============================================================

class Employee(db.Model):
    __tablename__ = "employees"

    EmployeeID = db.Column(db.Integer, primary_key=True)
    Department = db.Column(db.String(50), nullable=False)
    JobRole = db.Column(db.String(50), nullable=False)


class EmployeeInput(db.Model):
    __tablename__ = "employee_inputs"

    id = db.Column(db.Integer, primary_key=True)

    employee_id = db.Column(
        db.Integer,
        db.ForeignKey("employees.EmployeeID"),
        nullable=False
    )

    jobsatisfaction = db.Column(db.Float)
    worklifebalance = db.Column(db.Float)
    performancerating = db.Column(db.Float)
    salary = db.Column(db.Float)
    yearsatcompany = db.Column(db.Float)

    overtime = db.Column(db.Integer)
    overtime_month = db.Column(db.String(20))
    overtime_count = db.Column(db.Integer)

    highsalary = db.Column(db.Integer)
    longtenure = db.Column(db.Integer)

    performance_index = db.Column(db.Float)

    prediction = db.Column(db.String(20))
    probability = db.Column(db.Float)


# ============================================================
# Authentication Configuration
# ============================================================

required_auth_vars = [
    "HR_USERNAME",
    "HR_PASSWORD_HASH",
    "CEO_USERNAME",
    "CEO_PASSWORD_HASH"
]

missing_auth_vars = [
    key for key in required_auth_vars
    if not os.environ.get(key)
]

if missing_auth_vars:
    raise RuntimeError(
        "Missing authentication environment variables: "
        + ", ".join(missing_auth_vars)
    )

users = {
    os.environ["HR_USERNAME"]: os.environ["HR_PASSWORD_HASH"],
    os.environ["CEO_USERNAME"]: os.environ["CEO_PASSWORD_HASH"],
}


def check_user(username, password):
    password_hash = users.get(username)

    if not password_hash:
        return False

    try:
        return bcrypt.check_password_hash(password_hash, password)
    except (TypeError, ValueError):
        return False


def is_logged_in():
    return "user" in session


def is_ceo():
    return session.get("user") == os.environ["CEO_USERNAME"]


# ============================================================
# Load Trained Machine Learning Model
# ============================================================

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model file not found: {MODEL_PATH}"
    )

# Only load pickle files that you created and trust.
with open(MODEL_PATH, "rb") as model_file:
    model_info = pickle.load(model_file)

model = model_info["model"]
MODEL_FEATURES = model_info["columns"]


# ============================================================
# Input Validation Helpers
# ============================================================

def parse_float(form, field, label, minimum=None, maximum=None):

    raw_value = form.get(field, "").strip()

    if raw_value == "":
        raise ValueError(f"{label} is required.")

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a valid number.")

    if not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{label} cannot be less than {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{label} cannot exceed {maximum}."
        )

    return value


def parse_int(
    form,
    field,
    label,
    default=None,
    minimum=None,
    maximum=None
):

    raw_value = form.get(field, "")

    if raw_value == "" and default is not None:
        value = default
    else:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{label} must be a whole number."
            )

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{label} cannot be less than {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{label} cannot exceed {maximum}."
        )

    return value


# ============================================================
# Collect Employee Prediction Inputs
# ============================================================

def get_prediction_inputs(form):

    data = {
        "jobsatisfaction": parse_float(
            form, "jobsatisfaction", "Job Satisfaction", 0, 5
        ),

        "worklifebalance": parse_float(
            form, "worklifebalance", "Work Life Balance", 0, 5
        ),

        "performancerating": parse_float(
            form, "performancerating", "Performance Rating", 0, 5
        ),

        "salary": parse_float(
            form, "salary", "Salary", 0
        ),

        "yearsatcompany": parse_float(
            form, "yearsatcompany", "Years at Company", 0
        ),
    }

    overtime = parse_int(
        form, "overtime", "Overtime",
        default=0, minimum=0, maximum=1
    )

    overtime_month = form.get(
        "overtime_month", ""
    ).strip()

    overtime_count = parse_int(
        form, "overtime_count", "Overtime Count",
        default=0, minimum=0
    )

    highsalary = parse_int(
        form, "highsalary", "High Salary",
        default=0, minimum=0, maximum=1
    )

    longtenure = parse_int(
        form, "longtenure", "Long Tenure",
        default=0, minimum=0, maximum=1
    )

    return (
        data,
        overtime,
        overtime_month,
        overtime_count,
        highsalary,
        longtenure
    )


# ============================================================
# Feature Engineering
# ============================================================

def build_features(
    employee,
    data,
    overtime,
    highsalary,
    longtenure
):

    # Calculate Performance Index
    performance_index = round(
        0.4 * data["jobsatisfaction"]
        + 0.3 * data["worklifebalance"]
        + 0.3 * data["performancerating"],
        2
    )

    # Initialize all trained features with zero
    feature_dict = {
        column: 0 for column in MODEL_FEATURES
    }

    # Map employee inputs to model feature names
    mapping = {
        "JobSatisfaction": data["jobsatisfaction"],
        "WorkLifeBalance": data["worklifebalance"],
        "PerformanceRating": data["performancerating"],
        "Salary": data["salary"],
        "YearsAtCompany": data["yearsatcompany"],
        "OverTime": overtime,
        "PerformanceIndex": performance_index,
        "HighSalary": highsalary,
        "LongTenure": longtenure,
    }

    for key, value in mapping.items():

        if key in feature_dict:
            feature_dict[key] = value

    # Department and JobRole encoding
    department_column = f"Department_{employee.Department}"
    role_column = f"JobRole_{employee.JobRole}"

    # Validate department category
    if department_column not in feature_dict:
        raise ValueError(
            f"Unknown department: {employee.Department}"
        )

    # Validate job role category
    if role_column not in feature_dict:
        raise ValueError(
            f"Unknown job role: {employee.JobRole}"
        )

    feature_dict[department_column] = 1
    feature_dict[role_column] = 1

    # Create DataFrame with the exact training feature order
    features = pd.DataFrame(
        [feature_dict],
        columns=MODEL_FEATURES
    )

    return features, performance_index


# ============================================================
# Attrition Prediction
# ============================================================

def predict_attrition(features):

    prediction_value = int(
        model.predict(features)[0]
    )

    # Find the probability column belonging to class 1
    class_index = list(model.classes_).index(1)

    probability_value = float(
        round(
            model.predict_proba(features)[0][class_index] * 100,
            2
        )
    )

    result = (
        "Attrition Likely"
        if prediction_value == 1
        else "Attrition Unlikely"
    )

    return result, probability_value


# ============================================================
# Login
# ============================================================

@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        if check_user(username, password):

            session.clear()
            session["user"] = username

            flash("Login successful!", "success")

            return redirect(url_for("index"))

        flash("Invalid credentials!", "danger")

    return render_template("login.html")


# ============================================================
# Add Employee Prediction
# ============================================================

@app.route("/index", methods=["GET", "POST"])
def index():

    if not is_logged_in():
        return redirect(url_for("login"))

    if request.method == "POST":

        employee_id_raw = request.form.get(
            "employee_id", ""
        ).strip()

        try:

            if not employee_id_raw:
                raise ValueError("Employee ID is required.")

            employee_id = int(employee_id_raw)

            if employee_id <= 0:
                raise ValueError(
                    "Employee ID must be a positive number."
                )

        except ValueError as error:

            flash(str(error), "danger")
            return render_template("index.html")

        employee = db.session.get(
            Employee, employee_id
        )

        if not employee:

            flash(
                f"Employee ID {employee_id} not found in the database.",
                "danger"
            )

            return render_template("index.html")

        existing = EmployeeInput.query.filter_by(
            employee_id=employee_id
        ).first()

        if existing:

            flash(
                f"Prediction already exists for Employee ID {employee_id}. "
                "You can edit it from the records page.",
                "warning"
            )

            return render_template("index.html")

        try:

            (
                data,
                overtime,
                overtime_month,
                overtime_count,
                highsalary,
                longtenure
            ) = get_prediction_inputs(request.form)

            features, performance_index = build_features(
                employee,
                data,
                overtime,
                highsalary,
                longtenure
            )

            result, probability_val = predict_attrition(
                features
            )

        except ValueError as error:

            flash(str(error), "danger")
            return render_template("index.html")

        new_entry = EmployeeInput(
            employee_id=employee_id,
            jobsatisfaction=data["jobsatisfaction"],
            worklifebalance=data["worklifebalance"],
            performancerating=data["performancerating"],
            salary=data["salary"],
            yearsatcompany=data["yearsatcompany"],
            overtime=overtime,
            overtime_month=overtime_month,
            overtime_count=overtime_count,
            highsalary=highsalary,
            longtenure=longtenure,
            performance_index=performance_index,
            prediction=result,
            probability=probability_val,
        )

        try:

            db.session.add(new_entry)
            db.session.commit()

        except Exception:

            db.session.rollback()

            app.logger.exception(
                "Could not save employee prediction."
            )

            flash(
                "Could not save the prediction. Please try again.",
                "danger"
            )

            return render_template("index.html")

        flash(
            "Prediction added successfully!",
            "success"
        )

        input_values = {
            "Department": employee.Department,
            "Job Role": employee.JobRole,
            "Employee ID": employee_id,
            "Job Satisfaction": data["jobsatisfaction"],
            "Work Life Balance": data["worklifebalance"],
            "Performance Rating": data["performancerating"],
            "Salary": data["salary"],
            "Years at Company": data["yearsatcompany"],
            "High Salary": "Yes" if highsalary else "No",
            "Long Tenure": "Yes" if longtenure else "No",
            "Overtime": "Yes" if overtime else "No",
            "Overtime Month": overtime_month,
            "Overtime Count": overtime_count,
        }

        return render_template(
            "index.html",
            prediction=result,
            probability=f"{probability_val}%",
            performance_index=performance_index,
            input_values=input_values,
        )

    return render_template("index.html")


# ============================================================
# View Records
# ============================================================

@app.route("/records")
def records():

    if not is_logged_in():

        flash(
            "Please log in to access records.",
            "warning"
        )

        return redirect(url_for("login"))

    results = (
        db.session.query(
            EmployeeInput,
            Employee.Department,
            Employee.JobRole
        )
        .join(
            Employee,
            Employee.EmployeeID == EmployeeInput.employee_id
        )
        .all()
    )

    return render_template(
        "records.html",
        records=results
    )


# ============================================================
# Edit and Re-predict a Record
# ============================================================

@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit(id):

    if not is_logged_in():

        flash(
            "Please log in to edit records.",
            "warning"
        )

        return redirect(url_for("login"))

    record = db.session.get_or_404(
        EmployeeInput, id
    )

    employee = db.session.get(
        Employee, record.employee_id
    )

    if request.method == "POST":

        try:

            new_emp_id = parse_int(
                request.form,
                "employee_id",
                "Employee ID",
                minimum=1
            )

        except ValueError as error:

            flash(str(error), "danger")

            return render_template(
                "edit.html",
                record=record,
                employee=employee
            )

        valid_employee = db.session.get(
            Employee, new_emp_id
        )

        if not valid_employee:

            flash(
                f"Employee ID {new_emp_id} not found in the master database. "
                "Please use a valid existing Employee ID.",
                "danger"
            )

            return render_template(
                "edit.html",
                record=record,
                employee=employee
            )

        existing = EmployeeInput.query.filter(
            EmployeeInput.employee_id == new_emp_id,
            EmployeeInput.id != id
        ).first()

        if existing:

            flash(
                f"A prediction already exists for Employee ID {new_emp_id}. "
                "Please edit that record instead.",
                "warning"
            )

            return render_template(
                "edit.html",
                record=record,
                employee=employee
            )

        try:

            (
                data,
                overtime,
                overtime_month,
                overtime_count,
                highsalary,
                longtenure
            ) = get_prediction_inputs(request.form)

            features, performance_index = build_features(
                valid_employee,
                data,
                overtime,
                highsalary,
                longtenure
            )

            result, probability_val = predict_attrition(
                features
            )

        except ValueError as error:

            flash(str(error), "danger")

            return render_template(
                "edit.html",
                record=record,
                employee=employee
            )

        record.employee_id = new_emp_id
        record.jobsatisfaction = data["jobsatisfaction"]
        record.worklifebalance = data["worklifebalance"]
        record.performancerating = data["performancerating"]
        record.salary = data["salary"]
        record.yearsatcompany = data["yearsatcompany"]
        record.overtime = overtime
        record.overtime_month = overtime_month
        record.overtime_count = overtime_count
        record.highsalary = highsalary
        record.longtenure = longtenure
        record.performance_index = performance_index
        record.prediction = result
        record.probability = probability_val

        try:

            db.session.commit()

        except Exception:

            db.session.rollback()

            app.logger.exception(
                "Could not update employee prediction."
            )

            flash(
                "Could not update the record. Please try again.",
                "danger"
            )

            return render_template(
                "edit.html",
                record=record,
                employee=employee
            )

        flash(
            "Record updated and re-predicted successfully!",
            "success"
        )

        input_values = {
            "Employee ID": record.employee_id,
            "Department": valid_employee.Department,
            "Job Role": valid_employee.JobRole,
            "Job Satisfaction": record.jobsatisfaction,
            "Work Life Balance": record.worklifebalance,
            "Performance Rating": record.performancerating,
            "Salary": record.salary,
            "Years at Company": record.yearsatcompany,
        }

        return render_template(
            "edit_result.html",
            prediction=record.prediction,
            probability=f"{record.probability}%",
            performance_index=record.performance_index,
            input_values=input_values,
        )

    return render_template(
        "edit.html",
        record=record,
        employee=employee
    )


# ============================================================
# Delete a Record
# ============================================================

@app.route("/delete/<int:id>", methods=["POST"])
def delete(id):

    if not is_logged_in():

        flash(
            "Please log in to delete records.",
            "warning"
        )

        return redirect(url_for("login"))

    record = db.session.get_or_404(
        EmployeeInput, id
    )

    try:

        db.session.delete(record)
        db.session.commit()

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "Could not delete employee prediction."
        )

        flash(
            "Could not delete the record. Please try again.",
            "danger"
        )

        return redirect(url_for("records"))

    flash(
        f"Record ID {id} deleted successfully!",
        "info"
    )

    return redirect(url_for("records"))


# ============================================================
# Refresh CSV for Power BI
# Local Machine Workflow
# ============================================================

@app.route("/refresh_dashboard", methods=["POST"])
def refresh_dashboard():

    if not is_logged_in():
        return redirect(url_for("login"))

    if not is_ceo():

        flash(
            "Access Denied. Only CEO can refresh the dashboard.",
            "danger"
        )

        return redirect(url_for("index"))

    check_db_path = os.path.join(
        BASE_DIR, "check_db.py"
    )

    if not os.path.isfile(check_db_path):

        flash(
            "check_db.py was not found in the project directory.",
            "danger"
        )

        return redirect(
            request.referrer or url_for("index")
        )

    try:

        completed = subprocess.run(
            [sys.executable, check_db_path],
            cwd=BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=120
        )

        app.logger.info(
            "Dashboard CSV refresh output: %s",
            completed.stdout
        )

        flash(
            "Dashboard data refreshed! CSV file updated. "
            "Open Power BI and click Refresh to load the latest data.",
            "success"
        )

    except subprocess.CalledProcessError as error:

        app.logger.error(
            "CSV refresh failed: %s",
            error.stderr
        )

        flash(
            "Could not refresh dashboard data. Check application logs.",
            "danger"
        )

    except subprocess.TimeoutExpired:

        flash(
            "Dashboard data refresh timed out. Please try again.",
            "danger"
        )

    except OSError:

        app.logger.exception(
            "Could not start check_db.py."
        )

        flash(
            "Could not start the dashboard refresh process.",
            "danger"
        )

    return redirect(
        request.referrer or url_for("index")
    )


# ============================================================
# Open Power BI Desktop
# Windows / Local Machine Only
# ============================================================

@app.route("/open_dashboard")
def open_dashboard():

    if not is_logged_in():
        return redirect(url_for("login"))

    if not is_ceo():

        flash(
            "Access Denied. Only CEO can view the dashboard.",
            "danger"
        )

        return redirect(url_for("index"))

    if os.name != "nt":

        flash(
            "Opening Power BI Desktop is available only on your local Windows computer.",
            "warning"
        )

        return redirect(
            request.referrer or url_for("index")
        )

    dashboard_path = os.environ.get(
        "POWERBI_FILE_PATH",
        os.path.join(BASE_DIR, "latest_predictions.pbix")
    )

    dashboard_path = os.path.abspath(
        dashboard_path
    )

    if not os.path.isfile(dashboard_path):

        flash(
            "Dashboard file not found. Check POWERBI_FILE_PATH in your .env file.",
            "danger"
        )

        return redirect(
            request.referrer or url_for("index")
        )

    try:

        os.startfile(dashboard_path)

        flash(
            "Power BI Dashboard opened! Click Refresh in Power BI to load the latest data.",
            "info"
        )

    except OSError:

        app.logger.exception(
            "Could not open the Power BI dashboard."
        )

        flash(
            "Could not open the Power BI dashboard.",
            "danger"
        )

    return redirect(
        request.referrer or url_for("index")
    )


# ============================================================
# Logout
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "Logged out successfully.",
        "info"
    )

    return redirect(url_for("login"))


# ============================================================
# Run Flask Application
# ============================================================

if __name__ == "__main__":

    print(
        "Flask server running at http://127.0.0.1:5000"
    )

    app.run(debug=False)
