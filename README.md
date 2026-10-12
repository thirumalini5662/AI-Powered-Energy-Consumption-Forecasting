⚡ AI-Powered Energy Consumption Forecasting System

An interactive machine learning project designed to analyze electricity consumption patterns, forecast future energy demand, estimate energy costs, and generate recommendations to support smarter energy management.

Built using Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Streamlit.

«Dataset disclosure: This project uses simulated energy consumption and contextual data for educational purposes. Its results have not been validated against real-world energy systems.»

📌 Project Overview

Electricity consumption changes throughout the day depending on usage patterns, weather conditions, occupancy, and other factors.

This project demonstrates how machine learning can analyze historical energy consumption and predict future demand. It also provides an interactive dashboard for exploring energy trends, evaluating model performance, estimating costs, and identifying periods that may deserve further investigation.

🎯 Objectives

- Analyze historical electricity consumption patterns.
- Explore the relationship between energy usage and contextual variables.
- Forecast future energy consumption using machine learning.
- Compare different regression models.
- Evaluate predictions using MAE, RMSE, and R².
- Estimate costs using assumed electricity tariffs.
- Identify important model features.
- Generate rule-based recommendations for energy management.
- Present insights through an interactive dashboard.

💡 Problem Statement

Unpredictable energy demand can make electricity planning, cost management, and resource allocation more difficult.

This project demonstrates a data-driven approach to forecasting energy consumption and identifying potential peak-demand periods.

The intended applications include smart buildings, manufacturing facilities, energy analytics, and smart-city planning.

🛠️ Technology Stack

Technology| Purpose
Python| Core programming
Pandas| Data processing and cleaning
NumPy| Numerical operations
Scikit-learn| Machine learning and evaluation
Random Forest Regressor| Energy consumption prediction
HistGradientBoosting Regressor| Alternative forecasting model
Matplotlib| Data visualization
Streamlit| Interactive web dashboard
Joblib| Saving trained models

📊 Dataset and Input Factors

The project includes a simulated hourly dataset with the following variables:

- "Datetime" — hourly timestamp
- "Energy_kWh" — electricity consumption
- "Temperature_C" — simulated temperature
- "Humidity_pct" — simulated relative humidity
- "Occupancy_0_1" — simulated occupancy level
- "Tariff_per_kWh" — assumed electricity tariff
- "Is_Holiday" — simulated holiday indicator

The system also derives time-based features and historical consumption indicators.

Note: These values are synthetic. They are not actual smart-meter readings, weather observations, occupancy sensor measurements, or utility tariffs.

🧠 Machine Learning Approach

The project supports two regression models:

1. Random Forest Regressor

Uses an ensemble of decision trees to learn relationships between input features and energy consumption.

2. HistGradientBoosting Regressor

Builds an ensemble of trees sequentially, with later trees helping correct errors made by earlier ones.

Both models are evaluated using a chronological training/testing split.

⚙️ Key Features

1. Energy Consumption Analysis

Visualize historical electricity consumption and explore usage patterns.

2. Contextual Factor Analysis

Examine model inputs such as temperature, humidity, occupancy, tariff, and holidays.

3. Model Comparison

Compare Random Forest and HistGradientBoosting using the same held-out test period.

4. Future Energy Forecasting

Generate simulated forecasts for the next 6–48 hours.

5. Evaluation Metrics

Evaluate the model using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

6. Estimated Energy Costs

Estimate forecast-period costs using predicted consumption and the assumed tariff.

7. Feature Importance

Explore which features contribute most to model predictions.

Feature importance indicates model associations, not proof of causation.

8. Energy Management Recommendations

Generate simple, rule-based suggestions to review potential consumption peaks and unusual usage patterns.

These suggestions are prompts for investigation, not guaranteed energy savings.

9. Data Quality Checks

Inspect the dataset, date range, missing values, and cleaned records.

10. Export and Reporting

Download forecast results and feature importance data as CSV files. Save the selected model and evaluation report from the dashboard.

🏗️ Project Architecture

Simulated Energy Dataset
          |
          v
Data Validation and Cleaning
          |
          v
Feature Engineering
          |
          v
Chronological Train/Test Split
          |
          v
Machine Learning Models
   /                  \
Random Forest    HistGradientBoosting
   \                  /
          v
Model Evaluation
   MAE / RMSE / R²
          |
          v
Energy Forecasting
          |
          v
Cost Estimation and
Rule-Based Recommendations
          |
          v
Interactive Streamlit Dashboard
          |
          v
Charts, Reports and CSV Exports

📁 Project Structure

AI-Energy-Forecasting-Pro/
├── data/
│   └── energy_context.csv
├── docs/
│   └── improvement_roadmap.md
├── images/
├── models/
├── outputs/
├── src/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

🚀 Installation and Setup

Prerequisites

- Python 3.10 or 3.11
- pip
- A code editor such as VS Code

Windows

Extract the project ZIP, open Command Prompt in the project folder, and run:

py -m venv .venv

Activate the environment:

.venv\Scripts\activate

Install the dependencies:

python -m pip install --upgrade pip
pip install -r requirements.txt

Launch the application:

streamlit run main.py

The dashboard should open in your browser.

📈 Dashboard Sections

Overview

- Historical consumption visualization
- Average energy usage
- Peak observed usage
- Temperature and occupancy summaries

Model Comparison

- Random Forest versus HistGradientBoosting
- Actual-versus-predicted visualization
- Regression evaluation metrics

Forecast and Recommendations

- Selectable forecasting horizon
- Predicted energy consumption
- Estimated forecast-period cost
- Downloadable forecast results
- Rule-based recommendations

Factors and Impact

- Feature importance analysis
- Ranking of model input variables

Data Quality

- Dataset inspection
- Date range and row count
- Missing-value checks

📏 Evaluation Metrics

Metric| Meaning
MAE| Average absolute prediction error
RMSE| Error metric that penalizes larger mistakes more heavily
R²| Measures model fit relative to a mean-based reference

Run the application to obtain the actual metrics for the selected model and test split.

Results are not pre-reported because they should be measured by running the implementation.

🌱 Potential Applications

The techniques demonstrated in this project may support investigation of:

- Energy usage patterns in buildings
- Peak-demand planning
- Electricity cost estimation
- Industrial energy analytics
- Smart-building management
- Sustainability and energy-efficiency initiatives

The current implementation is a student prototype, not a deployed industrial system.

🔍 Limitations

- The included dataset is simulated.
- Future contextual variables are assumed rather than obtained from live forecasts.
- The simple recommendation rules are not proof of achievable savings.
- Model performance on synthetic data does not establish real-world accuracy.
- Feature importance does not demonstrate causal relationships.

🚧 Future Improvements

- Integrate a documented, appropriately licensed real-world energy dataset.
- Add actual weather forecasts and public holiday calendars.
- Compare results with seasonal-naive and linear regression baselines.
- Perform rolling-origin backtesting.
- Add prediction intervals to communicate uncertainty.
- Add anomaly detection for unusual consumption.
- Evaluate alternative load schedules using explicit assumptions.
- Develop an API for external applications.
- Consider privacy protections for real occupancy and meter data.

📚 Learning Outcomes

This project demonstrates practical experience with:

- Data preprocessing and feature engineering
- Regression-based machine learning
- Time-aware model evaluation
- Forecast visualization
- Model comparison
- Feature importance analysis
- Interactive dashboard development
- Cost estimation
- GitHub project documentation

👩‍💻 Author

Developed as an educational AI and energy analytics project.

⚠️ Disclaimer

This project is intended for learning and demonstration only. It is not validated for electricity billing, grid dispatch, safety-critical operations, or guaranteed cost-saving decisions.
