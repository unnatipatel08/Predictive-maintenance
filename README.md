# Predictive-maintenance# AI-Based Predictive Maintenance System

A Python and Streamlit project that demonstrates how machine sensor readings can be used to estimate industrial machine failure risk and provide maintenance alerts.

> **Educational Demo Project:** This system uses demonstration data. Predictions are not validated for real industrial safety or maintenance decisions. The current app does not receive live physical-sensor data.
>
> # 🚀 Live AI-Based Predictive Maintenance System

[predictive-maintenance-npzh9ubtjnm9epvwhzteqq](url)

## 🚀 Project Features


### 1. Dashboard
- Displays all original columns and records from `ai4i2020.csv`.
- Search machine records and values.
- Download the original CSV dataset.
- View failure statistics and charts.

### 2. Machine Management
- Select a machine using its Machine ID.
- Remove a machine from the app's operational list.
- Restore a previously removed machine.
- Add and save maintenance history.
- Record inspection, cleaning, repair, calibration, and tool replacement activities.

### 3. Manual Machine Prediction
- Select a machine from the available dataset.
- Enter new sensor readings manually.
- Leave fields blank to use the selected machine's CSV readings.
- Estimate machine failure risk using a Random Forest model.

### 4. Machine Health Alerts
The system displays health alerts based on estimated failure risk:

- 🔴 **High Risk:** 50% or greater
- 🟡 **Medium Risk:** 20% to less than 50%
- 🟢 **Low Risk:** Less than 20%

These thresholds are demonstration settings and are not industrially validated alarm limits.

### 5. Smart Maintenance Suggestions
The application provides suggestions based on estimated risk and sensor readings, such as:
- Arrange a machine inspection.
- Monitor temperature and rotational speed.
- Check machine load and torque.
- Inspect tool wear and review maintenance schedules.

### 6. Prediction and Analysis
- Random Forest failure-risk estimation.
- Feature importance visualization.
- Statistical summary.
- Feature correlation analysis.
- Temperature hypothesis testing.

### 7. Personalized Maintenance Report
- View selected machine details.
- Review sensor readings and estimated failure risk.
- See machine health alerts and maintenance suggestions.
- Download the report as a CSV file.

## 📊 Dataset Information

The demonstration dataset is stored in `ai4i2020.csv`.

- **Total Records:** 500
- **Recorded Failure = 1:** 200
- **Recorded Failure = 0:** 300
- **Failure-label proportion:** 40%
- **Total Columns:** 14

The original CSV columns are:

`UDI`, `Product ID`, `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

**Important:** The 40% failure proportion is configured for demonstration and testing. It should not be interpreted as the real-world failure rate of industrial equipment.

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
- Random Forest Classifier
- SQLite
- CSV Dataset
- GitHub
- Streamlit Community Cloud

## 📁 Project Structure

Keep the following files in the same GitHub repository folder:

```text
predictive-maintenance/
├── app.py
├── ai4i2020.csv
├── requirements.txt
└── README.md
```

The app creates `maintenance.db` at runtime to store maintenance history and machine removal/restoration choices.

## 📦 Requirements

Create a file named `requirements.txt` containing:

```text
streamlit
pandas
numpy
matplotlib
scipy
scikit-learn
```

## ▶️ Run the Project Locally

1. Install Python on your computer.
2. Open a terminal in the project folder.
3. Install the required packages:

```bash
python -m pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
python -m streamlit run app.py
```

5. Open the local URL displayed in the terminal.

## ☁️ Deploy on Streamlit Community Cloud

1. Upload `app.py`, `ai4i2020.csv`, `requirements.txt`, and `README.md` to your GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub.
4. Create an app and select your repository.
5. Set the main file path to `app.py`.
6. Click **Deploy**.

When you commit changes to GitHub, Streamlit Cloud normally rebuilds the application automatically.

## 🤖 How the AI Prediction Works

The Random Forest model uses these four sensor features:

- Air Temperature (K)
- Rotational Speed (rpm)
- Torque (Nm)
- Tool Wear (min)

The target column is `Machine failure`.

The trained model estimates the probability associated with the failure class. The application uses the estimated percentage to display a health category and maintenance suggestions.

A prediction is an estimate, not a guarantee of actual machine failure.

## 🔌 Future Scope: Real-Time Sensor Integration

The current project uses historical readings from `ai4i2020.csv`; it is not connected to physical sensors.

A future version could integrate:

1. Temperature, speed, and torque sensors.
2. An ESP32 controller or industrial PLC.
3. An API or MQTT service to transmit sensor readings.
4. Live monitoring and automatic failure-risk alerts.
5. Persistent online database storage and notifications.

Tool wear may require operating-time tracking, inspection, or a specialised measurement method. It is not automatically measured by an ordinary temperature sensor.

## ⚠️ Limitations

- This project is designed for educational demonstration.
- The modified dataset's 40% failure-label proportion does not represent a verified real-world failure rate.
- Live physical-sensor integration is not currently implemented.
- The alert thresholds and maintenance suggestions are illustrative.
- SQLite storage on a hosted service may not persist across every restart or redeployment.
- Predictions should not replace professional machine inspection or validated industrial monitoring systems.

## 👩‍💻 Conclusion

The AI-Based Predictive Maintenance System demonstrates how Python, machine-learning techniques, statistical analysis, and data visualization can support machine-health monitoring.

The project provides a foundation for future improvements such as real-time sensor integration, improved models, persistent database storage, and automated maintenance notifications.
