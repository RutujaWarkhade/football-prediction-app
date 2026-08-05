# ⚽ Football Predictor Hub – AI-Powered Football Analytics Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-blue)

## 🌐 Live Demo

👉 https://football-predictor-app-avqfqwnuxcf6fnsxchbugr.streamlit.app/

---

# 📌 Project Overview

Football Predictor Hub is an AI-powered football analytics platform that leverages Machine Learning and Deep Learning techniques to predict football outcomes through an intuitive Streamlit interface.

The application combines multiple predictive models into a single dashboard, enabling users to analyze football data and generate predictions for:

- 🏆 League Winner Prediction
- ⚽ Match Outcome Prediction
- 🥅 Player Goals Prediction
- 🎯 Player Assists Prediction

The platform integrates Random Forest, XGBoost, and TensorFlow models with an interactive web interface, making advanced football analytics accessible to football enthusiasts, analysts, and researchers.

---

# 📸 Application Screenshots

## 🏠 Home Dashboard

<p align="center">
<img src="https://raw.githubusercontent.com/RutujaWarkhade/football-prediction-app/main/Frontend_Images/dashboard.png" width="900">
</p>

---

## 📊 Prediction Modules

<p align="center">
<img src="https://raw.githubusercontent.com/RutujaWarkhade/football-prediction-app/main/Frontend_Images/dashboard_prediction_module.png" width="900">
</p>

---

## 🚀 Dashboard Working

<p align="center">
<img src="https://raw.githubusercontent.com/RutujaWarkhade/football-prediction-app/main/Frontend_Images/dashboard_working.png" width="900">
</p>

---

## 🏆 League Winner Predictor

<table align="center">
<tr>

<td>
<img src="https://raw.githubusercontent.com/RutujaWarkhade/football-prediction-app/main/Frontend_Images/league_predictor.png" width="450">
</td>

<td>
<img src="https://raw.githubusercontent.com/RutujaWarkhade/football-prediction-app/main/Frontend_Images/league_predictor_result.png" width="450">
</td>

</tr>
</table>

---

## ⚽ Match Outcome Predictor

<p align="center">
<img src="https://raw.githubusercontent.com/RutujaWarkhade/football-prediction-app/main/Frontend_Images/match_outcome_predictor.png" width="900">
</p>

---

## 🥅 Player Performance Predictor

<table align="center">
<tr>

<td>
<img src="https://raw.githubusercontent.com/RutujaWarkhade/football-prediction-app/main/Frontend_Images/player_performance_predictor.png" width="450">
</td>

<td>
<img src="https://raw.githubusercontent.com/RutujaWarkhade/football-prediction-app/main/Frontend_Images/player_performance_result.png" width="450">
</td>

</tr>
</table>

---

# 🚀 Features

## 🏆 League Winner Prediction

Predicts whether a football team is likely to become the league champion based on historical league statistics and season performance.

Features:

- Random Forest Classification Model
- Champion Probability Prediction
- Optimized Decision Threshold
- Interactive Prediction Dashboard

---

## ⚽ Match Winner Prediction

Predicts the outcome of a football match using recent team performance and statistical features.

Prediction includes:

- Home Win Probability
- Match Outcome Prediction
- Team Form Analysis
- Winning Probability Score

---

## 🥅 Player Performance Prediction

Predicts individual player performance using machine learning.

Supports prediction of:

- Goals
- Assists

The application uses trained XGBoost regression models to estimate player statistics from historical performance data.

---

## 🎨 Interactive Streamlit Dashboard

The application provides an easy-to-use interface with:

- Responsive Design
- Sidebar Navigation
- Interactive Forms
- Real-time Predictions
- Clean Visualization

---

## 🤖 Multiple Machine Learning Models

This project integrates multiple machine learning algorithms:

- Random Forest
- XGBoost
- TensorFlow Neural Network

Each model is optimized for its specific prediction task.

---

# 🏗️ System Architecture

The Football Predictor Hub follows a modular machine learning architecture where each prediction task is handled by a dedicated model.

```
                User Input
                     │
                     ▼
        Streamlit Interactive Dashboard
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
 League Winner   Match Winner   Player Performance
   Predictor      Predictor     (Goals & Assists)
        │            │            │
        ▼            ▼            ▼
 Random Forest   TensorFlow     XGBoost Models
        │            │            │
        └────────────┼────────────┘
                     ▼
            Prediction Results
                     ▼
            Streamlit User Interface
```

---

# 🧠 Machine Learning Models

The application integrates multiple machine learning models, each optimized for a specific prediction task.

## 🏆 League Winner Prediction

**Algorithm**

- Random Forest Classifier

**Prediction**

- Champion Probability
- League Winner Classification

---

## ⚽ Match Winner Prediction

**Algorithm**

- TensorFlow Neural Network

**Prediction**

- Home Win Probability
- Match Outcome

---

## 🥅 Player Performance Prediction

**Algorithm**

- XGBoost Regressor

**Prediction**

- Goals
- Assists

---

# ⚙️ Prediction Workflow

## Step 1

The user selects a prediction module from the sidebar.

Available modules:

- Home
- League Winner
- Match Winner
- Goals & Assists

↓

## Step 2

The user enters the required football statistics.

↓

## Step 3

Input data is preprocessed using the saved scaler and metadata.

↓

## Step 4

The appropriate machine learning model is loaded.

↓

## Step 5

The trained model generates predictions.

↓

## Step 6

Results are displayed instantly in the Streamlit dashboard.

---

# 🧰 Technologies Used

## Frontend

- Streamlit
- HTML
- CSS

---

## Backend

- Python

---

## Machine Learning

- TensorFlow
- Keras
- Scikit-learn
- Random Forest
- XGBoost

---

## Data Processing

- Pandas
- NumPy

---

## Model Serialization

- Joblib
- JSON

---

## Development Tools

- VS Code
- Git
- GitHub

---

# 📂 Project Structure

```bash
football-prediction-app/
│
├── app_main_responsive.py
├── predict_match.py
│
├── models/
│   ├── best_football_predictor.h5
│   ├── rf_model.joblib
│   ├── scaler.joblib
│   ├── feature_scaler.pkl
│   ├── xgb_goals_pipeline.pkl
│   ├── xgb_assists_pipeline.pkl
│   ├── model_metadata.json
│   ├── metadata_goals.json
│   ├── metadata_assists.json
│   └── threshold.json
│
├── Frontend_Images/
├── requirements.txt
├── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/RutujaWarkhade/football-prediction-app.git

cd football-prediction-app
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Application

```bash
streamlit run app_main_responsive.py
```

---

# ▶️ How to Use

### 🏠 Home

View the project dashboard and choose a prediction module.

---

### 🏆 League Winner

Enter league statistics and click **Predict Champion Probability**.

---

### ⚽ Match Winner

Provide team performance features and click **Predict Match Outcome**.

---

### 🥅 Goals & Assists

Select either **Goals** or **Assists**, enter player statistics, and generate predictions.

---

# 📊 Key Functionalities

| Feature | Description |
|----------|-------------|
| League Winner Prediction | Predict league champion probability |
| Match Winner Prediction | Predict match outcome |
| Goals Prediction | Estimate player goals |
| Assists Prediction | Estimate player assists |
| Interactive Dashboard | Streamlit-based responsive UI |
| Machine Learning Models | Random Forest, TensorFlow & XGBoost |
| Live Deployment | Accessible through Streamlit Cloud |

---

# 📈 Future Improvements

- Live football data integration
- Team comparison dashboard
- Player recommendation system
- Season simulation
- Injury prediction
- Match visualization
- Explainable AI (SHAP)
- Historical performance analysis
- REST API using FastAPI
- Docker deployment
- Cloud deployment on AWS/Azure

---

# 🎯 Learning Outcomes

Through this project, I learned:

- Machine Learning Model Deployment
- Random Forest Classification
- XGBoost Regression
- TensorFlow Neural Networks
- Feature Scaling
- Model Serialization
- Streamlit Application Development
- Responsive Dashboard Design
- JSON Metadata Management
- End-to-End ML Project Development

---

# 👩‍💻 Author

**Rutuja Shivaji Warkhade**

B.Tech Computer Engineering Student

AI/ML & Data Science Enthusiast

📧 Email: **rutujawarkhade14@gmail.com**

💻 GitHub: **https://github.com/RutujaWarkhade**

---

# 📜 Disclaimer

This application is developed for educational and research purposes only. The predictions generated by the machine learning models are based on historical football data and statistical patterns. They should be used as analytical insights rather than guaranteed outcomes of future matches or tournaments.

---

⭐ **If you found this project useful, consider giving it a Star on GitHub!**
