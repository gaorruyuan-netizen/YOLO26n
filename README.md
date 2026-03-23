# PABC Compressive Strength Prediction System

This project develops a machine learning-based prediction system for estimating the compressive strength of PABC materials. The system integrates data-driven modeling, interpretability analysis, and a web-based prediction interface.

## Web Application
The online prediction system is available at:
https://pabc-strength-app.onrender.com

## Source Code
GitHub Repository:
https://github.com/MiladBolbolvand/UHPC-Machine-Learning.git

## Features
- Compressive strength prediction using machine learning models
- SHAP interpretability analysis
- PDP and ICE analysis
- Graphical User Interface (GUI)
- Web-based prediction system
- Batch prediction using Excel files
- Visualization of prediction results

## Input Parameters
The model uses the following input parameters:
- Cement
- Sand
- Water
- Fly Ash (FA)
- Silica Ash (SA)
- Coarse Aggregate (CA)
- Superplasticizer
- Fiber
- Temperature

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
