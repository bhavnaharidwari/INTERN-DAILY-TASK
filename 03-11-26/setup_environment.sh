#!/usr/bin/env bash
# Task 9: Virtual Environment Setup & Deployment Configuration Script
# Project: SalaryInsight: Predictive Modeling for Informed Salary Predictions


set -e


echo "SALARYINSIGHT: ENVIRONMENT & DEPENDENCY CONFIGURATION"
echo"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"

echo "[1/4] Checking Python environment..."
python3 --version

echo "[2/4] Setting up Python Virtual Environment in $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created successfully."
else
    echo "Existing virtual environment found."
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "[3/4] Installing required dependencies from requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "[4/4] Validating installations and model artifacts..."
python -c "
import streamlit
import pandas
import numpy
import sklearn
import xgboost
import plotly
import matplotlib
import seaborn
import joblib
print('✓ All core libraries imported successfully!')
"

