#!/usr/bin/env bash
# Task 9: Streamlit Application Production Runner
# Project: SalaryInsight: Predictive Modeling for Informed Salary Predictions


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# If virtual environment exists, use it
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi


echo "LAUNCHING SALARYINSIGHT STREAMLIT APPLICATION"
echo "Port: 8501 | Address: 0.0.0.0"

exec streamlit run "$SCRIPT_DIR/app.py" \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
