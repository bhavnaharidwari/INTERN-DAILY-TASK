import streamlit as st
import pandas as pd

# Task 1: Basic UI Layout with Responsive Design & Intuitive Interface
# SalaryInsight: Predictive Modeling for Informed Salary Predictions

def load_data():
    return pd.read_csv("data/statistics_practice_dataset_2000.csv")

def run_task_1():
    # 1. Page Configuration
    st.set_page_config(
        page_title="SalaryInsight - Basic UI Layout",
        page_icon="",
        layout="wide"
    )

    # 2. Header and Introduction
    st.title("SalaryInsight: Workforce Analytics")
    st.caption("A simple, accessible, and intuitive user interface for exploring employee compensation.")

    st.write("---")

    df = load_data()

    # 3. Top KPI Metric Cards (Responsive 4-column layout)
    st.subheader(" Key Metrics Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Employees", value=f"{len(df):,}")
    with col2:
        st.metric(label="Median Salary", value=f"₹{df['AnnualSalary'].median():,.0f}")
    with col3:
        st.metric(label="Avg Experience", value=f"{df['ExperienceYears'].mean():.1f} Years")
    with col4:
        st.metric(label="Avg Performance", value=f"{df['PerformanceScore'].mean():.1f} / 100")

    st.write("---")

    # 4. Main Content Layout (Two Columns)
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("ℹ About the Dataset")
        st.write("""
        This dataset contains information about **2,000 employees** across multiple functional departments and work locations.
        Key features include:
        - **Demographics:** Age, Gender, City
        - **Seniority:** Total Experience (Years), Tenure
        - **Compensation:** Annual Salary (INR)
        - **Performance & Workload:** Performance Score, Weekly Hours, Projects Completed
        """)

    with right_col:
        st.subheader(" Department Headcount Distribution")
        dept_counts = df['Department'].value_counts()
        st.bar_chart(dept_counts)

    st.write("---")

    # 5. Dataset Preview Section
    st.subheader(" Employee Dataset Preview")
    show_records = st.slider("Select number of records to preview:", min_value=5, max_value=25, value=10)
    st.dataframe(df.head(show_records), use_container_width=True)

    # 6. Footer
    st.write("---")
    st.caption("Designed with responsive layout principles and clear visual hierarchy.")

if __name__ == "__main__":
    run_task_1()
