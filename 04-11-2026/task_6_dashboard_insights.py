import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 6: Key Dataset Insights, Descriptive Statistics & KPI Metrics
# SalaryInsight: Predictive Modeling for Informed Salary Predictions

def load_data():
    return pd.read_csv("data/statistics_practice_dataset_2000.csv")

def run_task_6_insights():
    st.title("Task 6: Key Dataset Insights & Summary Dashboard")
    st.write("Overview of dataset statistics, summary tables, and high-level KPI metrics.")

    df = load_data()


    # 1. TOP-LEVEL KPI METRICS
    
    st.subheader(" Key Performance Indicators (KPIs)")
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    kpi1.metric("Total Records", f"{len(df):,}")
    kpi2.metric("Average Salary", f"₹{df['AnnualSalary'].mean():,.0f}")
    kpi3.metric("Median Salary", f"₹{df['AnnualSalary'].median():,.0f}")
    kpi4.metric("Avg Experience", f"{df['ExperienceYears'].mean():.1f} Years")
    kpi5.metric("Avg Performance", f"{df['PerformanceScore'].mean():.1f}/100")

    st.write("---")

    
    # 2. DESCRIPTIVE STATISTICS
    
    st.subheader(" Descriptive Statistics (Numerical Features)")
    st.write("Summary statistics including count, mean, standard deviation, min, max, and quantiles.")
    
    num_cols = ['Age', 'ExperienceYears', 'AnnualSalary', 'PerformanceScore', 'WeeklyHours', 'ProjectsCompleted', 'SatisfactionScore', 'MonthlySalesK']
    desc_stats = df[num_cols].describe().round(2)
    st.dataframe(desc_stats, use_container_width=True)

    st.write("---")

    
    # 3. SUMMARY TABLES BY CATEGORIES
    
    st.subheader(" Summary Breakdown Tables")

    tab1, tab2, tab3 = st.tabs(["Department Summary", "Gender Summary", "City Summary"])

    with tab1:
        st.write("#### Department Breakdown")
        dept_summary = df.groupby('Department').agg(
            Employee_Count=('EmployeeID', 'count'),
            Mean_Salary=('AnnualSalary', 'mean'),
            Median_Salary=('AnnualSalary', 'median'),
            Mean_Performance=('PerformanceScore', 'mean'),
            Mean_Weekly_Hours=('WeeklyHours', 'mean'),
            Mean_Projects=('ProjectsCompleted', 'mean')
        ).round(2)
        st.dataframe(dept_summary, use_container_width=True)

    with tab2:
        st.write("#### Gender Breakdown")
        gender_summary = df.groupby('Gender').agg(
            Employee_Count=('EmployeeID', 'count'),
            Mean_Salary=('AnnualSalary', 'mean'),
            Median_Salary=('AnnualSalary', 'median'),
            Mean_Satisfaction=('SatisfactionScore', 'mean')
        ).round(2)
        st.dataframe(gender_summary, use_container_width=True)

    with tab3:
        st.write("#### City / Location Breakdown")
        city_summary = df.groupby('City').agg(
            Employee_Count=('EmployeeID', 'count'),
            Mean_Salary=('AnnualSalary', 'mean'),
            Median_Salary=('AnnualSalary', 'median')
        ).round(2)
        st.dataframe(city_summary, use_container_width=True)

    st.write("---")

    
    # 4. KEY TAKEAWAYS & OBSERVATIONS
    
    st.subheader(" Key Insights for Decision-Makers")
    st.markdown("""
    1. **Salary Distribution:** The median salary across the company is **₹906,629**, while average salary is **₹1,087,100** due to higher executive pay brackets.
    2. **Department Performance:** IT and Sales represent the largest departments, with high project delivery and commercial sales volume.
    3. **Workload and Satisfaction:** Average weekly hours stand at **49.9 hours**, with an average satisfaction rating of **7.0 / 10**.
    4. **Experience Spread:** Average workforce experience is **11.2 years**, ranging from entry-level graduates to 37 years of industry experience.
    """)

if __name__ == "__main__":
    run_task_6_insights()
