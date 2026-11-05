import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 3: Department-wise Analysis with Dropdown Menu
# SalaryInsight: Predictive Modeling for Informed Salary Predictions

def load_data():
    df = pd.read_csv("data/statistics_practice_dataset_2000.csv")
    return df

def run_department_analysis():
    st.title("Department-Wise Analysis")
    st.write("Select a department from the dropdown to compare KPIs, trends, and productivity.")

    df = load_data()

    # Dropdown menu for department selection
    department_list = ["All Departments"] + list(df['Department'].unique())
    selected_dept = st.selectbox("Select Department:", department_list)

    # Filter data based on selected department
    if selected_dept == "All Departments":
        dept_df = df
    else:
        dept_df = df[df['Department'] == selected_dept]

    st.subheader(f"KPI Overview - {selected_dept}")

    # Calculate department KPIs
    total_employees = len(dept_df)
    avg_salary = dept_df['AnnualSalary'].mean()
    avg_perf = dept_df['PerformanceScore'].mean()
    avg_hours = dept_df['WeeklyHours'].mean()
    avg_projects = dept_df['ProjectsCompleted'].mean()

    # Display KPIs in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Employees", total_employees)
    col2.metric("Avg Salary", f"₹{avg_salary:,.0f}")
    col3.metric("Avg Performance", f"{avg_perf:.1f}/100")
    col4.metric("Avg Weekly Hours", f"{avg_hours:.1f} hrs")
    col5.metric("Avg Projects", f"{avg_projects:.1f}")

    st.write("---")

    # Productivity and Trends Visualizations
    st.subheader(f"Trends and Productivity - {selected_dept}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.write("#### Salary Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(data=dept_df, x='AnnualSalary', bins=20, kde=True, color='skyblue', ax=ax1)
        ax1.set_xlabel("Annual Salary (INR)")
        ax1.set_ylabel("Employee Count")
        ax1.set_title(f"Salary Distribution for {selected_dept}")
        st.pyplot(fig1)

    with col_right:
        st.write(" Productivity: Weekly Hours vs Projects Completed")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='WeeklyHours', y='ProjectsCompleted', data=dept_df, color='coral', ax=ax2)
        ax2.set_xlabel("Weekly Hours")
        ax2.set_ylabel("Projects Completed")
        ax2.set_title(f"Weekly Hours vs Projects for {selected_dept}")
        st.pyplot(fig2)

    st.write("---")

    # Department Comparison Summary Table
    st.subheader("Department-wise Comparison Summary Table")
    summary_table = df.groupby('Department').agg(
        Total_Employees=('EmployeeID', 'count'),
        Average_Salary=('AnnualSalary', 'mean'),
        Average_Performance=('PerformanceScore', 'mean'),
        Average_Weekly_Hours=('WeeklyHours', 'mean'),
        Average_Projects=('ProjectsCompleted', 'mean')
    ).round(2)

    st.dataframe(summary_table)

if __name__ == "__main__":
    run_department_analysis()
