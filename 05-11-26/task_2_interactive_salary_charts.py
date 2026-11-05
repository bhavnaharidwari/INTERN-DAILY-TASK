import streamlit as st
import pandas as pd
import plotly.express as px

# Task 2: Interactive Salary Distribution Charts with Filters
# SalaryInsight: Predictive Modeling for Informed Salary Predictions

def load_data():
    return pd.read_csv("data/statistics_practice_dataset_2000.csv")

def run_task_2():
    st.set_page_config(page_title="SalaryInsight - Salary Distribution Charts", layout="wide")

    st.title("Interactive Salary Distribution & Filters")
    st.write("Use the filter controls below to explore salary distributions across experience levels, departments, and performance ratings.")

    df = load_data()

    # 1. FILTER CONTROLS SECTION
    st.subheader("Filter Controls")
    
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        # Filter 1: Department multiselect
        dept_options = list(df['Department'].unique())
        selected_depts = st.multiselect(
            "Select Departments:",
            options=dept_options,
            default=dept_options
        )

    with col_f2:
        # Filter 2: Experience level range
        min_exp = int(df['ExperienceYears'].min())
        max_exp = int(df['ExperienceYears'].max())
        selected_exp = st.slider(
            "Experience Range (Years):",
            min_value=min_exp,
            max_value=max_exp,
            value=(min_exp, max_exp)
        )

    with col_f3:
        # Filter 3: Performance score range
        min_perf = int(df['PerformanceScore'].min())
        max_perf = int(df['PerformanceScore'].max())
        selected_perf = st.slider(
            "Performance Rating Range (0-100):",
            min_value=min_perf,
            max_value=max_perf,
            value=(min_perf, max_perf)
        )

    # Apply filters to dataframe
    filtered_df = df[
        (df['Department'].isin(selected_depts)) &
        (df['ExperienceYears'] >= selected_exp[0]) &
        (df['ExperienceYears'] <= selected_exp[1]) &
        (df['PerformanceScore'] >= selected_perf[0]) &
        (df['PerformanceScore'] <= selected_perf[1])
    ]

    st.write(f"**Filtered Employees Found:** `{len(filtered_df)}` out of `{len(df)}` total records")
    st.write("---")

    if len(filtered_df) == 0:
        st.warning("No employees match the selected filter criteria. Please broaden your filter ranges.")
        return
    # 2. INTERACTIVE CHARTS

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("1. Salary Distribution Histogram")
        fig_hist = px.histogram(
            filtered_df,
            x="AnnualSalary",
            nbins=30,
            color="Department",
            marginal="box",
            title="Interactive Annual Salary Distribution",
            labels={"AnnualSalary": "Annual Salary (INR)"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with chart_col2:
        st.subheader("2. Salary Spread by Department")
        fig_box = px.box(
            filtered_df,
            x="Department",
            y="AnnualSalary",
            color="Department",
            points="outliers",
            title="Department Salary Spread & Outliers",
            labels={"AnnualSalary": "Annual Salary (INR)"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.write("---")

    st.subheader("3. Experience vs. Salary by Performance Rating")
    fig_scatter = px.scatter(
        filtered_df,
        x="ExperienceYears",
        y="AnnualSalary",
        color="Department",
        size="PerformanceScore",
        hover_data=["EmployeeID", "Age", "City", "WeeklyHours"],
        title="Experience vs. Salary (Bubble Size indicates Performance Rating)",
        labels={
            "ExperienceYears": "Experience (Years)",
            "AnnualSalary": "Annual Salary (INR)",
            "PerformanceScore": "Performance Rating"
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

if __name__ == "__main__":
    run_task_2()
