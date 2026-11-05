import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 4: Sidebar Navigation with Structured Sections
# SalaryInsight: Predictive Modeling for Informed Salary Predictions

def load_data():
    return pd.read_csv("data/statistics_practice_dataset_2000.csv")

def run_app():
    st.set_page_config(page_title="SalaryInsight App", layout="wide")
    
    # Sidebar Navigation
    st.sidebar.title(" Navigation Menu")
    st.sidebar.write("Choose a section to explore:")
    
    menu_options = [
        "📊 Data Visualization",
        "🔮 Model Predictions",
        "👥 Employee Insights"
    ]
    
    selected_section = st.sidebar.radio("Go to:", menu_options)
    
    df = load_data()
    
    
    # SECTION 1: DATA VISUALIZATION
    
    if selected_section == " Data Visualization":
        st.title(" Data Visualization Section")
        st.write("Explore dataset distributions and relationships between features.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Salary by Department")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Department', y='AnnualSalary', data=df, estimator='mean', errorbar=None, palette='Blues_d', ax=ax1)
            plt.xticks(rotation=30)
            ax1.set_ylabel("Average Annual Salary (INR)")
            st.pyplot(fig1)
            
        with col2:
            st.write("### Experience vs Annual Salary")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x='ExperienceYears', y='AnnualSalary', hue='Department', data=df, alpha=0.7, ax=ax2)
            ax2.set_xlabel("Experience (Years)")
            ax2.set_ylabel("Annual Salary (INR)")
            st.pyplot(fig2)
            
        st.write("### Weekly Hours vs Performance Score")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(x='WeeklyHours', y='PerformanceScore', hue='Gender', data=df, alpha=0.6, ax=ax3)
        ax3.set_xlabel("Weekly Hours")
        ax3.set_ylabel("Performance Score")
        st.pyplot(fig3)

    
    # SECTION 2: MODEL PREDICTIONS
    
    elif selected_section == " Model Predictions":
        st.title("Salary Model Predictions")
        st.write("Enter employee details below to predict expected annual salary.")
        
        col_in1, col_in2 = st.columns(2)
        
        with col_in1:
            age = st.slider("Age", 18, 60, 30)
            experience = st.slider("Experience in Years", 0, 35, 5)
            performance = st.slider("Performance Score", 40, 100, 75)
            department = st.selectbox("Department", df['Department'].unique())
            
        with col_in2:
            weekly_hours = st.slider("Weekly Hours", 30.0, 70.0, 45.0, 1.0)
            projects = st.slider("Projects Completed", 1, 25, 8)
            satisfaction = st.slider("Satisfaction Score (1-10)", 1.0, 10.0, 7.0, 0.5)
            city = st.selectbox("City", df['City'].unique())
            
        if st.button("Predict Salary"):
            # Simple rule-based/baseline prediction formula for demonstration
            base_salary = 600000
            exp_bonus = experience * 30000
            perf_bonus = (performance - 50) * 5000
            project_bonus = projects * 10000
            
            estimated_salary = base_salary + exp_bonus + perf_bonus + project_bonus
            
            st.success(f"Estimated Annual Salary: ₹{estimated_salary:,.2f}")
            st.info(f"Department: {department} | City: {city} | Experience: {experience} Years")

    
    # SECTION 3: EMPLOYEE INSIGHTS
    
    elif selected_section == "👥 Employee Insights":
        st.title("👥 Employee Insights & Summary")
        st.write("Key takeaways and summary metrics from the employee dataset.")
        
        # Display Quick KPI Metrics
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Headcount", len(df))
        kpi2.metric("Median Salary", f"₹{df['AnnualSalary'].median():,.0f}")
        kpi3.metric("Average Experience", f"{df['ExperienceYears'].mean():.1f} Years")
        
        st.write("---")
        
        st.write("### Top Performing Employees (Performance >= 90)")
        top_performers = df[df['PerformanceScore'] >= 90][['EmployeeID', 'Age', 'Department', 'ExperienceYears', 'PerformanceScore', 'AnnualSalary', 'ProjectsCompleted']]
        st.dataframe(top_performers.head(10))
        
        st.write("### Key Takeaways")
        st.markdown("""
        - **Experience Impact:** Experience is a strong driver of compensation across technical and business departments.
        - **Workload Balance:** Most employees work around 45–50 hours per week with good project output.
        - **Performance Rewards:** High performance scores (90+) are distributed across all departments.
        """)

if __name__ == "__main__":
    run_app()
