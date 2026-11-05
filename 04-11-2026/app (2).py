import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==============================================================================
# SalaryInsight: Predictive Modeling for Informed Salary Predictions
# Integrated Human-Readable Streamlit Application
# Incorporates Tasks 3, 4, 5, 6, 7, 8, 9
# ==============================================================================

st.set_page_config(
    page_title="SalaryInsight Dashboard",
    page_icon="💼",
    layout="wide"
)

# ------------------------------------------------------------------------------
# DATA LOADING FUNCTION
# ------------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    data_path = "data/statistics_practice_dataset_2000.csv"
    if not os.path.exists(data_path):
        data_path = "uploads/statistics_practice_dataset_2000.csv"
    return pd.read_csv(data_path)

df = load_dataset()

# ------------------------------------------------------------------------------
# TASK 5: DYNAMIC ML MODEL LOADING & ERROR HANDLING
# ------------------------------------------------------------------------------
MODEL_PATH = "saved_models/salary_prediction_model.joblib"

@st.cache_resource
def load_or_train_ml_model():
    """Loads trained model dynamically with error handling and fallback training."""
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return model, "Loaded saved model from disk"
        else:
            # Train a simple fallback Random Forest model
            features = ['Age', 'ExperienceYears', 'PerformanceScore', 'WeeklyHours', 'ProjectsCompleted', 'SatisfactionScore', 'MonthlySalesK']
            X = df[features]
            y = df['AnnualSalary']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            os.makedirs("saved_models", exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            return model, "Trained fresh baseline model"
    except Exception as e:
        # Fallback in memory
        features = ['Age', 'ExperienceYears', 'PerformanceScore', 'WeeklyHours', 'ProjectsCompleted', 'SatisfactionScore', 'MonthlySalesK']
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(df[features], df['AnnualSalary'])
        return model, f"In-memory fallback model ({e})"

ml_model, model_status = load_or_train_ml_model()

# ------------------------------------------------------------------------------
# TASK 4: SIDEBAR NAVIGATION WITH STRUCTURED SECTIONS
# ------------------------------------------------------------------------------
st.sidebar.title("💼 SalaryInsight Menu")
st.sidebar.write("Navigate between application sections:")

menu_options = [
    "📋 Dashboard Overview & KPIs (Task 6)",
    "🏢 Department-Wise Analysis (Task 3)",
    "📊 Data Visualization (Task 4)",
    "🔮 Model Predictions (Task 5)",
    "👥 Employee Insights (Task 4 & 8)",
    "🧬 Clustering & Personas (Task 7)"
]

selected_page = st.sidebar.radio("Go to:", menu_options)

st.sidebar.markdown("---")
st.sidebar.caption("SalaryInsight v1.0 | Streamlit Local App")

# ==============================================================================
# SECTION 1: DASHBOARD OVERVIEW & KEY DATASET INSIGHTS (TASK 6)
# ==============================================================================
if selected_page == "📋 Dashboard Overview & KPIs (Task 6)":
    st.title("📋 SalaryInsight: Key Dataset Insights & KPIs")
    st.write("Summary statistics, descriptive metrics, and key performance indicators.")

    # KPI Metrics
    st.subheader("📌 Key Performance Indicators (KPIs)")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Employees", f"{len(df):,}")
    k2.metric("Average Salary", f"₹{df['AnnualSalary'].mean():,.0f}")
    k3.metric("Median Salary", f"₹{df['AnnualSalary'].median():,.0f}")
    k4.metric("Avg Experience", f"{df['ExperienceYears'].mean():.1f} Yrs")
    k5.metric("Avg Performance", f"{df['PerformanceScore'].mean():.1f}/100")

    st.write("---")

    # Descriptive Statistics
    st.subheader("📊 Descriptive Statistics")
    st.write("Summary table of all numerical variables in the workforce dataset:")
    num_cols = ['Age', 'ExperienceYears', 'AnnualSalary', 'PerformanceScore', 'WeeklyHours', 'ProjectsCompleted', 'SatisfactionScore', 'MonthlySalesK']
    st.dataframe(df[num_cols].describe().round(2), use_container_width=True)

    st.write("---")

    # Summary Tables
    st.subheader("📋 Department & Demographic Summary Tables")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.write("#### Department Summary")
        dept_sum = df.groupby('Department').agg(
            Employees=('EmployeeID', 'count'),
            Mean_Salary=('AnnualSalary', 'mean'),
            Mean_Performance=('PerformanceScore', 'mean'),
            Mean_Hours=('WeeklyHours', 'mean')
        ).round(2)
        st.dataframe(dept_sum, use_container_width=True)

    with col_t2:
        st.write("#### Gender Summary")
        gen_sum = df.groupby('Gender').agg(
            Employees=('EmployeeID', 'count'),
            Mean_Salary=('AnnualSalary', 'mean'),
            Mean_Satisfaction=('SatisfactionScore', 'mean')
        ).round(2)
        st.dataframe(gen_sum, use_container_width=True)

# ==============================================================================
# SECTION 2: DEPARTMENT-WISE ANALYSIS WITH DROPDOWN (TASK 3)
# ==============================================================================
elif selected_page == "🏢 Department-Wise Analysis (Task 3)":
    st.title("🏢 Department-Wise Analysis")
    st.write("Select a department from the dropdown menu to analyze KPIs, trends, and productivity variations.")

    dept_options = ["All Departments"] + list(df['Department'].unique())
    selected_dept = st.selectbox("Choose Department:", dept_options)

    if selected_dept == "All Departments":
        d_df = df
    else:
        d_df = df[df['Department'] == selected_dept]

    st.subheader(f"KPIs for {selected_dept}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employee Count", len(d_df))
    c2.metric("Average Salary", f"₹{d_df['AnnualSalary'].mean():,.0f}")
    c3.metric("Avg Performance Score", f"{d_df['PerformanceScore'].mean():.1f}/100")
    c4.metric("Avg Weekly Hours", f"{d_df['WeeklyHours'].mean():.1f} hrs")

    st.write("---")

    st.subheader("Department Trends & Productivity")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.write("#### Salary Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(d_df['AnnualSalary'], bins=18, kde=True, color='royalblue', ax=ax)
        ax.set_xlabel("Annual Salary (INR)")
        ax.set_ylabel("Count")
        ax.set_title(f"Salary Distribution - {selected_dept}")
        st.pyplot(fig)

    with col_g2:
        st.write("#### Productivity: Weekly Hours vs Projects Completed")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='WeeklyHours', y='ProjectsCompleted', data=d_df, color='darkorange', alpha=0.7, ax=ax2)
        ax2.set_xlabel("Weekly Hours")
        ax2.set_ylabel("Projects Completed")
        ax2.set_title(f"Hours vs Projects - {selected_dept}")
        st.pyplot(fig2)

    st.write("---")
    st.subheader("All Departments Comparison Table")
    comp_table = df.groupby('Department').agg(
        Count=('EmployeeID', 'count'),
        Avg_Salary=('AnnualSalary', 'mean'),
        Avg_Performance=('PerformanceScore', 'mean'),
        Avg_Weekly_Hours=('WeeklyHours', 'mean'),
        Avg_Projects=('ProjectsCompleted', 'mean')
    ).round(2)
    st.dataframe(comp_table, use_container_width=True)

# ==============================================================================
# SECTION 3: DATA VISUALIZATION (TASK 4)
# ==============================================================================
elif selected_page == "📊 Data Visualization (Task 4)":
    st.title("📊 Data Visualization Section")
    st.write("Visual exploration of salary trends, employee experience, and performance distributions.")

    v1, v2 = st.columns(2)

    with v1:
        st.write("### 1. Average Salary by Department")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Department', y='AnnualSalary', data=df, estimator='mean', errorbar=None, palette='Blues_r', ax=ax1)
        plt.xticks(rotation=25)
        ax1.set_ylabel("Mean Annual Salary (INR)")
        st.pyplot(fig1)

    with v2:
        st.write("### 2. Experience vs Annual Salary")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='ExperienceYears', y='AnnualSalary', hue='Department', data=df, alpha=0.7, ax=ax2)
        ax2.set_xlabel("Experience (Years)")
        ax2.set_ylabel("Annual Salary (INR)")
        st.pyplot(fig2)

    v3, v4 = st.columns(2)

    with v3:
        st.write("### 3. Weekly Hours vs Performance Score")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='WeeklyHours', y='PerformanceScore', hue='Gender', data=df, alpha=0.6, ax=ax3)
        ax3.set_xlabel("Weekly Hours")
        ax3.set_ylabel("Performance Score")
        st.pyplot(fig3)

    with v4:
        st.write("### 4. Salary Distribution by Gender")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='Gender', y='AnnualSalary', data=df, palette='Set2', ax=ax4)
        ax4.set_ylabel("Annual Salary (INR)")
        st.pyplot(fig4)

# ==============================================================================
# SECTION 4: MODEL PREDICTIONS (TASK 5)
# ==============================================================================
elif selected_page == "🔮 Model Predictions (Task 5)":
    st.title("🔮 Real-Time Salary Prediction Model")
    st.write("Dynamic machine learning model inference with input validation and error handling.")

    st.info(f"Model Status: {model_status}")

    st.subheader("Enter Employee Attributes:")
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        in_age = st.slider("Age", 18, 60, 30)
        in_exp = st.slider("Experience in Years", 0, 35, 5)
        in_perf = st.slider("Performance Score", 40, 100, 75)
        in_hours = st.slider("Weekly Hours", 30.0, 70.0, 48.0, 0.5)

    with p_col2:
        in_projects = st.slider("Projects Completed (Annual)", 1, 25, 8)
        in_sat = st.slider("Satisfaction Score (1-10)", 1.0, 10.0, 7.0, 0.1)
        in_sales = st.number_input("Monthly Sales (₹ Thousands)", 0.0, 1500.0, 150.0, 10.0)
        in_dept = st.selectbox("Department", df['Department'].unique())

    if st.button("🚀 Calculate Predicted Salary", use_container_width=True):
        input_data = pd.DataFrame([{
            'Age': in_age,
            'ExperienceYears': in_exp,
            'PerformanceScore': in_perf,
            'WeeklyHours': in_hours,
            'ProjectsCompleted': in_projects,
            'SatisfactionScore': in_sat,
            'MonthlySalesK': in_sales
        }])

        try:
            predicted_salary = ml_model.predict(input_data)[0]
            st.success(f"🎯 **Predicted Annual Salary:** ₹{predicted_salary:,.2f}")
            st.metric("Estimated Monthly Compensation", f"₹{predicted_salary/12:,.2f}")

            # Salary Band calculation
            if predicted_salary < 700000:
                tier = "Entry Band (< ₹700k)"
            elif predicted_salary <= 1200000:
                tier = "Mid-Level Band (₹700k - ₹1.2M)"
            else:
                tier = "Senior / Executive Band (> ₹1.2M)"
            st.info(f"**Assigned Compensation Tier:** {tier}")

        except Exception as err:
            st.error(f"Prediction Error: {err}")

# ==============================================================================
# SECTION 5: EMPLOYEE INSIGHTS & RECOMMENDATIONS
# ==============================================================================
elif selected_page == "👥 Employee Insights (Task 4 & 8)":
    st.title("👥 Employee Insights & HR Recommendations")
    st.write("Summary insights, top performer records, and strategic HR recommendations.")

    # High performers table
    st.subheader("⭐ High Performers List (Score >= 85)")
    high_perf = df[df['PerformanceScore'] >= 85][['EmployeeID', 'Age', 'Gender', 'Department', 'City', 'ExperienceYears', 'PerformanceScore', 'AnnualSalary', 'ProjectsCompleted']]
    st.dataframe(high_perf.head(15), use_container_width=True)

    st.write("---")

    st.subheader("💡 Strategic HR Decision Recommendations")
    st.markdown("""
    1. **Pay Parity & Experience Alignments:** Ensure mid-career professionals (8-12 years) receive clear compensation step adjustments.
    2. **Retention for Top Talent:** Fast-track appraisal reviews for employees with Performance ≥ 85 in lower salary quartiles.
    3. **Workload Ceilings:** Keep weekly hours balanced around 48–50 hours to maintain high job satisfaction.
    4. **Promotion Ladders:** Establish structured promotion pathways for junior associates into senior technical lead roles.
    """)

# ==============================================================================
# SECTION 6: CLUSTERING & PERSONAS (TASK 7)
# ==============================================================================
elif selected_page == "🧬 Clustering & Personas (Task 7)":
    st.title("🧬 Workforce Clustering & Personas")
    st.write("Unsupervised K-Means clustering segmentation and visual persona breakdowns.")

    c_img_path = "visualizations/task_7/annotated_decision_boundaries_2d.png"
    if os.path.exists(c_img_path):
        st.image(c_img_path, caption="Annotated 2D PCA Decision Boundaries & Workforce Zones", use_container_width=True)

    r_img_path = "visualizations/task_7/cluster_radar_profiles.png"
    if os.path.exists(r_img_path):
        st.image(r_img_path, caption="Cluster Persona Multi-Attribute Radar Profiles", use_container_width=True)

    summary_file = "data/cluster_profiles_summary.csv"
    if os.path.exists(summary_file):
        st.write("#### Cluster Persona Profile Table")
        st.dataframe(pd.read_csv(summary_file), use_container_width=True)
