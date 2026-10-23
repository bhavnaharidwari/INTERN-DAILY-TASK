"""Generates the synthetic SalaryInsight dataset (employee_data.csv)."""
import numpy as np, pandas as pd

def generate(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    dept = rng.choice(["Engineering","Sales","HR","Finance","Marketing","Operations"], n,
                      p=[.30,.20,.08,.12,.12,.18])
    edu = rng.choice(["Bachelors","Masters","PhD","Diploma"], n, p=[.50,.30,.07,.13])
    city = rng.choice(["Tier1","Tier2","Tier3"], n, p=[.45,.35,.20])
    age = rng.integers(22, 60, n)
    exp = np.clip(age - 22 - rng.integers(0, 6, n), 0, None)
    perf = np.clip(rng.normal(3.4, .8, n), 1, 5).round(1)
    hours = np.clip(rng.normal(44, 6, n), 30, 70).round(0)
    train = rng.integers(0, 80, n)
    certs = rng.poisson(1.2, n)
    wlb = np.clip(rng.normal(3.3, 1.0, n), 1, 5).round(1)   # work-life balance
    d_mult = pd.Series(dept).map({"Engineering":1.25,"Sales":1.0,"HR":.85,"Finance":1.15,
                                  "Marketing":.95,"Operations":.9}).values
    e_add = pd.Series(edu).map({"Diploma":0,"Bachelors":2,"Masters":4.5,"PhD":8}).values
    c_mult = pd.Series(city).map({"Tier1":1.2,"Tier2":1.0,"Tier3":.82}).values
    base = (4 + e_add + exp*0.55 + perf*1.1 + certs*0.6 + train*0.02)  # LPA
    salary = base * d_mult * c_mult * rng.lognormal(0, .08, n)
    # non-linear bonus for senior high performers
    salary += np.where((exp > 12) & (perf > 4), 5, 0)
    salary = salary.round(2)
    # satisfaction (high=1 / low=0): driven by pay-vs-peers, WLB, hours, performance recognition
    z = (0.08*(salary - salary.mean())/salary.std()*10 + 0.9*(wlb-3.3) - 0.07*(hours-44)
         + 0.25*(perf-3.4) + 0.2*(city=="Tier1") + rng.normal(0, .9, n) - 1.1)  # shift -> imbalance
    satisfied = (z > 0).astype(int)
    df = pd.DataFrame(dict(Age=age, Department=dept, Education=edu, City_Tier=city,
        Experience_Years=exp, Performance_Rating=perf, Weekly_Hours=hours, Training_Hours=train,
        Certifications=certs, Work_Life_Balance=wlb, Salary_LPA=salary,
        Satisfaction=np.where(satisfied==1,"High","Low")))
    # inject a few missing values for realism
    for col in ["Performance_Rating","Training_Hours"]:
        df.loc[rng.choice(n, 25, replace=False), col] = np.nan
    return df

if __name__ == "__main__":
    df = generate(); df.to_csv("employee_data.csv", index=False)
    print(df.shape); print(df.Satisfaction.value_counts(normalize=True)); print(df.describe().T)
