# Task 2 - report on how the salary models performed (reads task1_results.csv)
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

os.makedirs("task2_outputs", exist_ok=True)
r = pd.read_csv("task1_results.csv").set_index("Model")
order = r.sort_values("Test_RMSE").index
names = [m.replace("Decision Tree (deep)", "DT (deep)").replace("Decision Tree (depth=5)", "DT (d=5)")
          .replace("Linear Regression", "Lin. Reg.").replace("Gradient Boosting", "Grad. Boost")
          .replace("Random Forest", "Rand. Forest") for m in order]
hl = {"Random Forest": "#d95f02", "Gradient Boosting": "#1b9e77"}
cols = [hl.get(m, "#7570b3") for m in order]

# charts
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].barh(names[::-1], r.loc[order, "Test_RMSE"][::-1], color=cols[::-1])
ax[0].set_title("Test RMSE (lower is better)")
ax[1].barh(names[::-1], r.loc[order, "Test_R2"][::-1], color=cols[::-1])
ax[1].set_xlim(0.7, 1)
ax[1].set_title("Test R2 (higher is better)")
plt.tight_layout(); plt.savefig("task2_outputs/c1.png", dpi=160); plt.close()

plt.figure(figsize=(7.5, 4))
plt.bar(names, r.loc[order, "CV_R2_std"], color=cols)
plt.title("Std-dev of 10-fold CV R2 (lower = more stable)")
plt.xticks(rotation=25, fontsize=8)
plt.tight_layout(); plt.savefig("task2_outputs/c2.png", dpi=160); plt.close()

x = np.arange(len(order))
plt.figure(figsize=(7.5, 4))
plt.bar(x - 0.19, r.loc[order, "Train_RMSE"], 0.38, label="Train RMSE")
plt.bar(x + 0.19, r.loc[order, "Test_RMSE"], 0.38, label="Test RMSE")
plt.xticks(x, names, rotation=25, fontsize=8)
plt.legend(); plt.title("Train vs test RMSE (LPA)")
plt.tight_layout(); plt.savefig("task2_outputs/c3.png", dpi=160); plt.close()

# pdf
ss = getSampleStyleSheet()
H = ParagraphStyle("H", parent=ss["Heading1"], textColor=colors.HexColor("#1f3a5f"), spaceBefore=10)
B = ParagraphStyle("B", parent=ss["BodyText"], leading=15, fontSize=10)
Bl = ParagraphStyle("Bl", parent=B, leftIndent=14, bulletIndent=4)

rf, gb, lr, dt = (r.loc[m] for m in ["Random Forest", "Gradient Boosting", "Linear Regression", "Decision Tree (deep)"])
cut_rf = 100 * (1 - gb.Test_RMSE / rf.Test_RMSE)
cut_lr = 100 * (1 - gb.Test_RMSE / lr.Test_RMSE)

story = [
    Paragraph("SalaryInsight: Predictive Modeling for Informed Salary Predictions", ss["Title"]),
    Paragraph("Model performance report", ss["Heading3"]),
    Paragraph("I used a made-up employee dataset with 2,000 rows and 10 input columns (age, department, education, "
              "city tier, experience, performance rating, weekly hours, training hours, certifications, work-life "
              "balance). The thing being predicted is salary in LPA. 20% of the data was kept aside for testing and "
              "I used 10-fold cross-validation on the rest.", B),
    Paragraph("1. Summary", H),
    Paragraph(f"I compared seven regression models. Gradient Boosting did best on the test data "
              f"(RMSE {gb.Test_RMSE:.2f} LPA, R<super>2</super> {gb.Test_R2:.3f}). Random Forest "
              f"(RMSE {rf.Test_RMSE:.2f}, R<super>2</super> {rf.Test_R2:.3f}) was a lot better than a single decision tree "
              f"(RMSE {dt.Test_RMSE:.2f}) but only about equal to the linear models (Linear Regression RMSE {lr.Test_RMSE:.2f}). "
              "So my pick is Gradient Boosting, and I would keep Linear or Ridge Regression as a simple model to compare against.", B),
    Paragraph("2. Results", H),
]

head = ["Model", "CV R2 (mean +/- std)", "Train RMSE", "Test RMSE", "Test MAE", "Test R2"]
rows = [head]
for m in order:
    rows.append([m, f"{r.loc[m,'CV_R2_mean']:.3f} +/- {r.loc[m,'CV_R2_std']:.3f}", f"{r.loc[m,'Train_RMSE']:.2f}",
                 f"{r.loc[m,'Test_RMSE']:.2f}", f"{r.loc[m,'Test_MAE']:.2f}", f"{r.loc[m,'Test_R2']:.3f}"])
t = Table(rows, repeatRows=1, colWidths=[4.2*cm, 3.6*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm])
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                       ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                       ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                       ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#d9f0e6"))]))
story += [t, Spacer(1, 4), Paragraph("<i>Green row is the best model on the test data. Errors are in LPA.</i>", B),
          Spacer(1, 6), Image("task2_outputs/c1.png", width=17*cm, height=6.2*cm),
          Paragraph("3. What I found", H),
          Paragraph(f"<b>Accuracy:</b> Gradient Boosting has about {cut_rf:.0f}% lower test error than Random Forest and "
                    f"about {cut_lr:.0f}% lower than Linear Regression.", Bl, bulletText="•"),
          Paragraph(f"<b>Stability:</b> Gradient Boosting varied the least between folds (std {gb.CV_R2_std:.3f}). Random Forest "
                    f"({rf.CV_R2_std:.3f}) is about the same as the linear models ({lr.CV_R2_std:.3f}). The deep decision tree "
                    f"was the least stable ({dt.CV_R2_std:.3f}).", Bl, bulletText="•"),
          Paragraph(f"<b>Bias and variance:</b> the deep tree basically memorises the training data (train RMSE about 0, test {dt.Test_RMSE:.2f}), "
                    f"so it has high variance. The linear models have almost no gap between train and test but a higher error "
                    f"overall, probably because they miss non-linear effects like the bonus for senior high performers. "
                    f"Random Forest fixes most of the tree's variance, but its train/test gap ({rf.Overfit_Gap:.2f}) is still bigger "
                    f"than boosting's ({gb.Overfit_Gap:.2f}) and its test error is worse.", Bl, bulletText="•"),
          Paragraph("<b>Why Random Forest did not win:</b> salary in this data mostly goes up in a fairly straight line with experience, "
                    "education and rating, which linear models handle well. Boosting then fixes the leftover non-linear part. "
                    "My Random Forest was only lightly tuned, so tuning might close some of the gap.", Bl, bulletText="•"),
          PageBreak(), Paragraph("4. Stability and bias-variance charts", H),
          Image("task2_outputs/c2.png", width=13*cm, height=6.9*cm), Spacer(1, 6),
          Image("task2_outputs/c3.png", width=13*cm, height=6.9*cm),
          Paragraph("5. Recommendation", H),
          Paragraph("Use <b>Gradient Boosting</b>. It has the lowest error, the most stable results and a fairly small train/test gap.", Bl, bulletText="1."),
          Paragraph("Keep <b>Ridge or Linear Regression</b> as an easy-to-explain baseline. It is only about 0.5 LPA worse in RMSE.", Bl, bulletText="2."),
          Paragraph("Use <b>Random Forest</b> mostly to look at feature importance, and tune max_depth, min_samples_leaf and max_features first.", Bl, bulletText="3."),
          Paragraph("Don't use a single unpruned decision tree, it overfits badly.", Bl, bulletText="4."),
          Paragraph("Next I would tune Gradient Boosting with a randomized search and try the models on real HR data. "
                    "The data here is synthetic, so the exact numbers would be different on real data.", B),
          Paragraph("6. Notes on method", H),
          Paragraph("Missing values were filled with the median, numeric columns were scaled and text columns were one-hot encoded, "
                    "all inside a pipeline so nothing leaks from the test data. Stability is the spread of R<super>2</super> across the 10 folds. "
                    "For bias and variance I compared train and test error and looked at the learning curves from Task 1.", B)]
SimpleDocTemplate("Task2_Model_Performance_Report.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                  topMargin=1.8*cm, bottomMargin=1.8*cm, title="SalaryInsight - Model Performance Report").build(story)
print("done")
