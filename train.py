"""
TA-03 Pembelajaran Mesin
Prediksi Risiko Gagal Bayar Kartu Kredit dengan Regresi Logistik
Jalankan:  python train.py     (butuh credit.csv)
"""
# %% [markdown]
# # TA-03: Prediksi Risiko Gagal Bayar Kartu Kredit (Regresi Logistik)

# %% 1. Import & konfigurasi
import json
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
THRESHOLD = 0.5          # ambang keputusan (turunkan, mis. 0.4, agar Recall naik)
DATA_PATH = "credit.csv"

NUM_FEATURES = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "BILL_AMT1", "PAY_AMT1"]
CAT_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]


def done(fig, name):
    """Simpan grafik ke figures/, tampilkan jika di notebook."""
    fig.tight_layout()
    fig.savefig(f"figures/{name}", dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)


# %% 2. Muat data
df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"PAY_1": "PAY_0",
                        "default.payment.next.month": "default",
                        "default payment next month": "default",
                        "dpnm": "default"})
print("--- 5 Baris Pertama ---")
print(df.head())
print("\n--- Info ---")
df.info()
print("\nMissing values:", int(df.isna().sum().sum()))
print("Baris duplikat :", int(df.duplicated().sum()))

# %% 3. Cleaning
n0 = len(df)
df = df.drop(columns=[c for c in ["ID"] if c in df.columns])
df = df.drop_duplicates().dropna()
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
print(f"Cleaning: {n0} -> {len(df)} baris")

# %% 4. EDA
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="default", data=df, ax=ax)
ax.set_xticks([0, 1], ["Tidak Gagal Bayar (0)", "Gagal Bayar (1)"])
ax.set_title("Distribusi Target")
ax.set_xlabel("")
done(fig, "01_distribusi_target.png")
print("Proporsi kelas:\n", df["default"].value_counts(normalize=True).round(3))

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, col in zip(axes.ravel(), NUM_FEATURES):
    sns.histplot(df[col], bins=30, ax=ax)
    ax.set_title(col)
done(fig, "02_histogram_fitur.png")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sns.boxplot(x="default", y="LIMIT_BAL", data=df, ax=axes[0])
sns.countplot(x="PAY_0", hue="default", data=df, ax=axes[1])
sns.boxplot(x="default", y="AGE", data=df, ax=axes[2])
axes[0].set_title("Batas kredit vs default")
axes[1].set_title("Riwayat bayar terakhir (PAY_0) vs default")
axes[2].set_title("Usia vs default")
done(fig, "03_fitur_vs_target.png")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df[NUM_FEATURES + ["default"]].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", ax=ax)
ax.set_title("Heatmap Korelasi")
done(fig, "04_korelasi.png")

# %% 5. Persiapan fitur (one-hot untuk kategorikal)
X = pd.get_dummies(df[NUM_FEATURES + CAT_FEATURES], columns=CAT_FEATURES,
                   drop_first=True, dtype=float)
y = df["default"].astype(int)
FEATURE_COLUMNS = list(X.columns)
print("Kolom fitur:", FEATURE_COLUMNS)

# %% 6. Split -> Scaling -> SMOTE  (urutan penting agar tidak ada data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

scaler = StandardScaler().fit(X_train)            # fit HANYA di data latih
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

smote = SMOTE(random_state=RANDOM_STATE)          # SMOTE HANYA di data latih
X_train_res, y_train_res = smote.fit_resample(X_train_s, y_train)
print("Data latih sebelum SMOTE:", np.bincount(y_train))
print("Data latih setelah SMOTE:", np.bincount(y_train_res))

# %% 7. Latih model
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_res, y_train_res)

model_cw = LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=RANDOM_STATE).fit(X_train_s, y_train)

# %% 8. Evaluasi
proba = model.predict_proba(X_test_s)[:, 1]
pred = (proba >= THRESHOLD).astype(int)

print("\n--- Classification Report (SMOTE) ---")
print(classification_report(y_test, pred, target_names=["Tidak Berisiko", "Gagal Bayar"]))

cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Tidak Berisiko", "Gagal Bayar"]).plot(
    cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix (data uji)")
done(fig, "05_confusion_matrix.png")

auc = roc_auc_score(y_test, proba)
fpr, tpr, _ = roc_curve(y_test, proba)
print(f"AUC = {auc:.4f}")
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="orange", label=f"Regresi Logistik (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="darkblue", label="Tebakan acak")
ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR / Recall)")
ax.set_title("Kurva ROC")
ax.legend()
ax.grid(True)
done(fig, "06_roc_curve.png")

model_plain = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X_train_s, y_train)
rows = []
for name, m in [("Tanpa penyeimbangan", model_plain),
                ("class_weight='balanced'", model_cw),
                ("SMOTE (dipakai)", model)]:
    p = m.predict_proba(X_test_s)[:, 1]
    yp = (p >= THRESHOLD).astype(int)
    rows.append({"Pendekatan": name,
                 "Accuracy": accuracy_score(y_test, yp),
                 "Precision": precision_score(y_test, yp, zero_division=0),
                 "Recall": recall_score(y_test, yp),
                 "F1": f1_score(y_test, yp),
                 "AUC": roc_auc_score(y_test, p)})
print("\n--- Perbandingan pendekatan ---")
print(pd.DataFrame(rows).round(3).to_string(index=False))

thr_rows = []
for t in [0.3, 0.4, 0.5, 0.6]:
    yp = (proba >= t).astype(int)
    thr_rows.append({"Threshold": t,
                     "Precision": precision_score(y_test, yp, zero_division=0),
                     "Recall": recall_score(y_test, yp),
                     "F1": f1_score(y_test, yp)})
print("\n--- Pengaruh threshold ---")
print(pd.DataFrame(thr_rows).round(3).to_string(index=False))

coef = pd.Series(model.coef_[0], index=FEATURE_COLUMNS).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
coef.plot.barh(ax=ax)
ax.set_title("Koefisien Regresi Logistik (fitur terstandarisasi)")
done(fig, "07_koefisien.png")

# %% 9. Simpan model (Joblib) + ringkasan metrik
joblib.dump(model, "model/model_logreg.joblib")
joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(FEATURE_COLUMNS, "model/columns.joblib")

metrics = {
    "n_rows_clean": int(len(df)),
    "default_rate": float(y.mean()),
    "threshold": THRESHOLD,
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred)),
    "f1": float(f1_score(y_test, pred)),
    "auc": float(auc),
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "limits": {c: [float(df[c].min()), float(df[c].max())] for c in NUM_FEATURES},
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nModel, scaler, kolom, dan metrics.json tersimpan di folder model/")