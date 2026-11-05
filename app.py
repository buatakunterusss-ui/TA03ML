"""
Aplikasi web Flask: Prediksi Risiko Gagal Bayar Kartu Kredit (Regresi Logistik).
Jalankan lokal:  python app.py   -> http://127.0.0.1:5000
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

model = joblib.load(os.path.join(MODEL_DIR, "model_logreg.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
COLUMNS = joblib.load(os.path.join(MODEL_DIR, "columns.joblib"))
with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

THRESHOLD = METRICS["threshold"]
LIMITS = METRICS["limits"]

NUM_FIELDS = [
    ("LIMIT_BAL", "Batas kredit (NT$)", "Contoh: 50000"),
    ("AGE", "Usia (tahun)", "Contoh: 35"),
    ("PAY_0", "Status bayar bulan terakhir (PAY_0)",
     "-1 = lunas tepat waktu, 0 = bayar minimum, 1..8 = telat 1..8 bulan"),
    ("PAY_2", "Status bayar 2 bulan lalu (PAY_2)", "Skala sama dengan PAY_0"),
    ("BILL_AMT1", "Tagihan bulan terakhir (NT$)", "Contoh: 20000"),
    ("PAY_AMT1", "Pembayaran bulan terakhir (NT$)", "Contoh: 2000"),
]
SEX_OPTIONS = {1: "Laki-laki", 2: "Perempuan"}
EDU_OPTIONS = {1: "Pascasarjana", 2: "Universitas", 3: "SMA", 4: "Lainnya"}
MAR_OPTIONS = {1: "Menikah", 2: "Lajang", 3: "Lainnya"}


def parse_form(form):
    """Validasi input. Return (dict_nilai, daftar_error)."""
    values, errors = {}, []
    for key, label, _ in NUM_FIELDS:
        raw = str(form.get(key) or "").strip().replace(",", ".")
        if raw == "":
            errors.append(f"{label} wajib diisi.")
            continue
        try:
            v = float(raw)
        except ValueError:
            errors.append(f"{label} harus berupa angka.")
            continue
        if not np.isfinite(v):
            errors.append(f"{label} harus berupa angka.")
            continue
        lo, hi = LIMITS[key]
        if v < lo or v > hi:
            errors.append(f"{label} harus antara {lo:g} dan {hi:g} "
                          "(rentang data latih).")
            continue
        values[key] = v
    for key, options, label in [("SEX", SEX_OPTIONS, "Jenis kelamin"),
                                ("EDUCATION", EDU_OPTIONS, "Pendidikan"),
                                ("MARRIAGE", MAR_OPTIONS, "Status pernikahan")]:
        try:
            v = int(form.get(key, ""))
        except (ValueError, TypeError):
            v = None
        if v not in options:
            errors.append(f"{label} tidak valid.")
        else:
            values[key] = v
    return values, errors


def build_row(values):
    """Susun 1 baris fitur dengan urutan kolom yang SAMA seperti saat training."""
    row = {c: 0.0 for c in COLUMNS}
    for key, _, _ in NUM_FIELDS:
        row[key] = values[key]
    for cat in ("SEX", "EDUCATION", "MARRIAGE"):
        dummy = f"{cat}_{values[cat]}"
        if dummy in row:          # kategori acuan (drop_first) = semua dummy 0
            row[dummy] = 1.0
    return pd.DataFrame([row], columns=COLUMNS)


def predict(values):
    X = scaler.transform(build_row(values))
    p = float(model.predict_proba(X)[0, 1])
    return p, p >= THRESHOLD


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"num_fields": NUM_FIELDS, "sex_options": SEX_OPTIONS,
           "edu_options": EDU_OPTIONS, "mar_options": MAR_OPTIONS,
           "metrics": METRICS, "limits": LIMITS, "form": {},
           "errors": [], "result": None}
    if request.method == "POST":
        ctx["form"] = request.form
        values, errors = parse_form(request.form)
        if errors:
            ctx["errors"] = errors
        else:
            p, risky = predict(values)
            ctx["result"] = {
                "label": "Berisiko Gagal Bayar" if risky else "Tidak Berisiko",
                "risky": risky,
                "prob_default": round(p * 100, 1),
                "prob_safe": round((1 - p) * 100, 1),
            }
    return render_template("index.html", **ctx)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint JSON opsional."""
    data = request.get_json(silent=True) or request.form
    values, errors = parse_form(data)
    if errors:
        return jsonify({"errors": errors}), 400
    p, risky = predict(values)
    return jsonify({"prediction": "Berisiko Gagal Bayar" if risky else "Tidak Berisiko",
                    "probability_default": round(p, 4)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)