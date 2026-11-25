FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" scikit-learn==1.9.1 joblib
COPY service_klasifikasi.py model_klasifikasi_desa.joblib ./
ENV PORT=8000
CMD ["sh", "-c", "uvicorn service_klasifikasi:app --host 0.0.0.0 --port ${PORT}"]
