# ---- Base Image ----
FROM python:3.10-slim

# ---- Set workdir ----
WORKDIR /app

# ---- Install system deps ----
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && apt-get clean

# ---- Install Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy project ----
COPY . .

# ---- Expose Streamlit port ----
EXPOSE 8501

# ---- Run Streamlit ----
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
