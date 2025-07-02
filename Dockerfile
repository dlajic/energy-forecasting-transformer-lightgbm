#Basis-Image mit Python
FROM python:3.10-slim

#Arbeitsverzeichnis im Container
WORKDIR /app

# Git installieren (für GitHub-Pakete)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

#Requirements installieren
COPY requirements.txt .
RUN pip install -r requirements.txt #for production insert: --no-cache-dir before -r

#Pythonpath to tell its modular
ENV PYTHONPATH="/app"

#Start App
CMD ["streamlit", "run", "streamlit_simulation/app.py", "--server.port=8501", "--server.address=0.0.0.0"]