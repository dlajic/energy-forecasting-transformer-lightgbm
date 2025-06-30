docker-compose up --build -d
Start-Sleep -Seconds 2
Start-Process "http://localhost:8501"

