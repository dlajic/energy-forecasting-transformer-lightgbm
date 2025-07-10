#!/bin/bash

# Startet docker-compose im Hintergrund
docker-compose up --build &

# Kleiner Delay, damit der Server Zeit zum Starten hat
sleep 3

# Öffnet den Standardbrowser auf localhost:8501
xdg-open http://localhost:8501