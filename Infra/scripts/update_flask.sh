#!/bin/bash

echo "Updating Flask API..."
cd /home/ec2-user/ModelHosting
git pull origin main
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
docker-compose up --build -d
