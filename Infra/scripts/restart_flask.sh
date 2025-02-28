#!/bin/bash

echo "Restarting Flask API..."
cd /home/ec2-user/ModelHosting
docker-compose restart

echo "Flask API restarted successfully!"
