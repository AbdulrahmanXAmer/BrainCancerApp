#!/bin/bash

echo "Restarting React UI..."
cd /home/ec2-user/UI/classifierui
docker-compose restart
sudo systemctl restart nginx

echo "React UI restarted successfully!"
