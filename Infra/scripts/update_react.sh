#!/bin/bash

echo "Updating React UI..."
cd /home/ec2-user/UI/classifierui
git pull origin main
npm install
npm run build
sudo systemctl restart nginx
