#!/bin/bash

echo "Updating system and installing dependencies..."
sudo yum update -y
sudo yum install -y nodejs npm docker docker-compose nginx

echo "Starting Docker service..."
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

echo "Setting up React UI..."
cd /home/ec2-user/UI/classifierui
npm install
npm run build

echo "Configuring NGINX..."
sudo rm -f /etc/nginx/nginx.conf
cat <<EOT | sudo tee /etc/nginx/nginx.conf
events {}

http {
    server {
        listen 80;
        server_name myfuturedomain.com;

        location / {
            root /home/ec2-user/UI/classifierui/build;
            index index.html;
            try_files \$uri /index.html;
        }
    }
}
EOT

sudo systemctl enable nginx
sudo systemctl restart nginx
