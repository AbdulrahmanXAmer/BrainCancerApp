#!/bin/bash

echo "Updating system and installing dependencies..."
sudo yum update -y
sudo yum install -y python3.13 python3-pip docker docker-compose nginx

echo "Starting Docker service..."
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

echo "Setting up Flask API..."
cd /home/ec2-user/ModelHosting
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn

echo "Configuring Gunicorn service..."
cat <<EOT | sudo tee /etc/systemd/system/gunicorn.service
[Unit]
Description=Gunicorn instance to serve Flask API
After=network.target

[Service]
User=ec2-user
Group=ec2-user
WorkingDirectory=/home/ec2-user/ModelHosting
Environment=\"PATH=/home/ec2-user/ModelHosting/venv/bin\"
ExecStart=/home/ec2-user/ModelHosting/venv/bin/gunicorn --workers 4 --bind 0.0.0.0:5000 app:app

[Install]
WantedBy=multi-user.target
EOT

echo "Starting Gunicorn..."
sudo systemctl daemon-reload
sudo systemctl enable gunicorn
sudo systemctl start gunicorn

echo "Configuring NGINX..."
sudo rm -f /etc/nginx/nginx.conf
cat <<EOT | sudo tee /etc/nginx/nginx.conf
events {}

http {
    server {
        listen 80;
        server_name myfuturedomain.com;

        location / {
            proxy_pass http://127.0.0.1:5000;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        }
    }
}
EOT

echo "Starting NGINX..."
sudo systemctl enable nginx
sudo systemctl restart nginx

echo "Flask API setup completed!"
