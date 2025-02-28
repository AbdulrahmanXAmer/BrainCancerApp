provider "aws" {
  region = "us-east-1"
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
}

resource "aws_security_group" "allow_http_https" {
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "flask_api" {
  ami                    = "ami-0c55b159cbfafe1f0"
  instance_type          = "t3.medium"
  subnet_id              = aws_subnet.public.id
  security_groups        = [aws_security_group.allow_http_https.name]
  iam_instance_profile   = aws_iam_instance_profile.ssm_profile.name

  user_data = <<EOF
#!/bin/bash
apt update && apt install -y docker.io docker-compose
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker
git clone https://github.com/YOUR_GITHUB_USER/ModelHosting.git /home/ubuntu/app
cd /home/ubuntu/app
docker-compose up -d
EOF
}

resource "aws_instance" "react_ui" {
  ami                    = "ami-0c55b159cbfafe1f0"
  instance_type          = "t3.small"
  subnet_id              = aws_subnet.public.id
  security_groups        = [aws_security_group.allow_http_https.name]
  iam_instance_profile   = aws_iam_instance_profile.ssm_profile.name

  user_data = <<EOF
#!/bin/bash
apt update && apt install -y docker.io docker-compose nginx
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker
git clone https://github.com/YOUR_GITHUB_USER/UI.git /home/ubuntu/ui
cd /home/ubuntu/ui
docker-compose up -d
rm /etc/nginx/sites-enabled/default
cat <<EOT > /etc/nginx/sites-available/myfuturedomain.com
server {
    listen 80;
    server_name myfuturedomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
EOT
ln -s /etc/nginx/sites-available/myfuturedomain.com /etc/nginx/sites-enabled/
systemctl restart nginx
EOF
}

resource "aws_iam_instance_profile" "ssm_profile" {
  name = "SSMInstanceProfile"
  role = aws_iam_role.ssm_role.name
}

resource "aws_iam_role" "ssm_role" {
  name = "SSMRole"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.ssm_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
