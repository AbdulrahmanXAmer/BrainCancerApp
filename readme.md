# **Brain Cancer Classification - Full Stack AI project and Deployment with IaC and Docker**

## **Overview**
This project is a **full-stack AI-powered brain cancer classification system**. It consists of:
- **Frontend:** React UI for uploading brain scans and interacting with the model. Two sample images are provided for the UX experience of first time users.
- **Backend:** Flask API serving a deep learning model for classification. Note this can easily be replaced with a sagemaker serverless endpoint to avoid provisioning an EC2 instance. However -- this is a quick and dirty approach for showcasing my skills across the tech stack.
- **Infrastructure:** Terraform and AWS Systems Manager for cloud deployment and ease of deployment.
- **Machine Learning:** Jupyter notebook for model training -- can be adapted to a python script or run in sage maker studio prior to deploying via sagemaker serverless.

---
## **Directory Structure**

```
.
├── Infra                # Infrastructure as Code (Terraform, Docker, Scripts)
│   ├── .github          # GitHub Actions (Demo Only)
│   ├── docker           # Dockerfiles for Flask & React
│   ├── scripts          # Deployment scripts (helper scripts for now for anyone new to this)  
│   ├── terraform        # Terraform configurations
│
├── ml_env               # Python virtual environment (not committed to Git)
│
├── ModelHosting         # Flask API to serve ML model
│   ├── app.py           # Main Flask application
│
├── ModelTraining        # Jupyter notebook & dataset for model training
│   ├── Dataset          # Training dataset
│   ├── saved_models     # Trained ML models (will omit, build it on your own)
│   ├── BrainCancerImagingModel.ipynb  # Model training notebook
│
├── UI/classifierui      # React frontend application
│   ├── public           # Public assets
│   ├── src              # Source code
│   │   ├── components   # UI components
│   │   │   ├── images.js
│   │   │   ├── tumorClassifier.js
```
---
### NOTES
- Flask when deployed in this way relies on NGINX and Gunicorn. NGINX is a reverse proxy and gunicorn a WISGI which are both sevices that need to be configured in order for the API to open to the internet. 
- When deploying a react project, you want to run build and point your build folder appropriately via NGINX so there is visibility to your project on the website. 
- The app will not support HTTPS as it must be configured after you purchase a domain. 
- Route 53 is not neccesssary for your app so long as you configure your domains A name to match the public IP of your instance which can be found in AWS.
- Skip to the end of the notebook to understand chosen metrics for model evaluation and what changes were made in order to achieve this. 
---

## **Installation & Setup**

** Run the Jupyer Notebook in ModelTraining as the Flask application relies on this -- for sake of space in github I have ommitted the folder but you can easily create it on your own **

### **1️⃣ Local Development**
#### **Backend (Flask API)**

```bash
cd ModelHosting
python -m venv ml_env
source ml_env/bin/activate
pip install -r requirements.txt #Copy from the docker folder if you need to build it here
python app.py
```
- Runs on `http://127.0.0.1:5000`

#### **Frontend (React UI)**
```bash
cd UI/classifierui
npm install #Collects all packages
npm start
```
- Runs on `http://localhost:3000`

### **2️⃣ Docker Setup**
#### **Build & Run Flask API**
```bash
cd Infra/docker/flask
docker-compose up --build -d
```

#### **Build & Run React UI**
```bash
cd Infra/docker/react
docker-compose up --build -d
```

### **3️⃣ Deploy to AWS (Terraform & Systems Manager)**
#### **Provision Infrastructure**
```bash
cd Infra/terraform
terraform init
terraform apply
```
#### **Deploy Using AWS Systems Manager**
```bash
aws ssm send-command \
    --document-name "AWS-RunShellScript" \
    --targets "Key=instance-id,Values=FLASK_INSTANCE_ID" \
    --parameters 'commands=["cd /home/ubuntu/app && git pull && docker-compose up -d"]' \
    --region us-east-1
```
---

## **How it Works**
1. **Users upload a brain scan** via React UI
2. **Image is sent to Flask API**, which loads a trained deep learning model
3. **AI model classifies** the image as **Tumor** or **Healthy**
4. **Results are displayed** in the React frontend

## **Tech Stack**
- **Machine Learning:** PyTorch, OpenCV (for future implementation)
- **Backend:** Flask, Gunicorn, Docker
- **Frontend:** React, BulmaCSS, Axios, Docker
- **Infrastructure:** Terraform, AWS EC2, AWS Systems Manager, NGINX (future potential for sagemaker / ECR)

## **Next Steps**
- Automate HTTPS setup with Let’s Encrypt
- Add monitoring/logging for AI inference
- Deploy with AWS Fargate (Serverless Containers)

--- 
## **Deep Dive Into Model Training**
- The jupyter notebook relies on requirements from requirements.txt 
- `TumorDataset` class is used to load in data and label it accordingly. It workes with `load_images_from_folders` and was made robust by hadnling both png and jpg as a result of future iterations. 
- The photos are loaded in and are split into a train/validation/test sets. 
- A class for preprocessing is initialized `ImagePreprocessor`. This class relies on several image transformation and preprocessing steps and was made to take in several `**Kwargs` as avehicle to make it easy to try tuning the data in different ways when feeding models (for future implementations)
- `Visualize_samples` and `visualize_raw_samples` are two functions used to see how the data looks before and after transformation which I provided as a vehicle to help see differences in contrast/ rotation etc. 
- Originally, a `CNN` I built was trained. It looked good at first, but the issue is how it performed on unseen data. 
Accuracy: 0.8667
Precision: 1.0000
Recall (Sensitivity): 0.7288
F1 Score: 0.8431
F0.5 Score (Precision-focused): 0.9307
F2 Score (Recall-focused): 0.7706

These are really alarming scores. For a project like this, we would rather lean towards higher Recall and a Higher F2 score as this means we would increase the chance of pulling False Positives. 

- We take advantage of our processor class to add complexity to our data prior to training the model 
```python 
preprocessor = ImagePreprocessor(
    grayscale=True, 
    resize=(224, 224), 
    horizontal_flip=True, 
    rotation=30,  # Increased rotation
    normalize={"mean": [0.5], "std": [0.5]}, 
    brightness=0.4,  # Increased brightness variation
    contrast=0.4,  # Increased contrast variation
)
```

This complexity includes higher resolution, increased brightness and contrast, more flips and rotations etc. The variation in input images causes the model to learn and detect more subtle nuances. 


And then we follow up by adding complexity in our CNN2:

```python
class CNNModel2(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.dropout = nn.Dropout(0.5)  # Helps generalization

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
```
Where we add a third convultional Layer making the model a deeper model which captures more complex features, a larger kernel size in the first two layers to help capture borader patterns. Adaptive global pooling layer to add flexibvility for different input sizes and adding Dropout layers to reduce overfitting and improving the recall on unseen data. 

With these simple adjustments  we see an increase in Recall to almost 92% and an F2 score at 90% for a way better performing model (on a second attempt only).

Of course there are ways to make this model even better! For now -- I kept it simple and sweet to focus on the full project from conception to production. 
