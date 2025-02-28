import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Define the same CNN model architecture
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

        self.dropout = nn.Dropout(0.5)  
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)  
        return x

# Load the model
MODEL_PATH = "/Users/abdulrahman/Desktop/Projects/ModelTraining/saved_models/brain_cancer_classifier.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel2(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])



@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    # Convert prediction to label
    label = "Tumor" if prediction == 1 else "Healthy"

    print(label)
    response = jsonify({"prediction": label})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST")

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
