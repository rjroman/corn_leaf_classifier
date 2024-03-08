from flask import Flask, render_template, request, send_from_directory
from torchvision import transforms
from PIL import Image
import os
import torch
import torchvision.models as models
import torch.nn as nn

app = Flask(__name__)

# Directory to store uploaded images
UPLOAD_FOLDER = 'images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained ResNet18 model
model_architecture = models.resnet18(pretrained=False)
num_ftrs = model_architecture.fc.in_features
model_architecture.fc = nn.Linear(num_ftrs, 4)  # Modify the output size to match the number of classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load state dictionary
model_path = 'K:/projects/image classifier using flask- dharnish,gowtham/custom model/resnet18_model.pth'
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model_architecture.load_state_dict(state_dict)
model_architecture.to(device)
model_architecture.eval()

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Mapping of class indices to class names
class_labels = {
    0: 'Blight',
    1: 'Common_Rust',
    2: 'Gray_Leaf_Spot',
    3: 'Healthy'
    # Add more class labels as needed
}

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No image uploaded")

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('index.html', prediction="No image selected")

    # Save the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    # Preprocess the image
    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Move input data to GPU
    img = img.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model_architecture(img)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]

    # Convert tensor back to PIL Image
    img = transforms.functional.to_pil_image(img.squeeze(0).cpu())

    # Generate filename for output image
    output_image = 'output_' + image_file.filename  # Modify as needed
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_image)

    # Save the output image
    img.save(output_image_path)

    return render_template('output.html', input_image=image_path, output_image=output_image, prediction=predicted_class)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
