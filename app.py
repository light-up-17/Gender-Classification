import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'GCM_ep4_f1_0.9577.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architecture - SAME as training code
class GenderClassifier(nn.Module):
    """
    Gender classification model using an EfficientNetV2-S backbone.
    """
    def __init__(self):
        super().__init__()
        # load EfficientNetV2-S pretrained (SAME as training)
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=False)
        in_features = self.backbone.classifier.in_features
        # SAME classifier architecture as training
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x)

# Load model
print(f"Loading model from {MODEL_PATH}...")
print(f"Using device: {DEVICE}")
model = GenderClassifier().to(DEVICE)

# Load the trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")

# Image transforms - SAME as validation transform in training code
def get_transforms():
    return A.Compose([
        A.Resize(256, 256), 
        A.CenterCrop(224, 224), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def predict_gender(image_bytes):
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Convert to numpy array for albumentations
        image_np = np.array(image)
        
        # Apply transforms (SAME as validation)
        transform = get_transforms()
        transformed = transform(image=image_np)
        input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            # Model outputs a single logit (not 2 classes)
            output = model(input_tensor).squeeze()
            
            # Apply sigmoid to get probability
            probability = torch.sigmoid(output).item()
            
            # Convert to class: probability > 0.5 means female (class 1)
            # According to training: 0 = Male, 1 = Female
            predicted_class = 1 if probability > 0.5 else 0
            
            # Use the probability as confidence
            confidence = probability if predicted_class == 1 else (1 - probability)
        
        # Map class to gender (0 = Male, 1 = Female)
        gender = "Female" if predicted_class == 1 else "Male"
        
        print(f"Raw output: {output.item():.4f}, Probability: {probability:.4f}, Predicted: {gender}")
        
        return {
            "gender": gender,
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    files = request.files.getlist('images')
    
    if len(files) == 0 or files[0].filename == '':
        return jsonify({"error": "No images selected"}), 400
    
    results = []
    
    for image_file in files:
        try:
            image_bytes = image_file.read()
            result = predict_gender(image_bytes)
            
            if "error" not in result:
                results.append({
                    "filename": image_file.filename,
                    "gender": result["gender"],
                    "confidence": result["confidence"]
                })
            else:
                results.append({
                    "filename": image_file.filename,
                    "error": result["error"]
                })
        except Exception as e:
            results.append({
                "filename": image_file.filename,
                "error": str(e)
            })
    
    return jsonify({"results": results})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "device": str(DEVICE)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

