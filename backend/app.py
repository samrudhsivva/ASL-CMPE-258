from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import pyttsx3  # For text-to-speech

# Initialize FastAPI app
app = FastAPI()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class labels (A-Z + 'space')
labels = [chr(i) for i in range(65, 91)] + ['space']  # Adjust based on your dataset

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ASLClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(ASLClassifier, self).__init__()
        # Load the EfficientNet model
        self.base_model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)

        # Access the classifier's input features
        in_features = self.base_model.classifier.in_features  # Correct method to get input features
        self.base_model.classifier = torch.nn.Linear(in_features, num_classes)  # Replace classifier

    def forward(self, x):
        return self.base_model(x)



# Initialize and load model
model = ASLClassifier(num_classes=len(labels)).to(device)
model.load_state_dict(torch.load("asl_classifier.pth", map_location=device))
model.eval()

# Text-to-Speech Function
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# API Endpoint for Prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image = Image.open(file.file).convert("RGB")

        # Preprocess the image
        image = transform(image).unsqueeze(0).to(device)

        # Predict the label
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_label = labels[predicted.item()]

        # Speak the result (optional)
        speak_text(f"The predicted sign is {predicted_label}")

        # Return the result as JSON
        return JSONResponse({"predicted_label": predicted_label})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
