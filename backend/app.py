from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import pyttsx3  # For text-to-speech
from googletrans import Translator

# Initialize FastAPI app
# Create an instance of the FastAPI app
app = FastAPI()

# Enable CORS
# Add middleware to enable Cross-Origin Resource Sharing (CORS) for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define device
# Determine the device to use for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") else "cpu")

# Define class labels (A-Z + 'space')
# Define labels corresponding to the American Sign Language (ASL) alphabet and 'space'
labels = [chr(i) for i in range(65, 91)] + ['space']  # Adjust based on your dataset

# Define transformations
# Define preprocessing transformations for the input images, including resizing, normalization, and tensor conversion
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the ASL classifier model
# Define the ASLClassifier model that extends PyTorch's nn.Module
class ASLClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(ASLClassifier, self).__init__()
        # Load the EfficientNet model
                # Load the EfficientNet-b0 model from Torch Hub with pretrained weights
        self.base_model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)

        # Access the classifier's input features
        in_features = self.base_model.classifier.in_features  # Correct method to get input features
        self.base_model.classifier = torch.nn.Linear(in_features, num_classes)  # Replace classifier

    def forward(self, x):
        return self.base_model(x)

# Initialize and load model
# Initialize the ASLClassifier model with the appropriate number of classes and move it to the selected device
model = ASLClassifier(num_classes=len(labels)).to(device)
model.load_state_dict(torch.load("asl_classifier.pth", map_location=device))
model.eval()

# Initialize Google Translator
# Initialize the Google Translator instance for text translation functionality
translator = Translator()

# Text-to-Speech Function
# Function to convert text to speech using pyttsx3
# Enhances user interaction by providing auditory feedback
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# API Endpoint for Prediction
# Endpoint to predict the ASL gesture from an uploaded image
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

# API Endpoint for Translation
# Endpoint to translate text to a specified language
@app.post("/translate")
async def translate(request: dict):
    try:
        text = request.get("text", "")
        target_language = request.get("target", "en")

        if not text:
            raise HTTPException(status_code=400, detail="Text for translation is required")

        # Perform translation
        translation = translator.translate(text, dest=target_language)
        return JSONResponse({"translatedText": translation.text})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Root Endpoint
# Root endpoint to check if the API is running
@app.get("/")
async def root():
    return {"message": "ASL Gesture Recognition and Translation API is running."}
