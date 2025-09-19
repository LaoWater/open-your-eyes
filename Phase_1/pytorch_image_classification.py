import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np

class RestaurantSafetyClassifier:
    def __init__(self, num_classes=3):  # gloves_on, gloves_off, no_hands
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet (transfer learning - like fine-tuning an LLM!)
        self.model = models.resnet50(pretrained=True)
        
        # Replace final layer for our classes (like changing LLM output layer)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline (like tokenization for text)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        self.classes = ['gloves_on', 'gloves_off', 'no_hands']
    
    def preprocess_image(self, image_path_or_array):
        """Convert image to tensor (like tokenizing text)"""
        if isinstance(image_path_or_array, str):
            image = Image.open(image_path_or_array).convert('RGB')
        else:
            # OpenCV array (BGR) -> PIL (RGB)
            image = Image.fromarray(cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB))
        
        return self.transform(image).unsqueeze(0)  # Add batch dimension
    
    def predict(self, image):
        """Get safety prediction"""
        with torch.no_grad():  # Like inference mode for LLMs
            input_tensor = self.preprocess_image(image).to(self.device)
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            
            return {
                'class': self.classes[predicted_class],
                'confidence': confidence,
                'all_probabilities': {
                    cls: prob.item() for cls, prob in zip(self.classes, probabilities)
                }
            }

# Example usage
def demo_basic_classification():
    classifier = RestaurantSafetyClassifier()
    
    # Simulate with a webcam frame
    cap = cv2.VideoCapture(0)  # Your camera
    
    print("Press 'q' to quit, 's' to analyze current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow('Restaurant Safety Monitor', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Analyze current frame
            result = classifier.predict(frame)
            print(f"Prediction: {result['class']} (confidence: {result['confidence']:.2f})")
            
            # Draw result on frame
            color = (0, 255, 0) if result['class'] == 'gloves_on' else (0, 0, 255)
            cv2.putText(frame, f"{result['class']}: {result['confidence']:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('Result', frame)
            cv2.waitKey(2000)  # Show for 2 seconds
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demo_basic_classification()