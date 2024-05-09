import PySimpleGUI as sg
import cv2
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Initialize prediction history
prediction_history = []

# Define transformations and model
transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.network.fc.in_features
        self.network.fc = torch.nn.Linear(num_ftrs, 6)  # Adjust based on your number of classes

    def forward(self, xb):
        return self.network(xb)

def load_model(filepath):
    model = ResNet()
    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    model.eval()
    return model

model_path = 'C:\\Users\\Acer\\OneDrive\\Desktop\\New folder (5)\\resnet_model_weights.pth'  # Update this path
model = load_model(model_path)

def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Press Space to Capture; ESC to Exit', frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            print("Closing without capture.")
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif k % 256 == 32:  # Spacebar
            print("Image captured")
            cap.release()
            cv2.destroyAllWindows()
            return frame

def predict_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    image_tensor = transformations(pil_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'][predicted[0]]  # Update your classes
    prediction_history.append(predicted_class)  # Save prediction to history
    return predicted_class

# GUI Layout
layout = [
    [sg.Button("Predict Now"), sg.Button("Last Prediction"), sg.Button("Exit")]
]

window = sg.Window("Garbage Classifier", layout)

while True:
    event, values = window.read()
    if event == "Predict Now":
        frame = capture_image()
        if frame is not None:
            img_bytes = cv2.imencode('.png', frame)[1].tobytes()
            image_preview_layout = [[sg.Image(data=img_bytes)],
                                    [sg.Button("Retake"), sg.Button("Analyze")]]
            image_preview_window = sg.Window("Preview", image_preview_layout)
            while True:
                event, values = image_preview_window.read()
                if event == "Retake":
                    frame = capture_image()
                    if frame is not None:
                        img_bytes = cv2.imencode('.png', frame)[1].tobytes()
                        image_preview_window['-IMAGE-'].update(data=img_bytes)
                elif event == "Analyze":
                    predicted_class = predict_image(frame)
                    sg.popup(f"This image resembles: {predicted_class}")
                    break
                elif event == sg.WIN_CLOSED:
                    break
            image_preview_window.close()
    elif event == "Last Prediction":
        if prediction_history:
            history_str = "\n".join(prediction_history)
            sg.popup("Prediction History:", history_str)
        else:
            sg.popup("No predictions made yet.")
    elif event == "Exit" or event == sg.WIN_CLOSED:
        break

window.close()
