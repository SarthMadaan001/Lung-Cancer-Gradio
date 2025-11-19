import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import torch.nn as nn
import time

# --- MODEL LOADING AND PREPARATION ---

print("Starting model download and loading...")
try:
    ckpt = hf_hub_download(repo_id='Sarth001/LungCanver', filename='best_model_v2_fixed.pth')
    state = torch.load(ckpt, map_location='cpu')['model_state_dict']
    print("Model checkpoint loaded successfully.")

    if any(k.startswith('fc.1.weight') for k in state):
        use_dropout = True
        w = next(state[k] for k in state if k.startswith('fc.1.weight'))
        num_classes = w.shape[0]
    else:
        use_dropout = False
        w = next(state[k] for k in state if k.startswith('fc.weight'))
        num_classes = w.shape[0]

    model = models.resnet50(weights=None)
    
    if use_dropout:
        model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, num_classes))
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    model.load_state_dict(state, strict=False)
    model.eval()
    
except Exception as e:
    print(f"Error loading model: {e}")
    class DummyModel(nn.Module):
        def forward(self, x):
            return torch.zeros(x.size(0), 3)
    model = DummyModel()
    num_classes = 3


# Standard image transformation pipeline for ResNet
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labels = {0: 'Benign cases', 1: 'Malignant cases', 2: 'Normal cases'}

# --- PREDICTION FUNCTION ---

def predict_image(img):
    """Predicts the probability of lung cancer classes from an image."""
    
    if isinstance(img, str): 
        img = Image.open(img).convert('RGB')
    
    x = tf(img).unsqueeze(0)
    
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].numpy()
        
    time.sleep(1)
    
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# --- FRONTEND (AMAZING UI with Blocks and Theme) ---

try:
    example_dir = "examples"
    example_files = [os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith(('.jpg', '.png'))]
except FileNotFoundError:
    example_files = []

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), title='Lung Cancer Classifier') as app:
    
    gr.Markdown(
        """
        # 🫁 AI-Powered Lung Cancer Classifier
        Upload a medical image (X-ray or CT scan) to get a probability-based classification.
        
        ### **⚠️ DISCLAIMER: This tool is for research and educational use only and is NOT a substitute for professional medical diagnosis.**
        """
    )

    with gr.Row(variant='panel'):
        
        with gr.Column(scale=1):
            image_input = gr.Image(
                type='pil', 
                label='Input Image',
                show_label=True,
                height=300
            )
            
            gr.Examples(
                examples=example_files,
                inputs=image_input,
                label="Sample Images (Click to load)"
            )
            
            predict_button = gr.Button('🧬 Analyze Image', variant='primary')
            
        with gr.Column(scale=1):
            prediction_output = gr.Label(
                num_top_classes=3, 
                label='Diagnosis Probability',
                show_label=True
            )
            gr.Markdown(
                """
                The model classifies the image into one of three categories:
                - **Benign cases**
                - **Malignant cases** (Cancer)
                - **Normal cases**
                """
            )

    predict_button.click(
        fn=predict_image,
        inputs=image_input,
        outputs=prediction_output,
        show_progress='full' 
    )

# --- APP LAUNCH (Cloud-Safe Configuration) ---

if __name__ == '__main__':
    server_port = int(os.environ.get('PORT', 7860))
    app.launch(server_name='0.0.0.0', server_port=server_port)