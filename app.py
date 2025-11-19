import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# Download checkpoint from model repo (cached)
ckpt = hf_hub_download(repo_id='Sarth001/LungCanver', filename='best_model_v2_fixed.pth')
state = torch.load(ckpt, map_location='cpu')['model_state_dict']

if any(k.startswith('fc.1.weight') for k in state):
    use_dropout = True
    w = next(state[k] for k in state if k.startswith('fc.1.weight'))
    num_classes = w.shape[0]
else:
    use_dropout = False
    w = next(state[k] for k in state if k.startswith('fc.weight'))
    num_classes = w.shape[0]

import torch.nn as nn
model = models.resnet50(pretrained=False)
if use_dropout:
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, num_classes))
else:
    model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(state, strict=False)
model.eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

labels = {0:'Bengin cases', 1:'Malignant cases', 2:'Normal cases'}

def predict_image(img):
    if isinstance(img, str): img = Image.open(img).convert('RGB')
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].numpy()
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

iface = gr.Interface(fn=predict_image,
                      inputs=gr.Image(type='pil'),
                      outputs=gr.Label(num_top_classes=3),
                      title='LungCanver - Lung Cancer Classifier',
                      description='Research use only')
if __name__ == '__main__':
    iface.launch()