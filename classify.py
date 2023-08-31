import torch
import onnxruntime as ort
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import json
import urllib.request
from PIL import ImageOps

# Load ONNX model
resnet_session = ort.InferenceSession("resnet18.onnx")

# Load an actual image with PIL (substitute the filename)
image = Image.open("cats.jpg")

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Preprocess the image
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Run inference
ort_inputs = {resnet_session.get_inputs()[0].name: to_numpy(input_batch)}
ort_outs = resnet_session.run(None, ort_inputs)
output = ort_outs[0]

# Download ImageNet labels
url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
class_idx = json.load(open('imagenet_class_index.json'))

# Convert to a more usable format: index to label
idx_to_label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# Convert output to class label
_, predicted_class_idx = torch.max(torch.from_numpy(output), 1)
predicted_class_label = idx_to_label[predicted_class_idx]

print(f'Predicted class index: {predicted_class_idx.item()}')
print(f'Predicted class label: {predicted_class_label}')
