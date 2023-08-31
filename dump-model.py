import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()
x = torch.randn(1, 3, 224, 224)
torch.onnx.export(model,               # model being run
                  x,                   # model input (or a tuple for multiple inputs)
                  "resnet18.onnx",     # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'], # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})

# Download labels

import requests

url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
r = requests.get(url, allow_redirects=True)
open('imagenet_class_index.json', 'wb').write(r.content)
