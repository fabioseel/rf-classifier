from rf_classifier.models import CNNClassifier
from rf_classifier.datasets import SingleImageFolder
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from rf_classifier.transforms import ColorRescale
import torch
import os
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_base_path", type=str, help="for all sub(subsub...)folders that contain .png images a file will be generated with the predicted labels for each image")
parser.add_argument("-m", "--model", type=str, help="path of the model to be used for classification")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inp_transform = transforms.Compose(
    [
        transforms.Resize(
            (32, 32), antialias=True, interpolation=transforms.InterpolationMode.NEAREST
        ),
        ColorRescale(-1, 1),
    ]
)

img_transforms = transforms.Compose([transforms.ToTensor(), inp_transform])

model_path = args.model
model_name = os.path.split(model_path)[1]

# Load weights and extract number of classes to build model
weights = torch.load(model_path)
n_classes = weights[next(reversed(weights))].shape[0]

classifier = CNNClassifier(n_classes=n_classes) # TODO: store parametrization of classifier in more meaningful way
classifier.load_state_dict(weights)
classifier.to(device)
classifier.eval()

dest_base_path = args.img_base_path

networks = []
for _dir in os.walk(dest_base_path):
    for file in _dir[2]:
        if(file.endswith(".png")):
            networks.append(_dir[0])
            break

for network in tqdm(networks):
    print(network)
    dataset = SingleImageFolder(network, img_transforms)
    dataloader = DataLoader(dataset, batch_size=1)
    pred_dict = {}
    for img, name in tqdm(dataloader):
        pred = classifier(img.to(device))
        pred_dict[name[0]]=pred[0].detach().cpu().numpy().tolist()
    with open(os.path.join(network,"autolabels.json"), "w") as f:
        json.dump(pred_dict, f)