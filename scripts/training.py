from rf_classifier.train import train, validate
from rf_classifier.models import CNNClassifier
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from rf_classifier.transforms import RandomAffine, ColorRescale
from torch.optim import AdamW, SGD
import torch

if torch.cuda.is_available():
  dev = "cuda:2"
else:
  dev = "cpu"
device = torch.device(dev)
print("Using", device)

data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32), antialias=False), RandomAffine(180), ColorRescale(-1,1)])
valid_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32), antialias=False), ColorRescale(-1,1)])
data = ImageFolder("../data/example_rfs", transform=data_transforms)
id_to_class = {data.class_to_idx[class_name]: class_name for class_name in data.class_to_idx}

train_size = int(0.9 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classifier = CNNClassifier(n_classes=len(data.classes))
classifier.to(device)
optimizer=AdamW(classifier.parameters())
num_epochs = 500

early_stop = 50
prev_val_acc = 0
early_stop_count = 0

for i in range(num_epochs):
    train_epoch_loss, train_epoch_acc = train(classifier, optimizer=optimizer, train_loader=train_loader, device=device)
    val_acc = validate(classifier, test_loader, device)
    if val_acc > prev_val_acc:
        prev_val_acc = val_acc
        early_stop_count = 0
    else:
        early_stop_count+=1
        if early_stop_count > early_stop:
            break

torch.save(classifier.state_dict(), "../weights/first_simple_classifier.pth")