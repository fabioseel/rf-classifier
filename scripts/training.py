from rf_classifier.train import train, validate
from rf_classifier.models import CNNClassifier
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from rf_classifier.transforms import RandomAffine, ColorRescale, RandomColorRescale
from rf_classifier.datasets import SyntheticRFsDataset
from torchvision.transforms import InterpolationMode
from torch.optim import AdamW, SGD
import torch

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)
print("Using", device)

data_transforms = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.RandomInvert(),
        transforms.Resize(
            (32, 32), antialias=False
        ),
        RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.5,2), shear=(-10,10,-10,10)),
        RandomColorRescale(),
        ColorRescale(-1, 1),
    ]
)
data = SyntheticRFsDataset(1000, transform=data_transforms)#ImageFolder("../data/example_rfs", transform=data_transforms)

train_size = int(0.9 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    data, [train_size, test_size]
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

classifier = CNNClassifier(n_classes=len(data.classes))
classifier.to(device)
optimizer = AdamW(classifier.parameters(), lr=0.0001)
num_epochs = 500

early_stop = 20
prev_val_acc = 0
early_stop_count = 0

for i in range(num_epochs):
    train_epoch_loss, train_epoch_acc = train(
        classifier, optimizer=optimizer, train_loader=train_loader, device=device
    )
    val_acc = validate(classifier, test_loader, device)
    if val_acc > prev_val_acc:
        prev_val_acc = val_acc
        early_stop_count = 0
        torch.save(classifier.state_dict(), "../weights/cnn_classifier/synthetic_classifier.pth")
    else:
        early_stop_count += 1
        if early_stop_count > early_stop:
            break
