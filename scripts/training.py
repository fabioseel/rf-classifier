import torch
from torch.optim import SGD, AdamW
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from rf_classifier.datasets import SyntheticRFsDataset
from rf_classifier.models import CNNClassifier
from rf_classifier.train import train, validate
from rf_classifier.transforms import (ColorRescale, RandomAffine,
                                      RandomColorRescale)

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)
print("Using", device)


inp_transform = transforms.Compose(
    [
        transforms.Resize(
            (32, 32), antialias=True, interpolation=transforms.InterpolationMode.NEAREST
        ),
        ColorRescale(-1, 1),
    ]
)

data_transforms = transforms.Compose(
    [
        transforms.RandomInvert(),
        RandomAffine(
            degrees=180, translate=(0.1, 0.1), scale=(0.5, 2), shear=(-10, 10, -10, 10)
        ),
        RandomColorRescale(),
        inp_transform,
    ]
)
data = SyntheticRFsDataset(1000, transform=data_transforms)
test_data = ImageFolder(
    "../data/example_rfs",
    transform=transforms.Compose([transforms.ToTensor(), inp_transform]),
)

train_loader = DataLoader(data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

classifier = CNNClassifier(n_classes=len(data.classes))
classifier.to(device)
optimizer = AdamW(classifier.parameters(), lr=0.0001)
num_epochs = 1000

early_stop = 20
prev_val_acc = 0
early_stop_count = 0

for i in range(num_epochs):
    train_epoch_loss, train_epoch_acc = train(
        classifier, optimizer=optimizer, train_loader=train_loader, device=device
    )
    val_acc, _, _ = validate(classifier, test_loader, device)
    if val_acc > prev_val_acc:
        prev_val_acc = val_acc
        early_stop_count = 0
        torch.save(
            classifier.state_dict(),
            "../weights/cnn_classifier/synth_classifier_InterpNear.pth",
        )
    else:
        early_stop_count += 1
        if early_stop_count > early_stop:
            break
