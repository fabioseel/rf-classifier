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
        transforms.Grayscale(3),
        ColorRescale(-1,1)
        # transforms.Normalize(mean=0.5, std=1),
    ]
)

data_transforms = transforms.Compose(
    [
        transforms.RandomInvert(),
        transforms.RandomAffine(
            degrees=180, translate=(0.1, 0.1), scale=(0.5, 2), shear=(-10, 10, -10, 10)
        ),
        RandomColorRescale(),
        inp_transform,
    ]
)
data = SyntheticRFsDataset(30000, transform=inp_transform)
big_train_data = ImageFolder(
    "../data/example_rfs2",
    transform=transforms.Compose([transforms.ToTensor(), data_transforms]),
)
test_data = ImageFolder(
    "../data/example_rfs",
    transform=transforms.Compose([transforms.ToTensor(), inp_transform]),
)

train_loader_synth = DataLoader(torch.utils.data.dataset.ConcatDataset([data, big_train_data]), batch_size=32, shuffle=True, prefetch_factor=3, pin_memory=True, num_workers=4)
test_loader = DataLoader(torch.utils.data.dataset.ConcatDataset([test_data, big_train_data]), batch_size=256, shuffle=False)

classifier = CNNClassifier(n_classes=len(data.classes))
classifier.to(device)
optimizer = AdamW(classifier.parameters(), lr=0.0001)
num_epochs = 1000

early_stop = 50
prev_val_acc = 0
early_stop_count = 0

# Phase 1: Train on synthetic dataset with some
for i in range(num_epochs):
    train_epoch_loss, train_epoch_acc = train(
        classifier, optimizer=optimizer, train_loader=train_loader_synth, device=device
    )
    val_acc, _, _ = validate(classifier, test_loader, device)
    if val_acc > prev_val_acc:
        prev_val_acc = val_acc
        early_stop_count = 0
        torch.save(
            classifier.state_dict(),
            "../weights/cnn_classifier/greyscale_classifier.pth",
        )
    else:
        early_stop_count += 1
        if early_stop_count > early_stop:
            break