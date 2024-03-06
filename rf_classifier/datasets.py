from torch.utils.data import Dataset
from rf_classifier import rf_generator
from random import uniform as urand, randint
import torch
from torchvision.transforms import Resize, InterpolationMode
import numpy as np


class SyntheticRFsDataset(Dataset):
    def __init__(self, num_samples_per_class, transform=None):
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 7
        self.classes = [
            "center_surround",
            "color",
            "gabor",
            "mult_freq",
            "noise",
            "simple_edge",
            "unclassifiable",
        ]
        self.transform = transform

        # Generate synthetic data
        self.data = self.generate_data()

    def generate_data(self):
        data = []
        for class_idx in range(self.num_classes):
            for _ in range(self.num_samples_per_class):
                # Generate image for each class
                image = self.generate_image(class_idx)
                # Convert image to tensor
                image_tensor = torch.from_numpy(image).to(dtype=torch.float32).movedim(2, 0)
                # Append image and label to data list
                data.append((image_tensor, class_idx))
        return data

    def generate_image(self, class_idx, image_size=None):
        if image_size is None:
            image_size = [randint(7, 64)] * 2

        if class_idx == 0:
            sigma1 = urand(1, image_size[0] / 2)
            sigma2 = urand(1, image_size[1] / 2)
            rf = rf_generator.center_surround(
                shape=image_size,
                theta=urand(0, torch.pi),
                sigma_x1=sigma1,
                sigma_y1=sigma1,
                sigma_x2=sigma2,
                sigma_y2=sigma2,
                center_offset=(
                    urand(-image_size[0] / 4, image_size[0] / 4),
                    urand(-image_size[1] / 4, image_size[1] / 4),
                ),
            )
            rf = rf_generator.greyscale_to_color(
                rf, urand(-1, 1), urand(-1, 1), urand(-1, 1)
            )
        elif class_idx == 1:
            rf = rf_generator.color(
                shape=image_size, color=(urand(-1, 1), urand(-1, 1), urand(-1, 1))
            )
        elif class_idx == 2:

            sigma_x = urand(image_size[0] / 8, image_size[0] * 3 / 4)
            rf = rf_generator.gabor_kernel(
                shape=image_size,
                frequency=1 / urand(image_size[0] / 8, image_size[0] / 2),
                theta=urand(0, torch.pi),
                sigma_x=sigma_x,
                sigma_y=urand(sigma_x * 0.5, sigma_x * 1.5),
                center_offset=(
                    urand(-image_size[0] / 4, image_size[0] / 4),
                    urand(-image_size[1] / 4, image_size[1] / 4),
                ),
            )
            rf = rf_generator.greyscale_to_color(
                rf, urand(-1, 1), urand(-1, 1), urand(-1, 1)
            )
        elif class_idx == 3:
            rf = rf_generator.mult_freq(
                shape=image_size,
                freq1=1 / urand(image_size[0] / 8, image_size[0] / 2),
                freq2=1 / urand(image_size[0] / 8, image_size[1] / 2),
                theta1=urand(0, torch.pi),
                theta2=urand(0, torch.pi),
            )
            rf = rf_generator.greyscale_to_color(
                rf, urand(-1, 1), urand(-1, 1), urand(-1, 1)
            )
        elif class_idx == 4:
            rf = rf_generator.decorrelated_color_noise(shape=image_size, low=-1)
        elif class_idx == 5:
            rf = rf_generator.simple_edge(shape=image_size, theta=urand(0, torch.pi))
            rf = rf_generator.greyscale_to_color(
                rf, urand(-1, 1), urand(-1, 1), urand(-1, 1)
            )
        elif class_idx == 6:  # Unclassifiable. Need better generation here...
            rf = (
                self.generate_image(0, image_size) * self.generate_image(1, image_size)
                + self.generate_image(2, image_size)
                * self.generate_image(3, image_size)
                + self.generate_image(4, image_size)
                * self.generate_image(5, image_size)
            )

        rf += rf_generator.decorrelated_color_noise(
            shape=image_size, low=-0.02, high=0.02
        )

        gauss_sigma_x = urand(image_size[0] / 8, image_size[0] * 3 / 4)
        return rf_generator.apply_gaussian(
            rf,
            theta=urand(0, torch.pi),
            sigma_x=gauss_sigma_x,
            sigma_y=urand(gauss_sigma_x * 0.5, gauss_sigma_x * 1.5),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
