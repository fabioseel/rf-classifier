from random import randint
from random import uniform as urand

import torch
from torch.utils.data import Dataset
from skimage.transform import resize

from rf_classifier import rf_generator
import os
from PIL import Image
from matplotlib.colors import hsv_to_rgb


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
                image_tensor = (
                    torch.from_numpy(image).to(dtype=torch.float32).movedim(2, 0)
                )
                # Append image and label to data list
                data.append((image_tensor, class_idx))
        return data
    
    def generate_clean_rf(self, class_idx, image_size=None):
        cmin, cmax = -1, 1
        if image_size is None:
            image_size = [randint(7, 32)] * 2  # TODO: Make "magic numbers" parameters

        if class_idx == 0:  # CENTER SURROUND
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
                rf, urand(cmin, cmax), urand(cmin, cmax), urand(cmin, cmax)
            )
        elif class_idx == 1:  # COLOR
            rgb = hsv_to_rgb([urand(0, 1), urand(0.3, 1), urand(0.3, 1)])
            rgb = rgb * (cmax - cmin) + cmin
            rf = rf_generator.color(shape=image_size, color=rgb)
        elif class_idx == 2:  # GABOR

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
                rf, urand(cmin, cmax), urand(cmin, cmax), urand(cmin, cmax)
            )
        elif class_idx == 3:  # MULTI FREQUENCY
            freq1 = 1 / urand(image_size[0] / 6, image_size[0] / 2)
            theta1 = urand(0, torch.pi)
            theta_diff = urand(torch.pi / 8, 7 / 8 * torch.pi)
            rf = rf_generator.mult_freq(
                shape=image_size,
                freq1=freq1,
                freq2=urand(freq1 * 0.75, freq1 * 1.25),
                theta1=theta1,
                theta2=theta1 + theta_diff,
            )
            rf = rf_generator.greyscale_to_color(
                rf, urand(cmin, cmax), urand(cmin, cmax), urand(cmin, cmax)
            )
        elif class_idx == 4:  # NOISE
            rf = rf_generator.decorrelated_color_noise(shape=image_size, low=-1)
            color_bias_strength = urand(0,0.4)
            rf = (1-color_bias_strength)*rf+color_bias_strength*rf_generator.color(shape=image_size,color=(urand(cmin,cmax),urand(cmin,cmax),urand(cmin,cmax)))
        elif class_idx == 5:  # SIMPLE EDGE
            sigma_x = urand(image_size[0] / 8, image_size[0] * 3 / 4)
            rf = rf_generator.simple_edge(
                shape=image_size,
                theta=urand(0, torch.pi),
                sigma_x=sigma_x,
                sigma_y=urand(sigma_x * 0.5, sigma_x * 1.5),
                center_offset=(
                    urand(-image_size[0] / 4, image_size[0] / 4),
                    urand(-image_size[1] / 4, image_size[1] / 4),
                ),
            )
            rf = rf_generator.greyscale_to_color(
                rf, urand(cmin, cmax), urand(cmin, cmax), urand(cmin, cmax)
            )
        elif (
            class_idx == 6
        ):  # TODO: UNCLASSIFIABLE. Need better generation here... Or do I?
            factor_x = urand(2, image_size[0] / 2)
            factor_y = urand(2, image_size[1] / 2)
            small_img_size = (
                int(image_size[0] // factor_x),
                int(image_size[1] // factor_y),
            )
            rf = rf_generator.decorrelated_color_noise(shape=small_img_size, low=-1)
            rf = resize(rf, image_size)
        return rf

    def generate_image(self, class_idx, image_size=None):
        rf = self.generate_clean_rf(class_idx,image_size)
        image_size = rf.shape[:-1]
        random_other_rf = self.generate_clean_rf(randint(0,6), image_size)

        rf_range = rf.max()-rf.min()
        rand_rf_range = random_other_rf.max()-random_other_rf.min()
        
        rf = rf+random_other_rf/rand_rf_range * urand(0,0.7) * rf_range

        noise_strength = urand(0, 0.2) * rf_range
        rf += rf_generator.decorrelated_color_noise(
            shape=image_size, low=-noise_strength, high=noise_strength
        )

        gauss_sigma_x = urand(image_size[0] / 6, image_size[0])
        return rf_generator.apply_gaussian(
            rf,
            theta=urand(0, torch.pi),
            sigma_x=gauss_sigma_x,
            sigma_y=urand(gauss_sigma_x * 0.5, gauss_sigma_x * 1.5)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class SingleImageFolder(Dataset):
    def __init__(self, folder_path, transform=None, file_ending=".png"):
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = []
        for file in os.listdir(folder_path):
            if file.endswith(file_ending):
                self.image_filenames.append(file)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_name
