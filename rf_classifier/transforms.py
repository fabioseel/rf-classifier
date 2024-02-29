import torch
from torchvision import transforms

class ColorRescale(torch.nn.Module):
    """

    Args:
        min (float): 
        max (float):
    """

    def __init__(self, min=0,max=1):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, img):
        """
        Args:
            img (Tensor): Image to be color rescaled.

        Returns:
            Image with rescaled color channels.
        """
        return (img - img.min()) / (img.max() - img.min())*(self.max-self.min)-self.min
    
class RandomAffine(torch.nn.Module):
    """

    Args:
        min (float): 
        max (float):
    """

    def __init__(self, degrees):
        super().__init__()
        self.degrees = degrees

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            Image with rescaled color channels.
        """
        inp_avg = img.mean(axis=(1,2))
        transf = transforms.RandomAffine(self.degrees, fill=[*inp_avg])
        return transf(img)