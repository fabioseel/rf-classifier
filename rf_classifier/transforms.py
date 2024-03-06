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

    def __init__(self, **affine_args):
        super().__init__()
        self.affine_args = affine_args

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            Image with rescaled color channels.
        """
        inp_avg = img.mean(axis=(1,2))
        transf = transforms.RandomAffine(**self.affine_args, fill=[*inp_avg])
        return transf(img)
    
class RandomColorRescale(torch.nn.Module):
    def __init__(self, r=(0,1), g=(0,1), b=(0,1)):
        super().__init__()
        self.r = r
        self.g = g
        self.b = b

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            Image with rescaled color channels.
        """
        rand = torch.rand(3)
        rand[0] = rand[0]*(self.r[1]-self.r[0])+self.r[0]
        rand[1] = rand[1]*(self.g[1]-self.g[0])+self.g[0]
        rand[2] = rand[2]*(self.b[1]-self.b[0])+self.b[0]
        return img*rand[:,None,None]
