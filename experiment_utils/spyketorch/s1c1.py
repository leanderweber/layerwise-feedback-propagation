################################################################################
# Title:            s1c1.py                                                    #
# Description:      Class for generating the temporal encoding of the input    #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.1                                                        #
# Usage:            None                                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import SpykeTorch.functional as sf
import torchvision.transforms as transforms
from SpykeTorch import utils


class S1C1:
    """Generate the temporal encoding of the input"""

    def __init__(self, filter, timesteps=15):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.cnt = 0

    def __call__(self, image):
        """Generate the temporal encoding of the input"""
        # if self.cnt % 1000 == 0:
        #     print(self.cnt)
        # image = transforms.Grayscale()(image)
        self.cnt += 1
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()
