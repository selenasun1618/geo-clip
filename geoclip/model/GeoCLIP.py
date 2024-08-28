import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import load_gps_data, file_dir

from PIL import Image
from torchvision.transforms import ToPILImage

class GeoCLIP(nn.Module):
    def __init__(self, from_pretrained=True, queue_size=4096):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()

        self.gps_gallery = load_gps_data(os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv"))
        self._initialize_gps_queue(queue_size)

        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            self._load_weights()

        self.device = "cpu"

    # ... (other methods remain the same)

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """ Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)

        # Remove the assertion and handle variable batch sizes
        if gps_batch_size > self.queue_size:
            gps = gps[:self.queue_size]
            gps_batch_size = self.queue_size

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        if gps_ptr + gps_batch_size <= self.queue_size:
            self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        else:
            # Handle wrap-around
            first_part = self.queue_size - gps_ptr
            self.gps_queue[:, gps_ptr:] = gps[:first_part].t()
            self.gps_queue[:, :gps_batch_size - first_part] = gps[first_part:].t()

        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    # ... (rest of the class remains the same)