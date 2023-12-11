from typing import Dict, Any
import cv2
import numpy as np
import random

from transforms.default_transforms import BaseTransform

class AddFog(BaseTransform):
    def __init__(self, 
                 intensity_range = (0.1, 0.2),
                 fog_prob: float = 0.4,
                 blur_ksize = (37, 37),
                 blur_prob: float = 0.5):
        
        super().__init__(change_img=True, 
                         change_metas=False, 
                         change_calib=False, 
                         change_label=False)
        
        self.intensity_range = intensity_range
        self.fog_prob = fog_prob

        self.blur_ksize = blur_ksize
        self.blur_prob = blur_prob
    
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.fog_prob:
            image = data_dict['img']
            image = self._add_fog(image)
            data_dict['img'] = image

        if random.random() < self.blur_prob:
            image = data_dict['img']
            image = self._add_blur(image)
            data_dict['img'] = image
        
        return data_dict
    
    def _add_blur(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, self.blur_ksize, 0)
    
    def _add_fog(self, image: np.ndarray) -> np.ndarray:
        fog = np.ones_like(image)
        alpha = random.uniform(*self.intensity_range)
        image = cv2.addWeighted(image, 1 - alpha, fog, alpha, 0)
        return image