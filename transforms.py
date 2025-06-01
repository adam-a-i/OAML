from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math
import numpy as np


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, crop_factor=0.8, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.crop_factor = crop_factor
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.crop_factor ** 2, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, mean=[0.485, 0.456, 0.406]):
        self.EPSILON = EPSILON
        self.mean = mean

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(0.02, 0.2) * area
            aspect_ratio = random.uniform(0.3, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]

                return img

        return img


class RandomOcclusion(object):
    def __init__(self, min_size=0.2, max_size=1):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        if self.max_size == 0:
            return img
        H = img.height
        W = img.width
        S = min(H, W)
        s = np.random.randint(max(1, int(S * self.min_size)), int(S * self.max_size))
        x0 = np.random.randint(W - s)
        y0 = np.random.randint(H - s)
        x1 = x0 + s
        y1 = y0 + s
        block = Image.new('RGB', (s, s), (255, 255, 255))
        img.paste(block, (x0, y0, x1, y1))
        return img

def randomize_point(point, max_offset=3):
    return (
        point[0] + np.random.uniform(-max_offset, max_offset),
        point[1] + np.random.uniform(-max_offset, max_offset)
    )

class MedicalMaskOcclusion(object):
    def __init__(self, mask_path='maskPic.jpg', prob=0.5, ref_points=None):
        """
        Args:
            mask_path: Path to the mask image (should have transparent background)
            prob: Probability of applying the mask
            ref_points: List of 3 (x, y) tuples for nose and mouth corners (CASIA webface standard)
        """
        self.prob = prob
        self.mask_path = mask_path
        self.mask_img = Image.open(mask_path).convert('RGBA')
        # casia webface standard reference points (for 112x112)
        # [nose, left mouth, right mouth]
        if ref_points is None:
            self.ref_points = [
                (48.0252, 71.7366),   # nose
                (33.5493, 92.3655),   # left mouth
                (62.7299, 92.2041)    # right mouth
            ]
        else:
            self.ref_points = ref_points

    def __call__(self, img):
        if np.random.random() > self.prob:
            return img
        img = img.convert('RGBA')
        # randomizing facial reference points
        nose = randomize_point(self.ref_points[0])
        left_mouth = randomize_point(self.ref_points[1])
        right_mouth = randomize_point(self.ref_points[2])
        # random disturbance
        shift_x = np.random.uniform(-5, 5)
        shift_y = np.random.uniform(-5, 5)
        scale = np.random.uniform(0.95, 1.10)
        angle = np.random.uniform(-7, 7)
        # compute mask width and height based on mouth corners
        mouth_width = np.linalg.norm(np.array(right_mouth) - np.array(left_mouth))
        # Increased width multiplier for wider coverage (accounting for strings)
        mask_width = int(mouth_width * 4.0 * scale)  # Increased multiplier for wider coverage (accounting for strings)
        # Increased height calculation multiplier for longer coverage
        mask_height = int((right_mouth[1] - nose[1]) * 4.0 * scale)  # Increased multiplier for longer coverage (covering chin)
        # resizing and rotating mask
        mask = self.mask_img.resize((mask_width, mask_height), Image.BILINEAR)
        mask = mask.rotate(angle, expand=True)
        # compute mask position (centered horizontally on nose, vertically just above nose)
        center_x = int(nose[0] + shift_x)
        # Adjusted vertical paste position (shifted upwards)
        # Increased vertical offset to make mask cover the nose and mouth properly
        vertical_offset = 25 # Pixels to shift the mask upwards from the nose (increased for better coverage)
        paste_y = int(nose[1] - vertical_offset + shift_y)
        paste_x = center_x - mask.width // 2

        img.paste(mask, (paste_x, paste_y), mask)
        return img.convert('RGB')
