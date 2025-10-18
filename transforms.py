from __future__ import absolute_import

from torchvision.transforms import *
from torchvision import transforms
from PIL import Image
import random
import math
import numpy as np
import torch


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
            # No occlusion applied - return image and all-ones mask (fully visible)
            binary_mask = Image.new('L', img.size, 255)  # All white = fully visible
            return img, binary_mask
            
        img = img.convert('RGBA')
        # Create binary mask to track occlusion (start with all visible)
        binary_mask = Image.new('L', img.size, 255)  # White = visible (1), Black = occluded (0)
        
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

        # Apply the medical mask to the image
        img.paste(mask, (paste_x, paste_y), mask)
        
        # Create binary occlusion mask from the medical mask's alpha channel
        alpha_mask = mask.split()[-1]  # Get alpha channel of medical mask
        
        # Convert alpha to binary occlusion mask:
        # - High alpha (opaque parts of medical mask) -> 0 (occluded)  
        # - Low alpha (transparent parts) -> 255 (visible)
        binary_alpha = alpha_mask.point(lambda x: 0 if x > 128 else 255)
        
        # Paste the binary mask onto the full image mask
        # Only the opaque parts of the medical mask will be marked as occluded (0)
        binary_mask.paste(binary_alpha, (paste_x, paste_y))
        
        return img.convert('RGB'), binary_mask


class SyntheticOcclusionMask(object):
    """
    Synthetic occlusion transform that generates both occluded images and ground truth masks.
    
    This transform creates random rectangular occlusions and returns both the occluded image
    and a binary mask indicating occluded regions (0=occluded, 1=visible).
    The mask is designed to match the spatial dimensions of CNN feature maps (7x7 for 112x112 input).
    """
    def __init__(self, prob=0.5, min_size=0.1, max_size=0.4, max_patches=3, feature_map_size=7):
        """
        Args:
            prob: Probability of applying occlusion
            min_size: Minimum occlusion size as fraction of image dimension
            max_size: Maximum occlusion size as fraction of image dimension  
            max_patches: Maximum number of occlusion patches
            feature_map_size: Target feature map size (7 for 112x112 input, 14 for 224x224)
        """
        self.prob = prob
        self.min_size = min_size
        self.max_size = max_size
        self.max_patches = max_patches
        self.feature_map_size = feature_map_size
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
            
        Returns:
            tuple: (occluded_image, occlusion_mask)
                - occluded_image: PIL Image with synthetic occlusions
                - occlusion_mask: numpy array [feature_map_size, feature_map_size] 
                  where 1=visible, 0=occluded
        """
        if np.random.random() > self.prob:
            # No occlusion - return original image with all-visible mask
            mask = np.ones((self.feature_map_size, self.feature_map_size), dtype=np.float32)
            return img, mask
        
        # Create copy for occlusion
        occluded_img = img.copy()
        H, W = img.height, img.width
        
        # Initialize mask as all visible
        mask = np.ones((self.feature_map_size, self.feature_map_size), dtype=np.float32)
        
        # Generate random number of occlusion patches
        num_patches = np.random.randint(1, self.max_patches + 1)
        
        for _ in range(num_patches):
            # Random occlusion size
            size_frac = np.random.uniform(self.min_size, self.max_size)
            patch_size = int(min(H, W) * size_frac)
            
            # Random aspect ratio (slightly rectangular)
            aspect_ratio = np.random.uniform(0.7, 1.4)
            patch_h = int(patch_size / np.sqrt(aspect_ratio))
            patch_w = int(patch_size * np.sqrt(aspect_ratio))
            
            # Ensure patch fits in image
            patch_h = min(patch_h, H - 1)
            patch_w = min(patch_w, W - 1)
            
            if patch_h <= 0 or patch_w <= 0:
                continue
                
            # Random position
            x0 = np.random.randint(0, max(1, W - patch_w))
            y0 = np.random.randint(0, max(1, H - patch_h))
            x1 = x0 + patch_w
            y1 = y0 + patch_h
            
            # Apply occlusion to image (random color or black)
            if np.random.random() < 0.5:
                # Random color block
                color = tuple(np.random.randint(0, 256, 3))
            else:
                # Black block
                color = (0, 0, 0)
            
            block = Image.new('RGB', (patch_w, patch_h), color)
            occluded_img.paste(block, (x0, y0, x1, y1))
            
            # Update mask - map image coordinates to feature map coordinates
            # Convert image coordinates to feature map coordinates
            fm_x0 = int((x0 / W) * self.feature_map_size)
            fm_y0 = int((y0 / H) * self.feature_map_size)
            fm_x1 = int((x1 / W) * self.feature_map_size)
            fm_y1 = int((y1 / H) * self.feature_map_size)
            
            # Ensure bounds
            fm_x0 = max(0, min(fm_x0, self.feature_map_size - 1))
            fm_y0 = max(0, min(fm_y0, self.feature_map_size - 1))
            fm_x1 = max(fm_x0 + 1, min(fm_x1 + 1, self.feature_map_size))
            fm_y1 = max(fm_y0 + 1, min(fm_y1 + 1, self.feature_map_size))
            
            # Mark occluded regions in feature map
            mask[fm_y0:fm_y1, fm_x0:fm_x1] = 0.0
        
        return occluded_img, mask


class OcclusionMaskWrapper(object):
    """
    Wrapper transform that handles (image, mask) tuples in transform pipelines.
    
    This allows existing transforms to work with the new (image, mask) format
    by applying transforms only to the image and preserving the mask.
    """
    def __init__(self, transform):
        """
        Args:
            transform: Any torchvision transform that works on PIL Images
        """
        self.transform = transform
    
    def __call__(self, input_data):
        """
        Args:
            input_data: Either PIL Image or (PIL Image, mask) tuple
            
        Returns:
            If input is image: transformed image
            If input is (image, mask): (transformed image, mask)
        """
        if isinstance(input_data, tuple):
            img, mask = input_data
            transformed_img = self.transform(img)
            return transformed_img, mask
        else:
            # Regular image - apply transform normally
            return self.transform(input_data)


class ToTensorWithMask(object):
    """
    Convert PIL Image and optional mask to tensors.
    
    Handles both regular images and (image, mask) tuples from occlusion transforms.
    """
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, input_data):
        """
        Args:
            input_data: Either PIL Image or (PIL Image, mask) tuple
            
        Returns:
            If input is image: tensor [C, H, W]
            If input is (image, mask): (tensor [C, H, W], tensor [1, H, W])
        """
        if isinstance(input_data, tuple):
            img, mask = input_data
            img_tensor = self.to_tensor(img)
            # Convert mask to tensor and add channel dimension
            if isinstance(mask, Image.Image):
                # Convert PIL Image to numpy array first
                mask_array = np.array(mask)
                mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)  # [1, H, W]
            else:
                # Already numpy array
                mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # [1, H, W]
            
            # Normalize mask to [0, 1] range (from [0, 255])
            mask_tensor = mask_tensor / 255.0
            
            return img_tensor, mask_tensor
        else:
            # Regular image
            return self.to_tensor(input_data)
