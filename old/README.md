# Multiple lanes segmentation

## Data
1. **Images**: 1280x512x3 (uint8 0-255)
2. **Masks**: 1280x512 (uint8 0-2)

## Datasets
* **Innopolis**: 633 images
  > **2018-11-30**: 319 images
  > **2018-12-04**: 207 images
  > **2018-12-10**: 107 images

* **Polygon (31st January)**: 1111 images
  > **11-27-12**: 171 images
  > **11-31-11**: 186 images
  > **11-35-56**: 68 images
  > **11-38-11**: 249 images
  > **12-34-55**: 115 images
  > **14-12-33**: 105 images
  > **14-18-09**: 110 images
  > **14-26-27**: 11 images
  > **15-11-58**: 31 images
  > **15-14-17**: 29 images
  > **15-17-03**: 36 images
## Masks

## Training
### 1. Preprocessing
* **Image**:
  1. Read from directory
  2. Resize to 640x256x3
  3. Convert to numpy ndarray
  4. Convert to float
  5. Normalize

* **Mask**:
  1. Read from directory
  2. Resize to 640x256
  3. Convert to numpy ndarray
  4. One hot encoding (i.e. 640x256x3)
  5. Convert to integer

### 2. Training process

#### Augmentation
##### Using [**albumentations**](https://albumentations.readthedocs.io/en/latest/index.html) library: 

```python
from albumentations import (
    OneOf,
    Blur,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MedianBlur,
    CLAHE
)

aug = OneOf([
        Blur(blur_limit=5, p=1.),
        RandomGamma(gamma_limit=(50, 150), p=1.),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.),
        RGBShift(r_shift_limit=15, g_shift_limit=5, b_shift_limit=15, p=1.),
        RandomBrightness(limit=.25, p=1.),
        RandomContrast(limit=.25, p=1.),
        MedianBlur(blur_limit=5, p=1.),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.)
        ], p=1.)
        
def augment(image, aug=aug):
        augmented = aug(image=image)
        return augmented['image']
```

### 3. Postprocessing
### 4. Results
