import albumentations as A
import cv2



def get_transform(train, cfg):
    if train:
        use = []

        if cfg.hflip:
            use.append(A.HorizontalFlip(0.5))
        if cfg.contrast:
            use.append(A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=cfg.prob))
        if cfg.blur:
            use.append(A.OneOf([
                                A.MotionBlur(blur_limit=cfg.blur_limit, p=1),
                                A.MedianBlur(blur_limit=cfg.blur_limit, p=1),
                                A.GaussianBlur(blur_limit=cfg.blur_limit, p=1),
                            ], p=cfg.prob))
        if cfg.noise:
            use.append(A.OneOf([A.GaussNoise(var_limit=(5.0, 100.0), per_channel=True, p=1),
                                ],p=cfg.prob))
        if cfg.downscale:
            use.append(A.OneOf([A.Downscale(scale_min=0.5, scale_max=0.9, interpolation_pair={"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_LINEAR}, p=1),
                                ],p=cfg.prob))
        if cfg.distort:
            use.append(A.OneOf([
                                A.OpticalDistortion(distort_limit=0.05, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_LINEAR, value=0, p=1),
                                A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_LINEAR, value=0, normalized=True, p=1),
                                A.ElasticTransform(alpha=3, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_LINEAR, value=0, p=1),
                            ], p=cfg.prob))
        if cfg.rotate:
            use.append(A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_LINEAR, value=0, p=cfg.prob))
        if cfg.normalize:
            use.append(A.Normalize(mean=0.5, std=0.5))
        if cfg.crop.use:
            use.append(A.RandomResizedCrop(width=cfg.crop.size, height=cfg.crop.size, ratio=(0.99, 1.01), scale=(0.7, 1.0)))
        if cfg.channel_shuffle:
            use.append(A.ChannelShuffle(p=0.5))
        if cfg.sharpen:
            use.append(A.Sharpen(p=0.5))
        return A.Compose(use)
    else:
        use = []
        if cfg.normalize:
            use.append(A.Normalize(mean=0.5, std=0.5))
        return A.Compose(use)