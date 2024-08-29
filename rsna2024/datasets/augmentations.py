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
                                A.MotionBlur(blur_limit=5, p=1),
                                A.MedianBlur(blur_limit=5),
                                # A.GaussianBlur(blur_limit=(1, 5)),
                                A.GaussianBlur(blur_limit=5),
                                A.GaussNoise(var_limit=(5.0, 30.0)),
                            ], p=cfg.prob))
        if cfg.distort:
            use.append(A.OneOf([
                                A.OpticalDistortion(distort_limit=0.05, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_LINEAR, value=0),
                                A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_LINEAR, value=0, normalized=True),
                                A.ElasticTransform(alpha=3, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_LINEAR, value=0),
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