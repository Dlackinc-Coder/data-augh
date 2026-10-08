import segmentation_models_pytorch as smp


def build_model(cfg):
    m = cfg["model"]
    enc = m.get("encoder", "efficientnet-b0")
    weights = m.get("encoder_weights", "imagenet")
    classes = m.get("classes", 4)
    model = smp.Unet(
        encoder_name=enc,
        encoder_weights=weights,
        in_channels=3,
        classes=classes,
        activation=None,
    )
    return model
