import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


def build_unet(cfg):
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


def build_dpt(cfg):
    """DPT (Dense Prediction Transformer) con backbone timm via smp.

    Soporta: Swin, ViT, PVTv2, ConvNeXt y cualquier otro timm model.
    El config debe indicar el encoder con prefijo 'tu-' (timm universal),
    p. ej. encoder: "tu-swin_tiny_patch4_window7_224".

    Campos opcionales en model:
      - encoder_kwargs: dict de argumentos para timm (p. ej. img_size: 512)
      - readout: "cat" | "add" | "ignore" (tokens prefix/cls)
      - encoder_depth: niveles de features (default 4)
      - freeze_encoder_epochs: épocas con encoder congelado
    """
    m = cfg["model"]
    enc = m.get("encoder")
    if not enc:
        raise ValueError("model.encoder es requerido para DPT (ej. tu-swin_tiny_patch4_window7_224)")
    if not enc.startswith("tu-"):
        enc = "tu-" + enc
    weights = m.get("encoder_weights", "imagenet")
    classes = m.get("classes", 4)
    depth = m.get("encoder_depth", 4)
    readout = m.get("readout", "cat")
    enc_kwargs = m.get("encoder_kwargs", {}) or {}
    freeze_encoder_epochs = m.get("freeze_encoder_epochs", 0)

    model = smp.DPT(
        encoder_name=enc,
        encoder_weights=weights,
        encoder_depth=depth,
        decoder_readout=readout,
        in_channels=3,
        classes=classes,
        activation=None,
        **enc_kwargs,
    )
    model._freeze_encoder_epochs = freeze_encoder_epochs
    model._encoder_frozen = False
    return model


def build_segformer(cfg):
    from transformers import SegformerForSemanticSegmentation
    
    m = cfg["model"]
    encoder_name = m.get("encoder", "mit_b2")
    pretrained = m.get("pretrained", True)
    classes = m.get("classes", 4)
    freeze_encoder_epochs = m.get("freeze_encoder_epochs", 0)
    
    encoder_map = {
        "mit_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "mit_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
        "mit_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "mit_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
        "mit_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
        "mit_b5": "nvidia/segformer-b5-finetuned-ade-640-640",
    }
    
    model_id = encoder_map.get(encoder_name, "nvidia/segformer-b2-finetuned-ade-512-512")
    
    if pretrained:
        # Use safetensors to avoid torch.load vulnerability issue
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )
    else:
        from transformers import SegformerConfig
        config = SegformerConfig.from_pretrained(
            encoder_map.get(encoder_name, "nvidia/segformer-b2-finetuned-ade-512-512"),
            num_labels=classes,
        )
        model = SegformerForSemanticSegmentation(config)
    
    model._freeze_encoder_epochs = freeze_encoder_epochs
    model._encoder_frozen = False
    
    return model


class LiteMedSAMSeg(nn.Module):
    def __init__(self, encoder, num_classes=4):
        super().__init__()
        self.encoder = encoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_enc = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        feat = self.encoder(x_enc)
        x = self.dec1(feat)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        logits = self.classifier(x)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


def build_litemedsam(cfg):
    from tiny_vit_sam import TinyViT

    m = cfg["model"]
    classes = m.get("classes", 4)
    ckpt_path = m.get("pretrained_checkpoint", "checkpoints/lite_medsam.pth")
    freeze_encoder_epochs = m.get("freeze_encoder_epochs", 0)

    encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )

    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        enc_state = {
            k.replace("image_encoder.", ""): v
            for k, v in ckpt.items()
            if k.startswith("image_encoder.")
        }
        missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
        print(f"[LiteMedSAM] Loaded encoder weights: {len(enc_state)} tensors, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        del ckpt
    else:
        print(f"[LiteMedSAM] WARNING: checkpoint not found at {ckpt_path}, using random init")

    model = LiteMedSAMSeg(encoder, num_classes=classes)
    model._freeze_encoder_epochs = freeze_encoder_epochs
    model._encoder_frozen = False
    return model


def freeze_encoder(model, freeze=True):
    if hasattr(model, 'segformer'):
        for param in model.segformer.encoder.parameters():
            param.requires_grad = not freeze
        model._encoder_frozen = freeze
    elif hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = not freeze
        model._encoder_frozen = freeze


def build_model(cfg):
    m = cfg["model"]
    model_type = m.get("name", "unet")
    
    if model_type == "unet":
        return build_unet(cfg)
    elif model_type == "segformer":
        return build_segformer(cfg)
    elif model_type == "litemedsam":
        return build_litemedsam(cfg)
    elif model_type == "dpt":
        return build_dpt(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
