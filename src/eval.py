import torch


def dice_multiclass(logits, targets, eps=1e-6, ignore_index=None):
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    num_classes = probs.shape[1]
    dices = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        p = (pred == c).float()
        g = (targets == c).float()
        inter = (p * g).sum()
        dices.append((2 * inter + eps) / (p.sum() + g.sum() + eps))
    return torch.stack(dices).mean().item()


def iou_multiclass(logits, targets, eps=1e-6):
    pred = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    ious = []
    for c in range(num_classes):
        p = (pred == c).float()
        g = (targets == c).float()
        ious.append(((p * g).sum()) / ((p + g).sum() - (p * g).sum() + eps))
    return torch.stack(ious).mean().item()


@torch.no_grad()
def evaluate(model, loader, device, half=True):
    model.eval()
    dices, ious = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.cuda.amp.autocast(enabled=half):
            logits = model(imgs)
        dices.append(dice_multiclass(logits.float(), masks))
        ious.append(iou_multiclass(logits.float(), masks))
    return sum(dices) / len(dices), sum(ious) / len(ious)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from train_seg import load_cfg
    from dataset import make_loaders
    from seg_model import build_model

    cfg = load_cfg(sys.argv[1] if len(sys.argv) > 1 else "configs/seg.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_dl = make_loaders(cfg)
    model = build_model(cfg).to(device)
    ckpt = torch.load(cfg.get("ckpt_path", "checkpoints/best.pt"),
                      map_location=device)
    model.load_state_dict(ckpt["model"])
    d, i = evaluate(model, test_dl, device)
    print(f"TEST | dice {d:.4f} | iou {i:.4f} (ckpt epoch {ckpt.get('epoch')})")
