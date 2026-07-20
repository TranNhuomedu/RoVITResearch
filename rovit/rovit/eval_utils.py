"""ImageNet Top-1 evaluation for HuggingFace and timm model backends."""

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import HFCollate

TIMM_TRANSFORM = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def top1_hf(model, processor, dataset, device, batch_size=64, workers=4,
            max_batches=None, desc="eval"):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=HFCollate(processor),
                        num_workers=workers, pin_memory=True)
    correct = total = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            labels = batch.pop("labels").to(device)
            preds = model(**{k: v.to(device) for k, v in batch.items()}
                          ).logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def _timm_collate(batch, transform=TIMM_TRANSFORM):
    imgs = torch.stack([transform(e["image"].convert("RGB")) for e in batch])
    labels = torch.tensor([e["label"] for e in batch])
    return imgs, labels


def top1_timm(model, dataset, device, batch_size=64, workers=4,
              max_batches=None, desc="eval"):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=_timm_collate,
                        num_workers=workers, pin_memory=True)
    correct = total = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(loader, desc=desc, leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            preds = model(imgs.to(device)).argmax(dim=-1)
            correct += (preds == labels.to(device)).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def hf_calibration_runner(model, processor, dataset, indices, device):
    """Return a zero-arg callable that forwards the calibration images."""
    def run():
        model.eval()
        with torch.no_grad():
            for i in tqdm(indices, desc="calibrate", leave=False):
                img = dataset[int(i)]["image"].convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)
                model(**inputs)
    return run
