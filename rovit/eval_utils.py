"""rovit.eval_utils (tai tao) -- eval Top-1 va calibration runner."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class _HFCollate:            # class cap module: picklable cho Windows spawn
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, items):
        imgs = [it["image"].convert("RGB") for it in items]
        x = self.processor(images=imgs, return_tensors="pt")["pixel_values"]
        y = torch.tensor([it["label"] for it in items])
        return x, y


class _TimmCollate:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, items):
        x = torch.stack([self.transform(it["image"].convert("RGB"))
                         for it in items])
        y = torch.tensor([it["label"] for it in items])
        return x, y


@torch.no_grad()
def top1_hf(model, processor, dataset, device, batch=64, workers=4):
    loader = DataLoader(dataset, batch_size=batch, num_workers=workers,
                        collate_fn=_HFCollate(processor), pin_memory=True)
    model.eval().to(device)
    correct = total = 0
    for x, y in tqdm(loader, desc="[Eval]", leave=False, ascii=True):
        logits = model(pixel_values=x.to(device, non_blocking=True)).logits
        correct += (logits.argmax(-1).cpu() == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


@torch.no_grad()
def top1_timm(model, dataset, device, batch=64, workers=4):
    import timm
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    loader = DataLoader(dataset, batch_size=batch, num_workers=workers,
                        collate_fn=_TimmCollate(tf), pin_memory=True)
    model.eval().to(device)
    correct = total = 0
    for x, y in tqdm(loader, desc="[Eval]", leave=False, ascii=True):
        out = model(x.to(device, non_blocking=True))
        correct += (out.argmax(-1).cpu() == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def hf_calibration_runner(model, processor, dataset, indices, device,
                          batch=16):
    """Tra ve callable chay forward tren tap calibration (dung voi calibrate)."""
    subset = dataset.select(list(indices))

    @torch.no_grad()
    def run():
        model.eval().to(device)
        loader = DataLoader(subset, batch_size=batch, num_workers=0,
                            collate_fn=_HFCollate(processor))
        for x, _ in loader:
            model(pixel_values=x.to(device))
    return run
