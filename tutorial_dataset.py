import json
import random
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
import albumentations


class MyDataset(Dataset):
    def __init__(self, root='./data/prompt.json'):
        self.manifest_path = Path(root).resolve()
        self.data = []
        with open(self.manifest_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self._transform = albumentations.Compose(
            [albumentations.Resize(height=384, width=384)]
        )

    def _resolve_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        candidates = []

        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(Path.cwd() / p)
            candidates.append(self.manifest_path.parent / p)
            candidates.append(self.manifest_path.parent.parent / p)
            candidates.append(Path.cwd() / 'data' / p)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not resolve dataset path '{raw_path}'. Tried: "
            + ", ".join(str(c) for c in candidates)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = str(self._resolve_path(item['source']))
        target_filename = str(self._resolve_path(item['target']))
        prompt_target = item['prompt_target']

        p = random.random()
        if p > 0.95:
            prompt_target = ""

        source = Image.open(source_filename).convert('L')
        source_array = np.array(source)
        threshold = 127
        binary_array = np.where(source_array > threshold, 255, 0).astype(np.uint8)
        binary_image = Image.fromarray(binary_array)
        source = binary_image.convert('RGB')

        target = Image.open(target_filename).convert('RGB')

        source = np.array(source).astype(np.uint8)
        target = np.array(target).astype(np.uint8)

        preprocess = self._transform(image=target, mask=source)
        source, target = preprocess['mask'], preprocess['image']

        ############ Mask-Image Pair ############
        source = source.astype(np.float32) / 255.0
        target = target.astype(np.float32) / 127.5 - 1.0

        return dict(jpg=target, txt=prompt_target, hint=source)
