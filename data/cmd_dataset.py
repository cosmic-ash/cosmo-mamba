"""
CAMELS Multifield Dataset (CMD) data loading for cosmological parameter inference.

Data format (from CMD docs):
- 2D maps: .npy files with shape (N_maps, 256, 256), where N_maps = N_sims * 15
- Parameters: .txt files with shape (N_sims, 6) → [Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2]
- Map i belongs to simulation i // 15
- Available fields: Mgas, Vgas, T, P, Z, HI, ne, B, MgFe, Mcdm, Vcdm, Mstar, Mtot
- Available suites: IllustrisTNG, SIMBA, Astrid, Nbody
- LH set: 1000 simulations × 15 maps = 15,000 maps per field
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

FIELDS = ["Mgas", "Vgas", "T", "P", "Z", "HI", "ne", "B", "MgFe", "Mcdm", "Vcdm", "Mstar", "Mtot"]
SUITES = ["IllustrisTNG", "SIMBA", "Astrid", "Nbody"]
MAPS_PER_SIM = 15


class CMDDataset(Dataset):
    """
    Dataset for CMD 2D maps → cosmological parameter regression.

    Each sample is a single-channel 256×256 map paired with target
    cosmological parameters (Omega_m, sigma_8).
    Supports loading multiple fields as separate channels for multi-field inference.
    """

    def __init__(
        self,
        data_dir: str,
        fields: list[str],
        suite: str = "IllustrisTNG",
        set_name: str = "LH",
        split: str = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        normalize: bool = True,
        augment: bool = False,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.fields = fields
        self.suite = suite
        self.set_name = set_name
        self.split = split
        self.normalize = normalize
        self.augment = augment and (split == "train")

        params_file = os.path.join(data_dir, f"params_{set_name}_{suite}.txt")
        if suite.startswith("Nbody"):
            params_file = os.path.join(data_dir, f"params_{set_name}_Nbody.txt")
        self.params = np.loadtxt(params_file)
        n_sims = len(self.params)
        n_maps = n_sims * MAPS_PER_SIM

        self.maps = {}
        self.field_stats = {}
        for field in fields:
            fname = f"Maps_{field}_{suite}_{set_name}_z=0.00.npy"
            fpath = os.path.join(data_dir, fname)
            maps = np.load(fpath, mmap_mode="r")
            assert maps.shape[0] == n_maps, (
                f"Expected {n_maps} maps for {field}, got {maps.shape[0]}"
            )
            self.maps[field] = maps
            if normalize:
                sample = maps[:1000].astype(np.float32)
                nonzero = sample[sample != 0]
                if len(nonzero) > 0:
                    log_data = np.log10(np.abs(nonzero) + 1e-10)
                    self.field_stats[field] = (float(np.mean(log_data)), float(np.std(log_data)))
                else:
                    self.field_stats[field] = (0.0, 1.0)

        rng = np.random.RandomState(seed)
        sim_indices = np.arange(n_sims)
        rng.shuffle(sim_indices)

        n_train = int(n_sims * train_frac)
        n_val = int(n_sims * val_frac)

        if split == "train":
            sims = sim_indices[:n_train]
        elif split == "val":
            sims = sim_indices[n_train : n_train + n_val]
        elif split == "test":
            sims = sim_indices[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.indices = []
        for s in sims:
            for m in range(MAPS_PER_SIM):
                self.indices.append(s * MAPS_PER_SIM + m)
        self.sim_for_map = {idx: idx // MAPS_PER_SIM for idx in self.indices}

        self.target_mean = np.mean(self.params[:, :2], axis=0)
        self.target_std = np.std(self.params[:, :2], axis=0)

    def __len__(self):
        return len(self.indices)

    def _normalize_map(self, x: np.ndarray, field: str) -> np.ndarray:
        x = x.astype(np.float32)
        x = np.log10(np.abs(x) + 1e-10)
        mean, std = self.field_stats[field]
        if std > 0:
            x = (x - mean) / std
        return x

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Random rotations (0/90/180/270) and flips — these are symmetries of the CMD maps."""
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=(-2, -1))
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=(-1,))
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=(-2,))
        return x

    def __getitem__(self, idx):
        map_idx = self.indices[idx]
        sim_idx = map_idx // MAPS_PER_SIM

        channels = []
        for field in self.fields:
            m = self.maps[field][map_idx]
            if self.normalize:
                m = self._normalize_map(m, field)
            else:
                m = m.astype(np.float32)
            channels.append(m)

        x = torch.tensor(np.stack(channels, axis=0))

        if self.augment:
            x = self._augment(x)

        target = self.params[sim_idx, :2].astype(np.float32)
        y = torch.tensor(target)

        return x, y

    def get_target_stats(self):
        return self.target_mean, self.target_std


def get_data_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    common = dict(
        data_dir=data_cfg["data_dir"],
        fields=data_cfg["fields"],
        suite=data_cfg["suite"],
        set_name=data_cfg.get("set_name", "LH"),
        train_frac=data_cfg.get("train_frac", 0.8),
        val_frac=data_cfg.get("val_frac", 0.1),
        normalize=data_cfg.get("normalize", True),
        seed=data_cfg.get("seed", 42),
    )

    train_ds = CMDDataset(**common, split="train", augment=data_cfg.get("augment", True))
    val_ds = CMDDataset(**common, split="val", augment=False)
    test_ds = CMDDataset(**common, split="test", augment=False)

    loader_kwargs = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
