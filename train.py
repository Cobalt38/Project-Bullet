"""
Training rete neurale per IK (Inverse Kinematics).

Uso:
    python train.py                          # training completo
    python train.py --csv altro_dataset.csv  # dataset custom
    python train.py --epochs 20 --batch 2048 --hidden 512
    python train.py --fast-dev              # smoke test su 10k righe

Output (nella cartella --out-dir, default ./ik_training/):
    scaler_x.pkl        scaler input  (joblib)
    scaler_y.pkl        scaler output (joblib)
    best_model.ckpt     checkpoint Lightning con pesi migliori
    last_model.ckpt     checkpoint ultimo epoch
    hparams.yaml        iperparametri salvati da Lightning
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
#  COLONNE  (speculari a main_parallelizzato.py)
# ─────────────────────────────────────────────

INPUT_COLS  = ["target_x", "target_y", "target_z",
               "hand_quat_qx", "hand_quat_qy", "hand_quat_qz", "hand_quat_qw"]
OUTPUT_COLS = [f"joint_{i}" for i in range(7)]   # joint_0 … joint_6


# ─────────────────────────────────────────────
#  DATASET  — lettura a chunk per gestire 98M righe
# ─────────────────────────────────────────────

class IKDataset(Dataset):
    """
    Carica il CSV in chunk per non esaurire la RAM, poi
    converte tutto in un tensore numpy contiguo in memoria.
    Per dataset >20 GB considera la versione MemMap (vedi commento finale).
    """

    def __init__(self, csv_path: str, scaler_x=None, scaler_y=None,
                 max_rows: int = None, chunksize: int = 500_000):
        print(f"Lettura {csv_path} ...")
        chunks_x, chunks_y = [], []
        rows_read = 0

        for chunk in pd.read_csv(csv_path, chunksize=chunksize,
                                 usecols=INPUT_COLS + OUTPUT_COLS):
            if max_rows and rows_read >= max_rows:
                break
            if max_rows:
                chunk = chunk.iloc[:max_rows - rows_read]

            chunks_x.append(chunk[INPUT_COLS].values.astype(np.float32))
            chunks_y.append(chunk[OUTPUT_COLS].values.astype(np.float32))
            rows_read += len(chunk)
            print(f"  lette {rows_read:,} righe...", end="\r")

        print(f"\n  Totale: {rows_read:,} righe")
        self.X = np.concatenate(chunks_x, axis=0)
        self.Y = np.concatenate(chunks_y, axis=0)

        # Fit degli scaler solo se non forniti (fase training)
        if scaler_x is None:
            print("Fit StandardScaler input (solo posizione, non quaternioni)...")
            self.scaler_x = StandardScaler()
            # Normalizza solo xyz; i quaternioni sono già su sfera unitaria
            self.X[:, :3] = self.scaler_x.fit_transform(self.X[:, :3])
        else:
            self.scaler_x = scaler_x
            self.X[:, :3] = self.scaler_x.transform(self.X[:, :3])

        if scaler_y is None:
            print("Fit StandardScaler output (giunti)...")
            self.scaler_y = StandardScaler()
            self.Y = self.scaler_y.fit_transform(self.Y).astype(np.float32)
        else:
            self.scaler_y = scaler_y
            self.Y = self.scaler_y.transform(self.Y).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


# ─────────────────────────────────────────────
#  ARCHITETTURA  — Residual MLP
# ─────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Blocco con skip connection: aiuta il gradiente a fluire in reti profonde."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class IKNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int, n_blocks: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# ─────────────────────────────────────────────
#  LIGHTNING MODULE
# ─────────────────────────────────────────────

class LitIKRegressor(pl.LightningModule):
    def __init__(self, input_dim=7, output_dim=7,
                 hidden_dim=512, n_blocks=4, dropout=0.1,
                 lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = IKNet(input_dim, output_dim, hidden_dim, n_blocks, dropout)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        mae  = F.l1_loss(y_hat, y)
        return loss, mae

    def training_step(self, batch, _):
        loss, mae = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mae",  mae,  prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, mae = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae",  mae,  prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Cosine annealing: lr scende gradualmente fino a lr/100
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=self.lr / 100
        )
        return {"optimizer": opt, "lr_scheduler": sched}


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Training IK neurale")
    parser.add_argument("--csv",      default="dataset.csv")
    parser.add_argument("--out-dir",  default="ik_training")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--batch",    type=int,   default=1024)
    parser.add_argument("--hidden",   type=int,   default=512)
    parser.add_argument("--blocks",   type=int,   default=4,
                        help="Numero di ResidualBlock")
    parser.add_argument("--dropout",  type=float, default=0.1)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Frazione del dataset per validation (default 5%%)")
    parser.add_argument("--workers",  type=int,   default=4,
                        help="num_workers per DataLoader")
    parser.add_argument("--fast-dev", action="store_true",
                        help="Smoke test: carica solo 10k righe, 2 epoch")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Dataset ──
    max_rows = 10_000 if args.fast_dev else None
    epochs   = 2      if args.fast_dev else args.epochs

    full_dataset = IKDataset(args.csv, max_rows=max_rows)

    # Salva scaler su disco — necessari per l'inferenza
    scaler_x_path = os.path.join(args.out_dir, "scaler_x.pkl")
    scaler_y_path = os.path.join(args.out_dir, "scaler_y.pkl")
    joblib.dump(full_dataset.scaler_x, scaler_x_path)
    joblib.dump(full_dataset.scaler_y, scaler_y_path)
    print(f"Scaler salvati: {scaler_x_path}, {scaler_y_path}")

    # Split train / val
    n_val   = max(1, int(len(full_dataset) * args.val_frac))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Split → train: {n_train:,}  |  val: {n_val:,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch * 2, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)

    # ── Modello ──
    model = LitIKRegressor(
        input_dim=len(INPUT_COLS),
        output_dim=len(OUTPUT_COLS),
        hidden_dim=args.hidden,
        n_blocks=args.blocks,
        dropout=args.dropout,
        lr=args.lr,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri rete: {total_params:,}")

    # ── Callbacks ──
    checkpoint_best = ModelCheckpoint(
        dirpath=args.out_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,         # tieni solo il migliore
        verbose=True,
    )
    checkpoint_last = ModelCheckpoint(
        dirpath=args.out_dir,
        filename="last_model",
        save_last=True,       # aggiornato a ogni epoch
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,          # ferma se val_loss non migliora per 10 epoch
        mode="min",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── Trainer ──
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_best, checkpoint_last, early_stop, lr_monitor],
        log_every_n_steps=50,
        default_root_dir=args.out_dir,
    )

    print("\nAvvio training...")
    trainer.fit(model, train_loader, val_loader)

    print(f"\n{'='*50}")
    print(f"Training completato.")
    print(f"  Miglior val_loss epoch: {checkpoint_best.best_model_score:.6f}")
    print(f"  Checkpoint migliore   : {checkpoint_best.best_model_path}")
    print(f"  Checkpoint ultimo     : {os.path.join(args.out_dir, 'last_model.ckpt')}")
    print(f"  Scaler input          : {scaler_x_path}")
    print(f"  Scaler output         : {scaler_y_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
