"""main script for train"""

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from lion_pytorch import Lion
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader


MEAN = [
    -8.78093929e-01,
    2.31149627e-04,
    -8.87165054e-06,
    -1.50672547e-04,
    6.37255569e-04,
    -1.07927365e-03,
    -4.81858192e-04,
    1.65434892e-03,
    -7.74792415e-04,
    -1.34467365e-03,
    -5.14731376e-05,
    -5.20111948e-04,
    -9.14031054e-04,
    -2.52067837e-05,
    4.10533209e-04,
    -5.14132211e-02,
    5.46622546e-04,
    -2.46683707e-03,
    2.50821278e-03,
    -4.36100159e-04,
    -1.52239124e-03,
    -1.11813644e-03,
    -1.66076636e-03,
    7.91338746e-04,
    4.81694509e-04,
    -2.15963732e-04,
    -7.24749836e-04,
    -8.59584620e-05,
    -2.92625341e-04,
    7.02249968e-04,
]
STD = [
    0.04241032,
    0.08804874,
    0.10358423,
    0.10418098,
    0.09123401,
    0.08382551,
    0.06144433,
    0.09149756,
    0.03516072,
    0.04931449,
    0.03839293,
    0.02955545,
    0.0223636,
    0.01611575,
    0.01187125,
    0.18677025,
    0.12452101,
    0.10360797,
    0.09657316,
    0.09395662,
    0.08389781,
    0.06266316,
    0.09240691,
    0.03509482,
    0.0479539,
    0.03845499,
    0.03026489,
    0.02249408,
    0.01653571,
    0.01178938,
]


def get_data_from_files(*paths, sort_by=None, sep="\t", header=None, **kwargs):
    """Get files data"""
    res = {}
    for filename in paths:
        df = pd.read_csv(filename, sep=sep, header=header, **kwargs)
        # names = columns
        if sort_by:
            df = df.sort_values(sort_by)
        res.update({filename: df})

    return res


def complex_to_real(alpha_arr: np.ndarray) -> np.ndarray:
    """
    Convert an array of complex numbers to a 1D float array containing their real and imaginary parts.

    Args:
        alpha_arr: An array of complex numbers to convert.

    Returns:
        A 1D numpy array containing the real and imaginary parts of the complex numbers.
        The real parts are at even indices (0, 2, 4, ...) and the imaginary parts are at odd indices (1, 3, 5, ...).

    Example:
        >>> import numpy as np
        >>> complex_array = np.array([1 + 2j, 3 - 4j, 5 + 6j])
        >>> complex_to_real(complex_array)
        array([ 1., 3., 5., 2., -4., 6.])
    """
    real_parts = alpha_arr.real
    imag_parts = alpha_arr.imag
    y = np.concatenate([real_parts, imag_parts])
    return y


class Pixels(Dataset):
    """PCM dataset"""

    def __init__(
        self,
        root: str,
        annfile: str,
        mean: list = None,
        std: list = None,
        is_scaler: bool = False,
        ratio: float = 1.0,
    ) -> None:
        self.root = root
        self.annfile = annfile
        self.ratio = ratio
        self.standard_scaler = None
        if is_scaler:
            self.standard_scaler = StandardScaler()
            self.standard_scaler.mean_ = mean
            self.standard_scaler.scale_ = std
        self._setup_annotation()

    def _setup_annotation(self) -> None:
        self.root = os.path.join(self.root, self.annfile)
        files = os.listdir(self.root)
        self.x_files = list(filter(lambda x: x.startswith("structure_"), files))

    def __len__(self) -> int:
        l = len(self.x_files)

        if not (0 < self.ratio <= 1):
            raise ValueError(
                f"The ratio ({self.ratio}) should be in the range of 0 to 1"
            )

        if self.ratio != 1:
            l = int(l * self.ratio)
        return l

    def __getitem__(self, index):
        x_file = self.x_files[index]
        x_filepath = os.path.join(self.root, x_file)
        seed = x_file.split("_")[1]
        y_filepath = os.path.join(self.root, "result_" + seed)

        data = get_data_from_files(x_filepath, y_filepath, header=0, sep=" ")

        x, y = data[x_filepath], data[y_filepath]

        y = y.iloc[64 : 64 + 15].to_numpy().flatten()  # только альфа
        y = np.array(list(map(lambda x: complex(x.replace("i", "j")), y)))
        y = complex_to_real(y)

        x = x["Unnamed: 4"].to_numpy()

        if self.standard_scaler is not None:
            y = self.standard_scaler.transform(y[None, :])[0]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        seed = int(seed.split(".")[0])
        return x, y, seed

    def collate_fn(self, samples):
        "Function to collate samples into a batch."
        pixels = []
        alphas = []

        for sample in samples:
            pixels.append(sample[0])
            alphas.append(sample[1])

        pixels = torch.stack(pixels, 0)
        alphas = torch.stack(alphas, 0)  # [N, 30]

        return alphas, pixels


class Trainer:
    """Base trainer"""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device="cpu",
        patience=100,
        experiment_dir="./experiment",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stopping_counter = 0
        self.best_loss = float("inf")
        self.patience = patience
        self.experiment_dir = experiment_dir

        self.dataset_batch_step = 0

        self.is_train_dataset = None
        self.train_valid = "train"
        self.loss_history = {"train": [], "valid": []}

        os.makedirs(self.experiment_dir, exist_ok=True)

    def run_batch(self, inputs, targets) -> None:
        """Method for 1 batch"""
        self.optimizer.zero_grad()
        loss, _ = self.model(inputs, targets)  # "_" means alphas_predicted
        self.loss_history[self.train_valid][-1] += loss.item()
        if self.train_valid == "train":
            # self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run_dataset(self, dataloader):
        """For dataset"""
        self.dataset_batch_step = 0
        self.model.train(self.train_valid == "train")
        with torch.set_grad_enabled(self.train_valid == "train"):
            for inputs, targets in tqdm(dataloader):
                self.dataset_batch_step += 1
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.run_batch(inputs, targets)

        self.loss_history[self.train_valid][-1] /= self.dataset_batch_step

    def fit(self, train_dataloader, validation_dataloader, epochs):
        """Fit method"""
        for epoch in range(epochs):
            self.loss_history["train"].append(0)
            self.loss_history["valid"].append(0)

            self.train_valid = "train"
            self.run_dataset(train_dataloader)
            self.train_valid = "valid"
            self.run_dataset(validation_dataloader)

            # Print results
            print(
                f"Epoch: {epoch+1}/{epochs}\tlr: {self.optimizer.param_groups[0]['lr']:<.2E}"
            )
            print(f"train_loss = {self.loss_history['train'][-1]:.4f}")
            print(f"valid_loss = {self.loss_history['valid'][-1]:.4f}")
            print()

            if self.loss_history["valid"][-1] < self.best_loss:
                self.best_loss = self.loss_history["valid"][-1]
                self.early_stopping_counter = 0
                model_to_save = self.model.state_dict()
                torch.save(
                    model_to_save, os.path.join(self.experiment_dir, "model.pth")
                )
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    print("Stopping training early")
                    return

            self.scheduler.step(self.loss_history["valid"][-1])

        print("Finished training")


class GeneralModel(nn.Module):
    """General model"""

    def __init__(self, num_layers, input_size, hidden_size, output_size) -> None:
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Dropout(p=0.1))
        layers.append(nn.GELU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(p=0.1))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Some"""
        return self.model(x)


class ModeModel(nn.Module):
    """
    Design Network and Mode Network
    """

    def __init__(self) -> None:
        super().__init__()

        self.mode_model = GeneralModel(
            num_layers=4, input_size=44 * 8, hidden_size=500, output_size=30
        )

        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, inputs, targets):
        """
        alphas = [N, 30]
        pixels = [N, 44*8]
        """

        alphas, pixels = inputs, targets
        alphas_predicted = self.mode_model(pixels)
        mse = self.mse_loss(alphas_predicted, alphas)

        return mse, alphas_predicted


def plot_loss(train_loss, valid_loss, path_to_save):
    """
    Function to plot train and validation loss.

    Parameters:
    train_loss (list): List of training loss for each epoch.
    valid_loss (list): List of validation loss for each epoch.

    Returns:
    None
    """
    epochs = np.arange(len(train_loss))

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, color="darkorange", lw=2, label="Training Loss")
    plt.plot(epochs, valid_loss, color="#aab6d4", lw=2, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Save the ROC curve plot to the specified path
    plt.savefig(path_to_save, bbox_inches="tight", transparent=True)

    # Close the plot to free up memory (optional)
    plt.close()


def main():
    """main"""
    batch_size = 32
    num_epochs = 20000
    workers = 4
    experiment_dir = "./experiment"
    path_to_data = "./input/sweep"

    model = ModeModel()
    optimizer = Lion(model.parameters(), lr=1e-5, betas=(0.9, 0.99), weight_decay=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, verbose=True, mode="min", factor=0.1, patience=20, threshold=1e-4
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device = }")

    train_dataset = Pixels(
        root=path_to_data,
        annfile="train",
        is_scaler=True,
        mean=MEAN,
        std=STD,
    )

    valid_dataset = Pixels(
        root=path_to_data,
        annfile="valid",
        is_scaler=True,
        mean=MEAN,
        std=STD,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=train_dataset.collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=valid_dataset.collate_fn,
    )

    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        device=device,
        patience=100,
        experiment_dir=experiment_dir,
    )

    trainer.fit(
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader,
        epochs=num_epochs,
    )

    plot_loss(
        trainer.loss_history["train"],
        trainer.loss_history["valid"],
        path_to_save=os.path.join(experiment_dir, "loss.pdf"),
    )


if __name__ == "__main__":
    main()
