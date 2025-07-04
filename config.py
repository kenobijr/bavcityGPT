from dataclasses import dataclass
import os, datetime

"""
Bavarian City Name GPT // config classes for:
- training
- sampling
- data processing
"""


@dataclass
class TrainConfig:
    """training configuration"""

    batch_size: int = 64
    learning_rate: float = 3e-4
    train_iter: int = 1
    eval_iter: int = 150
    eval_interval: int = 500
    device: str = "mps"
    # dir with bin / meta files for training
    data_dir: str = "data"
    # save model after train
    saved_models_root: str = "saved_models"
    model_name: str = "bavGPT"
    # seed for torch
    seed: int = 42
    # print samples after training
    num_samples: int = 20

    @property
    def save_dir_current(self) -> str:
        """e.g. saved_models/bavGPT_20250703_173529"""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.saved_models_root,
                            f"{self.model_name}_{ts}")


@dataclass
class SampleConfig:
    """
    - sampling configuration
    - samples are always saved as .txt in model dir
    """

    device: str = "mps"
    num_samples: int = 25
    max_length: int = 50
    temperature: float = 1.0


@dataclass
class DataConfig:
    """
    - path to load in raw input data
    - path to load processed datasets so
    - other data processing config
    """

    # data processing
    input_file: str = "data/names.txt"
    output_dir: str = "data"
    # seed for shuffling names
    seed: int = 42
    # raw data validation
    min_name_length: int = 3
    max_name_length: int = 50
    # split sizes for train / dev; rest it test
    train_size: float = 0.8
    dev_size: float = 0.1
    test_size: float = 0.1
