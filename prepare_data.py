import os
import random
import numpy as np
from config import DataConfig
from typing import Dict, List, Tuple
import pickle


"""
Bavarian City Name GPT // data prep & encoding
- read in dataset of 60k bavarian city names and do some processing
- default params / options are saved in config.py -> DataConfig
- expects a input .txt file with continious stream of name strings sep by \n
- saves train / dev / test splits bin files & metadata to output_dir specified in DataConfig
"""


class NameProcessor:

    def __init__(self, config: DataConfig):
        assert isinstance(config, DataConfig), "Invalid config type."
        self.config = config
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.rng = random.Random(self.config.seed)

    def _load_raw_data(self) -> List[str]:
        """
        - loads all names into memory as list of str
        - checks names for len boundaries & shuffles them with seed from config
        """
        print(f"Loading data from {self.config.input_file}")
        assert os.path.exists(self.config.input_file), f"File not found: {self.config.input_file}"
        with open(self.config.input_file, mode="r", encoding="utf-8") as file:
            # excplicitly don't strip -> newline chars are kept for model
            names = file.readlines()
        # check names for certain criteria
        names = [name for name in names if self._is_valid_name(name)]
        print(f"Loaded {len(names)} valid names")
        print(f"Sample of first 5 names: {names[:5]}")
        return names

    def _is_valid_name(self, name: str) -> bool:
        """simple check if name meets criteria; as of now, check only len"""
        return self.config.min_name_length <= len(name) <= self.config.max_name_length
   
    def _shuffle_names(self, names: List[str]) -> List[str]:
        """ shuffle names using seeded RNG, returns a new shuffled list """
        shuffled_names = names.copy()
        self.rng.shuffle(shuffled_names)
        return shuffled_names

    def _build_vocabulary(self, names: List[str]) -> None:
        """creates mapping dicts from all distinct chars"""
        all_chars = sorted(set(''.join(names)))
        self.itos = {i: s for i, s in enumerate(all_chars)}
        self.stoi = {s: i for i, s in self.itos.items()}
        self.vocab_size = len(all_chars)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"All Characters: {repr(''.join(all_chars))}")

    def encode(self, text: str) -> List[int]:
        """encodes str into list of indexes with mapping dict"""
        return [self.stoi[i] for i in text]

    def decode(self, indexes: List[int]) -> str:
        """returns joined string for list of indexes with mapping dict"""
        return "".join([self.itos[i] for i in indexes])

    def _create_splits(self, names: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """split names into train / dev / test"""
        boundary_1 = int(self.config.train_size * len(names))
        boundary_2 = int((self.config.train_size + self.config.dev_size) * len(names))
        boundary_3 = int((1 - self.config.test_size) * len(names))
        train = names[:boundary_1]
        dev = names[boundary_1:boundary_2]
        test = names[boundary_3:]
        print(f"Train has {len(train):,} tokens")
        print(f"Dev has {len(dev):,} tokens")
        print(f"Test has {len(test):,} tokens")
        return train, dev, test

    def _export_data(self, splits: Tuple[List[int], List[int], List[int]]) -> None:
        """ convert idx splits in uint16 save bin files & metadata at output_dir of DataConfig """
        split_names = ["train", "dev", "test"]
        for name, data in zip(split_names, splits):
            np.array(data, dtype=np.uint16).tofile(
                os.path.join(self.config.output_dir, f"{name}.bin")
            )
        meta = {
            "vocab_size": self.vocab_size,
            "itos": self.itos,
            "stoi": self.stoi,
        }
        with open(os.path.join(self.config.output_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    def execute(self) -> None:
        """
        - load raw names from file & shuffle them
        - build mapping dicts from names
        - encode the data into indexes from joined string
        - create splits & export to np bin files & save some metadata
        """
        names = self._load_raw_data()
        names = self._shuffle_names(names)
        self._build_vocabulary(names)
        encoded_data = self.encode("".join(names))
        train, dev, test = self._create_splits(encoded_data)
        self._export_data((train, dev, test))
        # print stats
        print("\nData processing completed successfully!")
        print(f"Total tokens: {len(encoded_data):,}")


def main():
    """
    main entry point; execute data processing by:
        1. creating instance of DataConfig
        2. creating instance of NameProcessor with config
        3. call NameProcessor execute method
    """
    processor = NameProcessor(DataConfig())
    processor.execute()


if __name__ == "__main__":
    main()
