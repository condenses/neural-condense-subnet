from typing import Iterator
from datasets import IterableDataset


class InfiniteDataset:
    def __init__(self, dataset: IterableDataset):
        """
        Initialize the InfiniteDataset wrapper.

        Args:
            dataset (IterableDataset): The dataset object to wrap. It should be an iterable dataset.
        """
        self.dataset = dataset
        self.iterator = iter(self.dataset)  # Initialize the iterator

    def __iter__(self) -> Iterator:
        """
        Return the iterator for the dataset.

        Returns:
            Iterator: An iterator over the dataset.
        """
        return self

    def __next__(self):
        """
        Get the next item in the dataset. Automatically reinitialize the iterator if the end is reached.

        Returns:
            The next item in the dataset.
        """
        try:
            return next(self.iterator)
        except StopIteration:
            # Reinitialize iterator if the end is reached
            self.iterator = iter(self.dataset)
            return next(self.iterator)  # Return the first item of the new iterator
