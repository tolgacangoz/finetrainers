import torch
import unittest
from finetrainers.data.dataset import IterableDatasetPreprocessingWrapper

class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, num_samples=100):
        super().__init__()
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            yield {"caption": f"caption_{i}", "image": i}

class TestIterableDatasetMultiWorker(unittest.TestCase):
    def test_no_duplication_with_multiple_workers(self):
        """
        Tests that IterableDatasetPreprocessingWrapper in conjunction with a DataLoader:
        1. Does not yield duplicate data with multiple workers.
        2. Correctly drops last batch with multiple workers if data is not divisible.
        """
        num_samples = 101  # Not perfectly divisible by batch_size to test drop_last
        batch_size = 4

        for num_workers in range(9):
            with self.subTest(num_workers=num_workers):
                drop_last = num_workers > 1

                # 1. Create a dummy iterable dataset
                original_dataset = DummyIterableDataset(num_samples)

                # 2. Wrap it with the preprocessing wrapper
                wrapped_dataset = IterableDatasetPreprocessingWrapper(
                    dataset=original_dataset,
                    dataset_type="image",
                )

                # 3. Use a DataLoader with multiple workers
                dataloader = torch.utils.data.DataLoader(
                    wrapped_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=drop_last,
                )

                # 4. Collect all items loaded by the DataLoader
                loaded_items = []
                for batch in dataloader:
                    loaded_items.extend(batch['image'].tolist())

                # Calculate the expected number of samples that should be loaded
                if drop_last:
                    expected_num_loaded = 0
                    for worker_id in range(num_workers):
                        # The implementation in `IterableDatasetPreprocessingWrapper` uses `itertools.islice`
                        # to partition data, which results in an interleaved distribution (worker 0 gets
                        # items 0, N, 2N, ...; worker 1 gets 1, N+1, 2N+1, ... where N is num_workers).
                        num_samples_per_worker = (num_samples - 1 - worker_id) // num_workers + 1
                        # With drop_last=True, only full batches are considered
                        num_batches_per_worker = num_samples_per_worker // batch_size
                        expected_num_loaded += num_batches_per_worker * batch_size
                else:  # This case is for num_workers <= 1
                    expected_num_loaded = num_samples

                # 5. Assert that the number of loaded items is as expected and there are no duplicates
                self.assertEqual(len(loaded_items), expected_num_loaded, f"Total number of loaded items mismatch for {num_workers} workers.")
                self.assertEqual(len(set(loaded_items)), expected_num_loaded, f"There should be no duplicate items and count should match expected for {num_workers} workers.")

if __name__ == '__main__':
    unittest.main()