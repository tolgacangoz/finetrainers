import json
import os
import pathlib
import tempfile
import time
import unittest

import pytest
import torch
from diffusers.utils import export_to_video
from PIL import Image

from finetrainers import BaseArgs, SFTTrainer, TrainingType, get_logger


os.environ["WANDB_MODE"] = "disabled"
os.environ["FINETRAINERS_LOG_LEVEL"] = "INFO"

from ..models.flux.base_specification import DummyFluxModelSpecification


logger = get_logger()


@pytest.fixture(autouse=True)
def slow_down_tests():
    yield
    time.sleep(5)


class FSDPLoRATest(unittest.TestCase):
    model_specification_cls = DummyFluxModelSpecification
    num_data_files = 4
    num_frames = 4
    height = 64
    width = 64

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_files = []
        for i in range(self.num_data_files):
            data_file = pathlib.Path(self.tmpdir.name) / f"{i}.mp4"
            export_to_video(
                [Image.new("RGB", (self.width, self.height))] * self.num_frames, data_file.as_posix(), fps=2
            )
            self.data_files.append(data_file.as_posix())

        csv_filename = pathlib.Path(self.tmpdir.name) / "metadata.csv"
        with open(csv_filename.as_posix(), "w") as f:
            f.write("file_name,caption\n")
            for i in range(self.num_data_files):
                prompt = f"A cat ruling the world - {i}"
                f.write(f'{i}.mp4,"{prompt}"\n')

        dataset_config = {
            "datasets": [
                {
                    "data_root": self.tmpdir.name,
                    "dataset_type": "video",
                    "id_token": "TEST",
                    "video_resolution_buckets": [[self.num_frames, self.height, self.width]],
                    "reshape_mode": "bicubic",
                }
            ]
        }

        self.dataset_config_filename = pathlib.Path(self.tmpdir.name) / "dataset_config.json"
        with open(self.dataset_config_filename.as_posix(), "w") as f:
            json.dump(dataset_config, f)

    def tearDown(self):
        self.tmpdir.cleanup()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            time.sleep(3)

    def get_base_args(self) -> BaseArgs:
        args = BaseArgs()
        args.dataset_config = self.dataset_config_filename.as_posix()
        args.train_steps = 4
        args.max_data_samples = 8
        args.batch_size = 1
        args.gradient_checkpointing = True
        args.output_dir = self.tmpdir.name
        args.checkpointing_steps = 3
        args.enable_precomputation = False
        args.precomputation_items = self.num_data_files
        args.precomputation_dir = os.path.join(self.tmpdir.name, "precomputed")
        return args

    def get_lora_args(self) -> BaseArgs:
        args = self.get_base_args()
        args.parallel_backend = "ptd"
        args.training_type = TrainingType.LORA
        args.rank = 4
        args.lora_alpha = 4
        args.target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
        return args

    def _test_training(self, args: BaseArgs):
        model_specification = self.model_specification_cls()
        trainer = SFTTrainer(args, model_specification)
        trainer.run()

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
    def test_fsdp_lora_does_not_crash(self):
        """
        This is the critical test. It verifies that enabling LoRA training with FSDP
        (dp_shards > 1) completes without crashing due to the dtype mismatch.
        """
        args = self.get_lora_args()
        args.dp_degree = 1
        args.dp_shards = 2
        args.batch_size = 1
        args.compile_modules = []
        args.compile_scopes = []
        self._test_training(args)