from __future__ import annotations

import os
from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from usyd_learning.ml_utils import console
from fl_lora_sample.sfl_sample_entry import SampleAppEntry
from usyd_learning.ml_utils.model_utils import ModelUtils
from usyd_learning.ml_utils.training_utils import TrainingUtils


g_app = SampleAppEntry()


def main():
    g_app.load_app_config("./test_configurations/sfl_r001_epoch1.yaml")
    device = ModelUtils.accelerator_device()
    training_rounds = g_app.training_rounds
    g_app.run(device, training_rounds)


if __name__ == "__main__":
    TrainingUtils.set_seed_all(42)
    console.set_log_level("all")
    console.set_debug(True)
    console.set_console_logger(log_path="./log/", log_name="console_trace")
    console.set_exception_logger(log_path="./log/", log_name="exception_trace")
    console.set_debug_logger(log_path="./log/", log_name="debug_trace")
    console.enable_console_log(True)
    console.enable_exception_log(True)
    console.enable_debug_log(True)

    console.out("Simple SFL program")
    console.out("======================= PROGRAM BEGIN ==========================")
    main()
    console.out("\n======================= PROGRAM END ============================")
    console.wait_any_key()
