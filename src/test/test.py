from __future__ import annotations

# Init startup path, change current path to startup python file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

# import
from usyd_learning.ml_utils import console, DeviceChecker
from fl_lora_sample.standard_sample_entry import StandardSampleEntry

g_app = StandardSampleEntry()

def main():
    # Load app config set from yaml file
    g_app.load_app_config("./fl_lora_sample/agnews_test.yaml")

    # Get training rounds
    training_rounds = g_app.training_rounds

    # Run app
    g_app.run(DeviceChecker.get_device(), training_rounds)
    return

if __name__ == "__main__":
    #Initial console options
    console.set_log_level("all")  # Log level: error > warn > ok > info > out > all
    console.set_debug(True)  # True for display debug info

    # Set log path and name if needed
    console.set_console_logger(log_path="./log/", log_name = "console_trace")
    console.set_exception_logger(log_path="./log/", log_name = "exception_trace")
    console.set_debug_logger(log_path="./log/", log_name = "debug_trace")

    console.enable_console_log(True)  # True for log console info to file by log level
    console.enable_exception_log(True)  # True for log exception info to file
    console.enable_debug_log(True)  # True for log debug info to file

    console.out("Simple FL program")
    console.out("======================= PROGRAM BEGIN ==========================")
    main()
    console.out("\n======================= PROGRAM END ============================")
    #console.wait_any_key()
