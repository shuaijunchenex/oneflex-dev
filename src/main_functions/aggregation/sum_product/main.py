from __future__ import annotations

# Init startup path, change current path to startup python file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

# import
from usyd_learning.ml_utils import console
from product_sum.entry import Entry

g_app = Entry()

def main():
    # Load app config set from yaml file
    g_app.load_app_config("./sum_product/config.yaml")

    # Get training rounds
    general_yaml = g_app.get_app_object("general")
    training_rounds = general_yaml["general"]["training_rounds"]

    # Run app
    g_app.run(training_rounds)
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

    console.out("Training of aggregation strategy - Sum Product")
    console.out("======================= PROGRAM BEGIN ==========================")
    main()
    console.out("\n======================= PROGRAM END ============================")
    console.wait_any_key()
