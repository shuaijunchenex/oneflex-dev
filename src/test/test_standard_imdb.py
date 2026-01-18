import os
import sys

# Ensure the src directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from usyd_learning.ml_utils import console, ConfigLoader
from test.fl_lora_sample.standard_sample_entry import StandardSampleEntry

def main():
    # Use the 2-clients config
    config_path = os.path.join(current_dir, "fl_lora_sample/imdb_2clients.yaml")
    config = ConfigLoader.load(config_path)
    
    # Load the referenced files
    runner_yaml_path = os.path.join(current_dir, "fl_lora_sample", config["app"]["runner"])
    runner_yaml = ConfigLoader.load(runner_yaml_path)
    
    # Initialize the app
    app = StandardSampleEntry()
    
    # Manually populate the app objects as the lora_sample_entry.py expects them
    app.set_app_object("runner", runner_yaml)
    app.set_app_object("client_yaml", config)
    app.set_app_object("server_yaml", config)
    app.set_app_object("edge_yaml", {})
    
    # Run the experiment
    console.info("Starting FL experiment with 2 clients and Tiny Scratch Transformer on IMDB...")
    app.run(training_rounds=2)
    console.ok("Experiment completed!")

if __name__ == "__main__":
    main()
