import hydra
import os
from hydra import compose

hydra.initialize_config_dir(f"{os.path.abspath(__file__)}/../conf")
api_config = compose(config_name="config")
