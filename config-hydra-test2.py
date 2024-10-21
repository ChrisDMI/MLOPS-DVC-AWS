from hydra import initialize, compose
from omegaconf import OmegaConf

initialize(".")  # Assume the configuration file is in the current folder
cfg = compose(config_name="config-hydra-test.yaml")
print(OmegaConf.to_yaml(cfg))