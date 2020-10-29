import os
import yaml
from easydict import EasyDict as edict

def YamlParser(config):
    """Read config YAML file and convert them into edict
    
    Parameters
    ----------

    config : YAML file containing configuration settings
    """

    assert(os.path.isfile(config))

    with open(config) as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)
        cfg = edict(cfg)

    return cfg