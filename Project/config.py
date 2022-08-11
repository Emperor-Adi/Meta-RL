import yaml
from pathlib import Path
import argparse


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-f", "--config_file", \
    default="configs/default.yaml", help="Configuration File")
ARGS = parser.parse_args()


with open(ARGS.config_file,'r') as file:
    config = yaml.load(file)
print(f"Loaded configuration from {ARGS.config_file}")


def extern(func):
    """Wraps keyword args from config file"""
    def wrapper(*args, **kwargs):
        """Inject config keywords"""
        fname = Path(func.__globals__['__file__']).name
        conf = config[fname][func.__name__]
        conf.update(kwargs)
        return func(*args,**conf)
    return wrapper