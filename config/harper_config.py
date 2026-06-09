# encoding: utf-8

from config.config_utils import load_harper_config


config = load_harper_config()
cfg = config


if __name__ == "__main__":
    print(config)
