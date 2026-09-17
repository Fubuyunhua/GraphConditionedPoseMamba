import numpy as np
import os, sys
import json
import pickle
import yaml
from easydict import EasyDict as edict
from typing import Any, IO

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

class TextLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "w") as f:
            f.write("")
    def log(self, log):
        with open(self.log_path, "a+") as f:
            f.write(log + "\n")

class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

def _deep_merge(base, override):
    """Recursively merge YAML mappings without mutating either input."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_mapping(config_path, stack=None):
    config_path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(config_path))))
    stack = [] if stack is None else list(stack)
    if config_path in stack:
        chain = " -> ".join(stack + [config_path])
        raise ValueError(f"Circular YAML config inheritance: {chain}")

    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a YAML mapping: {config_path}")

    base_paths = config.pop("_base_", config.pop("extends", None))
    if base_paths is None:
        return config
    if isinstance(base_paths, (str, os.PathLike)):
        base_paths = [base_paths]
    if not isinstance(base_paths, list):
        raise TypeError(f"_base_ must be a path or a list of paths: {config_path}")

    merged = {}
    for base_path in base_paths:
        base_path = os.path.expanduser(os.path.expandvars(str(base_path)))
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        merged = _deep_merge(
            merged,
            _load_config_mapping(base_path, stack=stack + [config_path]),
        )
    return _deep_merge(merged, config)


def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    config = edict(_load_config_mapping(config_path))
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content
