import os
from typing import Any, Dict, Tuple

import torch

import model_hub.utils


def get_config_pretrained_url_mapping() -> Dict[str, str]:
    """
    Walks the MMDETECTION_CONFIG_DIR and creates a mapping of configs
    to urls for pretrained checkpoints.
    """
    models = {}
    config_dir = os.getenv("MMDETECTION_CONFIG_DIR")
    if config_dir:
        for root, _, files in os.walk(config_dir):
            for f in files:
                if "README" in f:
                    with open(os.path.join(root, f), "r") as readme:
                        lines = readme.readlines()
                        for line in lines:
                            if "[config]" in line:
                                start = line.find("[config]")
                                end = line.find(".py", start)
                                start = line.rfind("/", start, end)
                                config_name = line[start + 1 : end + 3]
                                start = line.find("[model]")
                                end = line.find(".pth", start)
                                ckpt_name = line[start + 8 : end + 4]
                                models[config_name] = ckpt_name
    return models


CONFIG_TO_PRETRAINED = get_config_pretrained_url_mapping()


def get_pretrained_ckpt_path(download_directory: str, config_file: str) -> Tuple[Any, Any]:
    """
    If the config_file has an associated pretrained checkpoint,
    return path to downloaded checkpoint and preloaded checkpoint

    Arguments:
        download_directory: path to download checkpoints to
        config_file: mmcv config file path for which to find and load pretrained weights
    Returns:
        checkpoint path, loaded checkpoint
    """
    config_file = config_file.split("/")[-1]
    if config_file in CONFIG_TO_PRETRAINED:
        ckpt_path = model_hub.utils.download_url(
            download_directory, CONFIG_TO_PRETRAINED[config_file]
        )
        return ckpt_path, torch.load(ckpt_path)  # type: ignore
    return None, None
