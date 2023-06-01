"""Utils."""
import json
import os
from pathlib import Path
from typing import Any, Dict


def create_dir(path: os.PathLike) -> Path:
    """Create directory and parents if it doesn't already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(filepath: os.PathLike) -> Dict[Any, Any]:
    """Load from JSON file."""
    return json.loads(Path(filepath).read_bytes())


def write_json(filepath: os.PathLike, data: Dict[Any, Any], indent: int = 2) -> None:
    """Write json file."""
    with open(filepath, "w", encoding="utf-8") as fd:
        json.dump(data, fd, indent=indent)


def reduce_config_paths(config: dict, config_dir: os.PathLike) -> dict:
    """Reduces absolute paths with respect to base_dir.

    Args:
        config (dict): Input config with absolute paths and no manifest.
        config_dir (str|Path): The path to the directory of the config.

    Returns:
        A new config with the paths reduced to being relative to the config dir.

    Note: Paths that are not in config_dir's tree will stay absolute.
    """
    if not Path(config_dir).is_absolute():
        raise ValueError(f"Circuit config's directory is not absolute: {config_dir}")

    if "manifest" in config:
        raise ValueError(
            f"A reduced config with absolute paths and no manifest must be provided.\n"
            f"Got instead: {config}"
        )

    reduced_config = {
        "version": config.get("version", "1"),
        "manifest": {"$BASE_DIR": "."},
    }

    if "node_sets_file" in config:
        reduced_config["node_sets_file"] = _reduce_path(config["node_sets_file"], config_dir)

    reducer = {
        "nodes_file": _reduce_path,
        "node_types_file": _reduce_path,
        "edges_file": _reduce_path,
        "edge_types_file": _reduce_path,
        "populations": _reduce_populations,
    }

    reduced_config["networks"] = {
        network_type: [
            {
                key: (reducer[key](value, config_dir) if key in reducer else value)
                for key, value in net_dict.items()
            }
            for net_dict in network_list
        ]
        for network_type, network_list in config["networks"].items()
    }
    return reduced_config


def _reduce_path(path: str, base_dir: os.PathLike) -> str:
    if not path or path.startswith("$"):
        return path

    path = Path(path)

    if path.is_absolute():
        if path.is_relative_to(base_dir):
            return str(Path("$BASE_DIR", path.relative_to(base_dir)))

    return str(Path("$BASE_DIR", path))


def _reduce_populations(populations_dict: dict, base_dir: os.PathLike) -> dict:
    def reduce_entry(key, value):
        if key.endswith(("file", "dir")):
            return _reduce_path(value, base_dir)

        if key == "alternate_morphologies":
            return {
                alt_key: _reduce_path(alt_path, base_dir) for alt_key, alt_path in value.items()
            }

        return value

    return {
        pop_name: {key: reduce_entry(key, value) for key, value in pop_dict.items()}
        for pop_name, pop_dict in populations_dict.items()
    }


class ConsoleColors:
    """Helper class for formatting console text."""

    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, _, DEFAULT = range(30, 40)
    NORMAL, BOLD, DIM, UNDERLINED, BLINK, INVERTED, HIDDEN = [a << 8 for a in range(7)]

    # These are the sequences needed to control output
    _CHANGE_SEQ = "\033[{}m"
    _RESET_SEQ = "\033[0m"

    @classmethod
    def reset(cls):
        """Reset colors."""
        return cls._RESET_SEQ

    @classmethod
    def set_text_color(cls, color):
        """Change text color."""
        return cls._CHANGE_SEQ.format(color)

    @classmethod
    def format_text(cls, text, color, style=None):
        """Format the text."""
        style = (style or color) >> 8
        format_seq = str(color & 0x00FF) + ((";" + str(style)) if style else "")
        return cls._CHANGE_SEQ.format(format_seq) + text + cls._RESET_SEQ
