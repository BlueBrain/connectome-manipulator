from pathlib import Path
import json

from click.testing import CliRunner
from connectome_manipulator.cli import app
from libsonata import EdgeStorage

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
WIRING_CONFIG_TPL = DATA_DIR / "wiring_config.json.tpl"


def gen_config(output_path):
    wiring_config = {}
    with open(WIRING_CONFIG_TPL, "r") as config_file:
        wiring_config = json.load(config_file)
    wiring_config["circuit_path"] = str(DATA_DIR)
    wiring_config["output_path"] = str(output_path)
    wiring_config_path = DATA_DIR / "wiring_config.json"
    with open(wiring_config_path, "w") as outf:
        json.dump(wiring_config, outf, indent=2)
    return wiring_config_path


def test_build_local_connectome(tmp_path):
    wiring_config_path = gen_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-local-connectome",
            str(wiring_config_path),
            "--output-dir",
            str(tmp_path),
        ],
        catch_exceptions=False,
    )
    print(result.output)
    assert result.exit_code == 0, f"Application exited with non-zero code.\n{result.output}"

    edge_storage = EdgeStorage(
        tmp_path / "sonata/networks/edges/functional/All/edges_ConnWiring_DD.h5"
    )
    edges = edge_storage.open_population("default")
    assert len(edges) == 167, "Wrong number of connections found in output"
