"""The one runner of every training and evaluation run.

    python experiments/run.py train <config>      calls check2.training.loop.train(config), or, for a config
                                                  with signal_forming: formed_model_only,
                                                  check2.training.forming.training.train(config)
    python experiments/run.py evaluate <config> [--epoch <k>]
                                                  calls check2.evaluation.runs.execute(config, checkpoint_epoch=k)

For training, the config is loaded for a run by `check2.config.schema.load_config`, so its
pinned model revision and selected read position are resolved before the run starts.
Evaluation receives the config file's path, because a run is named after its config file,
and loads it the same way. A command whose
function does not exist yet says so and exits non-zero.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMMANDS = {
    "train": ("check2.training.loop", "train"),
    "evaluate": ("check2.evaluation.runs", "execute"),
}
# The forming update trains with a copy of the reader, by its own training, not through the loop.
FORMING_TRAINING = ("check2.training.forming.training", "train")
NOT_BUILT = 2


def _function(command: str):
    module_name, name = COMMANDS[command]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        if err.name is not None and module_name.startswith(err.name):
            return None, f"{module_name} does not exist yet, so '{command}' cannot run"
        raise
    function = getattr(module, name, None)
    if function is None:
        return None, f"{module_name} has no function {name} yet, so '{command}' cannot run"
    return function, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one training or evaluation from its config.")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("config", type=Path)
    parser.add_argument("--epoch", type=int, default=None,
                        help="evaluate only: the epoch whose checkpoint a run over heldout loads")
    args = parser.parse_args(argv)
    if args.epoch is not None and args.command != "evaluate":
        parser.error("--epoch belongs to the evaluate command")
    function, problem = _function(args.command)
    if function is None:
        print(problem, file=sys.stderr)
        return NOT_BUILT
    if args.command == "evaluate":
        # The run is named after its config file, so evaluation receives the file and loads it itself.
        function(args.config, checkpoint_epoch=args.epoch)
        return 0
    from check2.config.schema import load_config

    config = load_config(args.config)
    if config.signal_forming == "formed_model_only":
        function = getattr(importlib.import_module(FORMING_TRAINING[0]), FORMING_TRAINING[1])
    # A training returns the folder of its checkpoints; a failure raises, so reaching here is success.
    function(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
