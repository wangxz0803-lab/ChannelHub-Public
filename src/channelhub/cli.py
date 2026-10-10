"""Command-line and isolated worker entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .config import RunConfig
from .simulation import simulate, save_result


def main(argv=None):
    parser = argparse.ArgumentParser(description="ChannelHub university research simulator")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the local web platform")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5088)
    serve.add_argument("--runtime", type=Path, default=Path("runtime"))
    generate = commands.add_parser("generate", help="Generate an observation package")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--x", type=Path, help="Complex .npy custom transmit grid")
    generate.add_argument("--scene", type=Path, help="Sionna XML scene (CLI only)")
    commands.add_parser("example", help="Print a complete example configuration")
    commands.add_parser("schema", help="Print the public configuration JSON schema")
    args = parser.parse_args(argv)
    if args.command == "example":
        print(RunConfig().model_dump_json(indent=2))
    elif args.command == "schema":
        print(json.dumps(RunConfig.model_json_schema(), indent=2))
    elif args.command == "serve":
        import uvicorn
        from .server import create_app

        uvicorn.run(create_app(args.runtime), host=args.host, port=args.port)
    else:
        try:
            cfg = RunConfig.model_validate_json(args.config.read_text(encoding="utf-8-sig"))
            if args.x and cfg.tx.waveform.kind != "custom":
                raise ValueError("--x requires tx.waveform.kind=custom")
            if args.scene and cfg.engine != "sionna_rt":
                raise ValueError("--scene requires engine=sionna_rt")
            custom = np.load(args.x, allow_pickle=False) if args.x else None

            def progress(percent, message):
                print(json.dumps({"percent": percent, "message": message}), flush=True)

            arrays, metadata = simulate(cfg, custom_x=custom, scene_path=args.scene, progress=progress)
            save_result(args.output, arrays, metadata)
            progress(100, "Completed")
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
