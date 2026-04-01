from __future__ import annotations

import argparse
import threading
import webbrowser

from convolution_visualizer.app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the PyTorch convolution visualizer."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6009,
        help="Port to serve the application on. Defaults to 6009.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app()
    url = f"http://127.0.0.1:{args.port}"

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
