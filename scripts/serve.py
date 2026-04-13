"""Start the FastAPI inference service."""

from __future__ import annotations

import argparse

import uvicorn

from cfd_operator.api.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve CFD operator inference API.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    app = create_app(checkpoint_path=args.checkpoint, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
