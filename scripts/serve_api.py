"""Local Serving CLI for Indonesian Insurance Claim & FNOL Triage API (Phase 3.1).

Usage:
    python scripts/serve_api.py [--host 127.0.0.1] [--port 8000] [--reload]
"""

import argparse
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start FastAPI Triage Inference Server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable live auto-reload")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("  STARTING INDONESIAN INSURANCE CLAIM & FNOL TRIAGE API SERVICE", flush=True)
    print("=" * 70, flush=True)
    print(f"Listening on: http://{args.host}:{args.port}", flush=True)
    print(f"Swagger Docs: http://{args.host}:{args.port}/docs", flush=True)
    print(f"Health Check: http://{args.host}:{args.port}/health", flush=True)
    print(f"Categories:   http://{args.host}:{args.port}/v1/categories", flush=True)
    print("=" * 70, flush=True)

    uvicorn.run("src.serving.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
