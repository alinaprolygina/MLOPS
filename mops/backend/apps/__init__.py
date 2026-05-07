import argparse
from typing import Literal


def parse_arguments(api_type: Literal["FastAPI", "gRPC"]):
    parser = argparse.ArgumentParser(description=f"Start a {api_type} server for MopsApp")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for the server")
    parser.add_argument("--port", type=int, default=9090, help="Port number for the server")
    args = parser.parse_args()
    return args
