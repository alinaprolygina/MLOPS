import argparse

__version__ = "0.1.0"
all = ["__version__"]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Backend sever for MopsApp")
    parser.add_argument(
        "--api-type", type=str, choices=["fastapi", "grpc"], default="grpc", help="API type of the server"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for the server")
    parser.add_argument("--port", type=int, default=9090, help="Port number for the server")
    parser.add_argument(
        "--local", default=False, action="store_true", help="Flag of running the application locally (not in docker)."
    )
    parser.add_argument(
        '--clearml', action=argparse.BooleanOptionalAction, help="Flag of using clearml to track models."
        )
    args = parser.parse_args()
    return args
