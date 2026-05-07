from mops import parse_arguments
from mops.backend import setup
from mops.backend.apps.grpc.server import serve as grpc_serve
from mops.backend.apps.fastapi.app import serve as fastapi_serve


def main():
    args = parse_arguments()

    setup(local=args.local, clearml=args.clearml)

    if args.api_type == "grpc":
        grpc_serve(args.host, args.port)
    elif args.api_type == "fastapi":
        fastapi_serve(args.host, args.port)
    else:
        err_msg = f"Unsupported api_type: {args.api_type}. Supported types: 'grpc', 'fastapi'"
        raise ValueError(err_msg)


if __name__ == "__main__":
    main()
