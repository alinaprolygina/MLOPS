from streamlit.web import cli

from mops import parse_arguments
from mops.frontend import setup
from mops.frontend.managers.client import Client


def main():
    args = parse_arguments()
    setup()
    Client.api_type = args.api_type
    Client.host = args.host
    Client.port = args.port
    Client.connect()

    cli.main_run(["mops/frontend/dashboard.py"])


if __name__ == "__main__":
    main()
