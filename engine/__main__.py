"""CLI entrypoint: `python -m engine --prompt "..."`.

Lives here (not in server.py) so the CLI is never imported as a side effect of
`import engine`, which would otherwise trigger a runpy double-import warning.
"""
from .server import main

if __name__ == "__main__":
    main()
