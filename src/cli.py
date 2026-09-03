"""Console entry point that boots the Doc QA Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Run the Doc QA web app (equivalent to ``streamlit run`` on the bundled app)."""
    from streamlit.web import cli as stcli

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
