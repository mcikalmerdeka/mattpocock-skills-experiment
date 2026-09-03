# Running Doc QA with uv

**Verdict: keep `uv run doc-qa` as the one documented way to start the app.** It is the command uv's own docs demonstrate for project-provided commands, it runs through the console-script entry point this project declares in `[project.scripts]` (pyproject.toml:14-15), and it avoids two failure modes of hand-launching: bypassing the Streamlit bootstrap and re-resolving the app path by hand. (Verified against uv 0.7.3 on Windows 11.)

## What `uv run doc-qa` actually does

1. **Project discovery.** uv finds the project "by walking up the directory tree" from the current directory to a `pyproject.toml`, together with its `.venv` ([uv run reference](https://docs.astral.sh/uv/reference/cli/)). So the command works from the repo root or any subdirectory of it.
2. **Automatic lock and sync.** "When used in a project, the project environment will be created and updated before invoking the command" ([uv run reference](https://docs.astral.sh/uv/reference/cli/)). The [projects guide](https://docs.astral.sh/uv/guides/projects/) is explicit: "Prior to every `uv run` invocation, uv will verify that the lockfile is up-to-date with the `pyproject.toml`, and that the environment is up-to-date with the lockfile ... `uv run` guarantees that your command is run in an environment with all required dependencies at their locked versions."
3. **Entry-point resolution.** `doc-qa` is not a uv keyword; it is "provided by the project environment" ([Running commands](https://docs.astral.sh/uv/concepts/projects/run/)). Because the project is installed editable into `.venv` ([Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)), the `[project.scripts]` declaration produces a command shim: on Windows, `.venv\Scripts\doc-qa.exe` (present in this repo; confirmed locally). The [entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/) says installers "set up wrappers ... in the scripts directory of the install scheme" and that on Windows `console_scripts` "are wrapped in a console executable".
4. **The shim's behavior.** An entry point's "object reference points to a function which will be called with no arguments when this command is run" ([entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/)) — here `src.cli:main`, roughly `sys.exit(main())`.
5. **Streamlit bootstrap.** `src/cli.py:13-15` resolves the bundled app as `Path(__file__).with_name("app.py")` — relative to cli.py's location, not the current directory — rewrites `sys.argv` to `["streamlit", "run", <app path>]`, and calls `streamlit.web.cli.main`. That is exactly the documented Streamlit launch path: "the easiest way to run it is with `streamlit run` ... a local Streamlit server will spin up" ([Run your app](https://docs.streamlit.io/develop/concepts/architecture/run-your-app)).

## What the alternatives do differently

- **`uv run python src/app.py`** — uv still sets up the same project environment, but then plain Python executes `app.py` as a standalone script. No `main()`, no `streamlit run`, no server: Streamlit's docs document only `streamlit run` (or the equivalent `python -m streamlit run`) as launch paths ([Run your app](https://docs.streamlit.io/develop/concepts/architecture/run-your-app)). That plain `python app.py` does not start the server is confirmed first-party by Streamlit's own tracker, where the feature request exists precisely because "It would be convenient to allow users to simply run `python streamlit_app.py`, without having to necessarily run `streamlit run streamlit_app.py`" — addressed in Streamlit 1.59 only via a new `st.App(...).run()` launcher API ([streamlit#9450](https://github.com/streamlit/streamlit/issues/9450)), which this repo does not use. It also bypasses the bundled-path resolution in cli.py.
- **`uv run python -m src.cli`** — functionally equivalent here (src is a package with `__init__.py`, and the `if __name__ == "__main__"` guard in cli.py fires), because `-m` runs the module's code as `__main__`. It just hand-rolls what the shim does, and its long form buys nothing.
- **`uv run src/app.py`** (bare `.py` path) — this *is* a thing, but note what it means: "`uv run file.py` is equivalent to `uv run python file.py`" ([uv run reference](https://docs.astral.sh/uv/reference/cli/)) — so same no-server outcome as above. Separately, if a file carried PEP 723 inline metadata, it would run "into an isolated, ephemeral environment", and "the project's dependencies will be ignored" even inside a project ([Running scripts](https://docs.astral.sh/uv/guides/scripts/)); `uv run --script` forces that script mode.

## What uv officially recommends

uv never forbids `uv run python ...` — both forms run in the same project environment. But the docs' stated default for anything the project provides is the entry point:

> "This environment is isolated from the current shell by default ... Instead, use `uv run` to run commands in the project environment." — [Running commands](https://docs.astral.sh/uv/concepts/projects/run/)

> "To run a command in the project environment, use `uv run`." — [Project structure and files](https://docs.astral.sh/uv/concepts/projects/layout/)

And the canonical pattern for `[project.scripts]` is to invoke the command directly: uv's project template defines `hello-world = "hello_world:main"` and says "Try it out with `uv run`: `uv run hello-world`" ([Working on projects](https://docs.astral.sh/uv/guides/projects/)). `python -m` / `python <file>` remain appropriate for ad-hoc debugging (running a test file, a REPL, profiling a single module).

## Windows specifics

- Scripts live in `.venv\Scripts\` (not `bin/`); uv's own Windows docs refer to files "installed by setuptools in `.venv\Scripts`" ([Running commands](https://docs.astral.sh/uv/concepts/projects/run/)), and the activation path is `.venv\Scripts\activate` ([Working on projects](https://docs.astral.sh/uv/guides/projects/)).
- The `.exe` in `doc-qa.exe` is the wrapper executable described by the [entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/); without `uv run`, "the virtual environment must be active to run scripts and commands in the project" ([Working on projects](https://docs.astral.sh/uv/guides/projects/)).

## Why `uv run doc-qa` is right for this repo, specifically

- The Streamlit target is resolved from `__file__` (src/cli.py:13), so the app path is correct no matter the cwd — no manual `streamlit run src/app.py` path juggling.
- Streamlit apps must go through `streamlit run`; cli.py does exactly that programmatically (`streamlit.web.cli.main`), which is the supported module form (`python -m streamlit run` is documented as "equivalent" in [Run your app](https://docs.streamlit.io/develop/concepts/architecture/run-your-app)).
- One caveat that applies to *every* launch method: settings come from `.env` discovered via `find_dotenv(usecwd=True)` (src/config.py:57) and the default data directory is the relative `"data"` (src/config.py:23). Start the app from the project root so `.env` and `data/` land where you expect.
- Operational footnote: `uv run` syncs by default; to skip that check use `uv run --no-sync`, and `uv sync` remains the explicit way to refresh the environment ([Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)).

## Comparison

| Command | What it does | When to use |
| --- | --- | --- |
| `uv run doc-qa` | Syncs env, then runs the `.venv\Scripts\doc-qa.exe` shim → `src.cli:main` → `streamlit run` on the bundled `app.py` | The normal way to start the app (what README documents) |
| `uv run python -m src.cli` | Syncs env, then runs `src.cli` as `__main__`; same effect as the shim | Debugging the entry-point code itself |
| `uv run python src/app.py` / `uv run src/app.py` | Syncs env, then plain-Python executes the app script; no Streamlit server | Not for starting the app; single-file experiments |
| `uv run --no-sync doc-qa` | Same as above but skips the environment check | When you know the env is current and want the fastest start |

## Rules of thumb

- If the project defines a command in `[project.scripts]`, run that command with `uv run <name>` — that is the documented pattern.
- Reach for `uv run python ...` when you need the interpreter, not the product: REPLs, one-off modules, debugging.
- `uv run <file>.py` is script mode (`== uv run python <file>.py`); PEP 723 metadata silently moves the script into an isolated environment.
- A Streamlit app is only truly started by `streamlit run` (or an equivalent wrapper) — never by executing the script directly.
- Sync is automatic under `uv run`; use `--no-sync`/`uv sync` only when you have a reason.

## Sources

- [uv: `uv run` command reference](https://docs.astral.sh/uv/reference/cli/)
- [uv: Running commands in projects](https://docs.astral.sh/uv/concepts/projects/run/)
- [uv: Project structure and files](https://docs.astral.sh/uv/concepts/projects/layout/)
- [uv: Working on projects (guide)](https://docs.astral.sh/uv/guides/projects/)
- [uv: Running scripts (guide)](https://docs.astral.sh/uv/guides/scripts/)
- [uv: Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
- [Python Packaging User Guide: Entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/)
- [Streamlit docs: Run your app](https://docs.streamlit.io/develop/concepts/architecture/run-your-app)
- [Streamlit docs: `$ streamlit run` CLI reference](https://docs.streamlit.io/develop/api-reference/cli/run)
- [streamlit/streamlit issue #9450 (first-party, maintainer thread)](https://github.com/streamlit/streamlit/issues/9450)
- Local files: `pyproject.toml`, `src/cli.py`, `src/config.py`, `src/app.py`, `README.md`, `.venv\Scripts\doc-qa.exe` (verified, uv 0.7.3)
