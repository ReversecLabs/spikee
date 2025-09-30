import os
from pathlib import Path
import importlib
import importlib.util
import pkgutil
import ast

from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.rule import Rule

console = Console()


def list_seeds(args):
    base = Path(os.getcwd(), "datasets")
    if not base.is_dir():
        console.print(
            Panel("No 'datasets/' folder found", title="[seeds]", style="red")
        )
        return

    want = {
        "base_user_inputs.jsonl",
        "base_documents.jsonl",
        "standalone_user_inputs.jsonl",
        "standalone_attacks.jsonl",
    }

    seeds = sorted(
        {
            d.name
            for d in base.iterdir()
            if d.is_dir() and any((d / fn).is_file() for fn in want)
        }
    )

    console.print(
        Panel(
            "\n".join(seeds) if seeds else "(none)", title="[seeds] Local", style="cyan"
        )
    )


def list_datasets(args):
    base = Path(os.getcwd(), "datasets")
    if not base.is_dir():
        console.print(
            Panel("No 'datasets/' folder found", title="[datasets]", style="red")
        )
        return
    files = [f.name for f in base.glob("*.jsonl")]
    panel = Panel(
        "\n".join(files) if files else "(none)", title="[datasets] Local", style="cyan"
    )
    console.print(panel)


# --- Helpers ---


def _has_options(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text())
        return any(
            isinstance(n, ast.FunctionDef) and n.name == "get_available_option_values"
            for n in tree.body
        )
    except Exception:
        return False


def _load_module(name, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_local(local_dir: str):
    entries = []
    path = Path(os.getcwd()) / local_dir
    if path.is_dir():
        for p in sorted(path.glob("*.py")):
            if p.name == "__init__.py":
                continue
            name = p.stem
            opts = None
            if _has_options(p):
                try:
                    mod = _load_module(f"{local_dir}.{name}", p)
                    opts = mod.get_available_option_values()
                except Exception:
                    opts = ["<error>"]
            entries.append((name, opts))
    return entries


def _collect_builtin(pkg: str):
    entries = []
    try:
        pkg_mod = importlib.import_module(pkg)
        for _, name, is_pkg in pkgutil.iter_modules(pkg_mod.__path__):
            if name == "__init__" or is_pkg:
                continue
            opts = None
            try:
                spec = importlib.util.find_spec(f"{pkg}.{name}")
                if spec and spec.origin and _has_options(Path(spec.origin)):
                    mod = importlib.import_module(f"{pkg}.{name}")
                    opts = mod.get_available_option_values()
            except Exception:
                opts = ["<error>"]
            entries.append((name, opts))
    except ModuleNotFoundError:
        pass
    return entries


def _render_section(title: str, local_entries, builtin_entries):
    console.print(Rule(f"[bold]{title}[/bold]"))
    # local
    tree = Tree(f"[bold]{title} (local)[/bold]")
    if local_entries:
        for name, opts in local_entries:
            node = tree.add(f"[bold]{name}[/bold]")
            if opts is not None:
                opt_line = (
                    [f"[bold]{opts[0]} (default)[/bold]"] + opts[1:] if opts else []
                )
                node.add("Available options: " + ", ".join(opt_line))
    else:
        tree.add("(none)")
    console.print(tree)

    # built-in
    tree2 = Tree(f"[bold]{title} (built-in)[/bold]")
    if builtin_entries:
        for name, opts in builtin_entries:
            node = tree2.add(f"[bold]{name}[/bold]")
            if opts is not None:
                opt_line = (
                    [f"[bold]{opts[0]} (default)[/bold]"] + opts[1:] if opts else []
                )
                node.add("Available options: " + ", ".join(opt_line))
    else:
        tree2.add("(none)")
    console.print(tree2)


# --- Commands ---


def list_judges(args):
    local = _collect_local("judges")
    builtin = _collect_builtin("spikee.judges")
    _render_section("Judges", local, builtin)


def list_targets(args):
    local = _collect_local("targets")
    builtin = _collect_builtin("spikee.targets")
    _render_section("Targets", local, builtin)


def list_plugins(args):
    local = _collect_local("plugins")
    builtin = _collect_builtin("spikee.plugins")
    _render_section("Plugins", local, builtin)


def list_attacks(args):
    local = _collect_local("attacks")
    builtin = _collect_builtin("spikee.attacks")
    _render_section("Attacks", local, builtin)
