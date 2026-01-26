import os
from pathlib import Path
import importlib
import importlib.util
import pkgutil

from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.rule import Rule

from spikee.utilities.modules import get_options_from_module, get_prefix_from_module

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


def _load_module(name, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_local(module_type: str):
    entries = []
    path = Path(os.getcwd()) / module_type
    if path.is_dir():
        for p in sorted(path.glob("*.py")):
            if p.name == "__init__.py":
                continue
            name = p.stem
            opts = None
            try:
                mod = _load_module(f"{module_type}.{name}", p)
                opts = get_options_from_module(mod, module_type)
                prefixes = get_prefix_from_module(mod, module_type)

                if prefixes and len(prefixes) == 2:
                    examples, prefixes = prefixes
                else:
                    examples = False
            except Exception:
                opts = ["<error>"]
                prefixes = ["<error>"]
                examples = False
            entries.append((name, opts, prefixes, examples))
    return entries


def _collect_builtin(pkg: str, module_type: str):
    entries = []
    try:
        pkg_mod = importlib.import_module(pkg)
        for _, name, is_pkg in pkgutil.iter_modules(pkg_mod.__path__):
            if name == "__init__" or is_pkg:
                continue
            opts = None
            try:
                mod = importlib.import_module(f"{pkg}.{name}")
                opts = get_options_from_module(mod, module_type)
                prefixes = get_prefix_from_module(mod, module_type)

                if prefixes and len(prefixes) == 2:
                    examples, prefixes = prefixes
                else:
                    examples = False
            except Exception:
                opts = ["<error>"]
                prefixes = ["<error>"]
                examples = False
            entries.append((name, opts, prefixes, examples))
    except ModuleNotFoundError:
        pass
    return entries


def _render_section(title: str, local_entries, builtin_entries):
    console.print(Rule(f"[bold]{title}[/bold]"))

    def print_section(entries, label) -> Tree:
        tree = Tree(f"[bold]{title} ({label})[/bold]")
        if entries:
            for name, opts, prefixes, examples in entries:
                node = tree.add(f"[bold]{name}[/bold]")
                if opts is not None:
                    opt_line = (
                        [f"[bold]{opts[0]} (default)[/bold]"] + opts[1:] if opts else []
                    )
                    if examples:  # Presume it's an example if there are prefixes
                        node.add("Example options: " + ", ".join(opt_line))
                    else:
                        node.add("Available options: " + ", ".join(opt_line))

                if prefixes is not None:
                    pref_line = prefixes if prefixes else []
                    node.add("Supported prefixes: " + ", ".join(pref_line))
        else:
            tree.add("(none)")

        return tree

    local_tree = print_section(local_entries, "local")
    console.print(local_tree)

    builtin_tree = print_section(builtin_entries, "built-in")
    console.print(builtin_tree)


# --- Commands ---


def list_judges(args):
    local = _collect_local("judges")
    builtin = _collect_builtin("spikee.judges", "judges")
    _render_section("Judges", local, builtin)


def list_targets(args):
    local = _collect_local("targets")
    builtin = _collect_builtin("spikee.targets", "targets")
    _render_section("Targets", local, builtin)


def list_plugins(args):
    local = _collect_local("plugins")
    builtin = _collect_builtin("spikee.plugins", "plugins")
    _render_section("Plugins", local, builtin)


def list_attacks(args):
    local = _collect_local("attacks")
    builtin = _collect_builtin("spikee.attacks", "attacks")
    _render_section("Attacks", local, builtin)
