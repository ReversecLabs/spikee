import os
from pathlib import Path
import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass

from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.rule import Rule

from spikee.utilities.modules import get_options_from_module, get_prefix_from_module, get_description_from_module

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

@dataclass
class Module:
    name: str
    options: list
    prefixes: list
    examples: bool = False
    classification: str = None
    description: str = ""


def _load_module(name, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_local(module_type: str):
    entries = {None: []}
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
                classification = get_description_from_module(mod, module_type)

                # Load Examples/Available Options Flags
                if prefixes and len(prefixes) == 2:
                    examples, prefixes = prefixes
                else:
                    examples = False

                # Get classification
                if classification is not None and len(classification) == 2:
                    classification, description = classification
                else:
                    classification, description = None, ""

            except Exception:
                opts = ["<error>"]
                prefixes = ["<error>"]
                examples = False
                classification = None
                description = ""

            if classification not in entries:
                entries[classification] = []

            entries[classification].append(
                Module(name, opts, prefixes, examples, classification, description)
            )
    return entries


def _collect_builtin(pkg: str, module_type: str):
    entries = {None: []}
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
                classification = get_description_from_module(mod, module_type)

                # Load Examples/Available Options Flags
                if prefixes and len(prefixes) == 2:
                    examples, prefixes = prefixes
                else:
                    examples = False

                # Get classification
                if classification is not None and len(classification) == 2:
                    classification, description = classification
                else:
                    classification, description = None, ""

            except Exception:
                opts = ["<error>"]
                prefixes = ["<error>"]
                examples = False
                classification = None
                description = ""

            if classification not in entries:
                entries[classification] = []

            entries[classification].append(
                Module(name, opts, prefixes, examples, classification, description)
            )
    except ModuleNotFoundError:
        pass
    return entries


def _render_section(title: str, local_entries, builtin_entries, description: bool = False):
    console.print(Rule(f"[bold]{title}[/bold]"))

    def print_section(entries, label) -> Tree:
        tree = Tree(f"[bold]{title} ({label})[/bold]")
        if entries:
            for classification in entries.keys():

                if classification is not None:
                    node = tree.add(f"[bold red]{classification.value}[/bold red]")
                else:
                    node = tree

                for module in entries[classification]:
                    if not description or module.description is None:
                        module_node = node.add(f"[bold cyan]{module.name}[/bold cyan]")
                    else:
                        module_node = node.add(f"[bold cyan]{module.name}[/bold cyan]: {module.description}")

                    if module.options is not None:
                        opt_line = (
                            [f"[bold]{module.options[0]} (default)[/bold]"] + module.options[1:] if module.options else []
                        )
                        if module.examples:  # Presume it's an example if there are prefixes
                            module_node.add("[bright_black]Example options: " + ", ".join(opt_line) + "[/bright_black]")
                        else:
                            module_node.add("[bright_black]Available options: " + ", ".join(opt_line) + "[/bright_black]")

                    if module.prefixes is not None:
                        pref_line = module.prefixes if module.prefixes else []
                        module_node.add("[bright_black]Supported prefixes: " + ", ".join(pref_line) + "[/bright_black]")
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
    _render_section("Judges", local, builtin, args.description)


def list_targets(args):
    local = _collect_local("targets")
    builtin = _collect_builtin("spikee.targets", "targets")
    _render_section("Targets", local, builtin)


def list_plugins(args):
    local = _collect_local("plugins")
    builtin = _collect_builtin("spikee.plugins", "plugins")
    _render_section("Plugins", local, builtin, args.description)


def list_attacks(args):
    local = _collect_local("attacks")
    builtin = _collect_builtin("spikee.attacks", "attacks")
    _render_section("Attacks", local, builtin, args.description)
