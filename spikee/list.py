import os
from pathlib import Path
import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass
from typing import List
import traceback

from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.rule import Rule

from spikee.utilities.enums import ModuleTag, module_tag_to_colour
from spikee.utilities.modules import get_options_from_module, get_description_from_module
from spikee.utilities.llm import get_example_llm_models, get_supported_llm_models, get_supported_prefixes

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
    util_llm: bool = False
    tags: List[ModuleTag] = None
    description: str = ""


def _load_module(name, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_local(module_type: str):
    entries = []
    any_util_llm = False

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
                description = get_description_from_module(mod, module_type)

                if opts is not None and isinstance(opts, tuple) and len(opts) == 2:
                    util_llm = opts[1]
                    opts = opts[0]
                else:
                    util_llm = False

                # Get classification
                if description is not None and len(description) == 2:
                    tags, description = description
                else:
                    tags, description = [], ""

            except Exception:
                opts = ["<error>"]
                util_llm = False
                tags = []
                description = ""

            entries.append(
                Module(name, opts, util_llm, tags, description)
            )
            if util_llm:
                any_util_llm = True

    return entries, any_util_llm


def _collect_builtin(pkg: str, module_type: str):
    entries = []
    any_util_llm = False

    try:
        pkg_mod = importlib.import_module(pkg)
        for _, name, is_pkg in pkgutil.iter_modules(pkg_mod.__path__):
            if name == "__init__" or is_pkg:
                continue
            opts = None
            try:
                mod = importlib.import_module(f"{pkg}.{name}")
                opts = get_options_from_module(mod, module_type)
                description = get_description_from_module(mod, module_type)

                if opts is not None and isinstance(opts, tuple) and len(opts) == 2:
                    util_llm = opts[1]
                    opts = opts[0]
                else:
                    util_llm = False

                # Get classification
                if description is not None and len(description) == 2:
                    tags, description = description
                else:
                    tags, description = [], ""

            except Exception:
                opts = ["<error>"]
                util_llm = False
                tags = []
                description = ""

                traceback.print_exc()

            entries.append(
                Module(name, opts, util_llm, tags, description)
            )
            if util_llm:
                any_util_llm = True
    except ModuleNotFoundError:
        pass
    return entries, any_util_llm


def _render_section(title: str, local_entries, builtin_entries, util_llm: bool = False, description: bool = False):
    console.print(Rule(f"[bold]{title}[/bold]"))

    if util_llm:
        console.print(Panel(
            f"""[yellow]Note:[/yellow] Modules with a [yellow][LLM][/yellow] tag, use the built-in LLM service.
The LLM options are available, using 'model=<option>':
Example Models: {", ".join(get_example_llm_models())}
Supported Prefixes: {", ".join(get_supported_prefixes())}
""", style="yellow"
        ))

    def print_section(entries, label) -> Tree:
        tree = Tree(f"[bold]{title} ({label})[/bold]")
        if entries:
            for module in entries:

                node_line = f"[bold cyan]{module.name}[/bold cyan]"

                if module.tags:
                    module.tags = sorted(module.tags, key=lambda x: x.value)

                    tags = []
                    for tag in module.tags:
                        tags.append(f"[{module_tag_to_colour(tag)}][{tag.value}][/{module_tag_to_colour(tag)}]")

                    node_line += " " + "".join(tags)

                module_node = tree.add(node_line)

                if description and module.description is not None:
                    module_node.add(f"Description: {module.description}")

                if module.options is not None and len(module.options) > 0:
                    opt_line = [f"[bold]{module.options[0]} (default)[/bold]"] + module.options[1:] if module.options else []
                    module_node.add("[bright_black]Available options: " + ", ".join(opt_line) + "[/bright_black]")

        else:
            tree.add("(none)")

        return tree

    local_tree = print_section(local_entries, "local")
    console.print(local_tree)

    builtin_tree = print_section(builtin_entries, "built-in")
    console.print(builtin_tree)


# --- Commands ---


def list_judges(args):
    local, any_util_llm_local = _collect_local("judges")
    builtin, any_util_llm_builtin = _collect_builtin("spikee.judges", "judges")
    _render_section("Judges", local, builtin, (any_util_llm_local or any_util_llm_builtin), args.description)


def list_targets(args):
    local, _ = _collect_local("targets")
    builtin, _ = _collect_builtin("spikee.targets", "targets")
    _render_section("Targets", local, builtin)


def list_plugins(args):
    local, any_util_llm_local = _collect_local("plugins")
    builtin, any_util_llm_builtin = _collect_builtin("spikee.plugins", "plugins")
    _render_section("Plugins", local, builtin, (any_util_llm_local or any_util_llm_builtin), args.description)


def list_attacks(args):
    local, any_util_llm_local = _collect_local("attacks")
    builtin, any_util_llm_builtin = _collect_builtin("spikee.attacks", "attacks")
    _render_section("Attacks", local, builtin, (any_util_llm_local or any_util_llm_builtin), args.description)
