import os
from pathlib import Path
import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
import rich.box

from spikee.templates import module
from spikee.utilities.enums import ModuleTag, module_tag_to_colour, formatting_priority
from spikee.utilities.modules import (
    load_module_from_path,  
    get_options_from_module,
    get_description_from_module,
    collect_seeds,
    collect_datasets,
    collect_modules,
)

console = Console()


def list_seeds(args):
    seeds = collect_seeds()
    console.print(
        Panel(
            "\n".join(seeds) if seeds else "(none)", title="[seeds] Local", style="cyan"
        )
    )


def list_datasets(args):
    files = collect_datasets()
    console.print(
        Panel(
            "\n".join(files) if files else "(none)", title="[datasets] Local", style="cyan"
        )
    )


# --- Helpers ---


@dataclass
class Module:
    name: str
    options: Optional[List[str]] = None
    tags: Optional[List[ModuleTag]] = None
    description: Optional[str] = ""

def _render_section(
    module_type: str,
    local,
    builtin,
    add_description: bool = False,
    tag_line: str = "Available options",
):
    console.print(Rule(f"[bold]{module_type.capitalize()}[/bold]"))

    uses_llm = False
    local_entries = []
    builtin_entries = []

    def collect_module_data(module):
        uses_llm = False

        try:
            mod = load_module_from_path(module, module_type)

            options_data = get_options_from_module(mod)
            if options_data is not None and isinstance(options_data, tuple) and len(options_data) == 2:
                if options_data[1]:
                    uses_llm = True

                options = options_data[0]
            else:
                options = options_data

            description_data = get_description_from_module(mod)
            if description_data is not None and len(description_data) == 2:
                tags, description = description_data
            else:
                tags, description = [], ""
        
        except Exception as e:
            error = e if len(str(e)) < 70 else str(e)[:70] + "..."
            options = [f"<error - {error}>"]
            tags = []
            description = ""
        
        module_data = Module(
            name=module,
            options=options,
            tags=tags,
            description=description,
        )
        return module_data, uses_llm

    for local_module in local:
        module, mod_uses_llm = collect_module_data(local_module)
        local_entries.append(module)
        if mod_uses_llm:
            uses_llm = True

    for builtin_module in builtin:
        module, mod_uses_llm = collect_module_data(builtin_module)
        builtin_entries.append(module)
        if mod_uses_llm:
            uses_llm = True

    # If any module in this section uses the built-in LLM service, show a note about LLM options
    if uses_llm:
        providers, _, _ = collect_modules("providers")
        console.print(
            Panel(
                f"""[yellow]Note:[/yellow] Modules with a [yellow][LLM][/yellow] tag, use the built-in LLM service.
The LLM options are available, using 'model=<option>':
Supported Providers (use 'spikee list providers' for more): {", ".join(providers) if providers else "(none)"} 
""",
                style="yellow",
            )
        )

    def print_section(entries, label):
        if not entries:
            console.print(f"\n[bold]{module_type.capitalize()} ({label})[/bold]")
            console.print("  (none)")
            return

        table = Table(title=f"{module_type.capitalize()} ({label})", box=rich.box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
        table.add_column("Name", style="bold cyan", no_wrap=True)
        table.add_column("Tags")
        table.add_column(tag_line)
        if add_description:
            table.add_column("Description")

        def _module_sort_key(m):
            if m.tags:
                return tuple(sorted((formatting_priority(t), t.value) for t in m.tags))
            return ((99, ""),)

        for module in sorted(entries, key=_module_sort_key):
            # Tags
            if module.tags:
                sorted_tags = sorted(module.tags, key=lambda x: (formatting_priority(x), x.value))
                tag_parts = []
                for tag in sorted_tags:
                    c = module_tag_to_colour(tag)
                    tag_parts.append(f"[{c}]{tag.value}[/{c}]")
                tags_str = ", ".join(tag_parts)
            else:
                tags_str = ""

            # Options
            if module.options is not None and len(module.options) > 0:
                if module.options[0].startswith("<error - ") or module.options == ["<import error - check module for dependencies>"]:
                    opts_str = f"[red]{module.options[0]}[/red]"
                else:
                    opt_parts = (
                        [f"{module.options[0]} [bold][white](default)[/white][/bold]"]
                        + module.options[1:]
                    )
                    opts_str = ", ".join(opt_parts)
            else:
                opts_str = ""

            row = [module.name, tags_str, opts_str]
            if add_description:
                row.append(module.description or "")

            table.add_row(*row)

        console.print()
        console.print(table)

    print_section(local_entries, "local")
    print_section(builtin_entries, "built-in")


# --- Commands ---


def list_judges(args):
    _, local, builtin = collect_modules("judges")

    _render_section(
        module_type="judges",
        local=local,
        builtin=builtin,
        add_description=args.description,
    )


def list_targets(args):
    _, local, builtin = collect_modules("targets")

    _render_section(
        module_type="targets",
        local=local,
        builtin=builtin,
        add_description=args.description,
    )


def list_plugins(args):
    _, local, builtin = collect_modules("plugins")

    _render_section(
        module_type="plugins",
        local=local,
        builtin=builtin,
        add_description=args.description,
    )


def list_attacks(args):
    _, local, builtin = collect_modules("attacks")

    _render_section(
        module_type="attacks",
        local=local,
        builtin=builtin,
        add_description=args.description,
    )


def list_providers(args):
    _, local, builtin = collect_modules("providers")

    _render_section(
        module_type="providers",
        local=local,
        builtin=builtin,
        add_description=args.description,
        tag_line="Known supported models",
    )
