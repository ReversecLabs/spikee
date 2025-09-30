# spikee/cli.py

# At the top of cli.py, add these imports:
import os
import sys
import shutil
import argparse
from . import __version__
from dotenv import load_dotenv
from pathlib import Path

from .generator import generate_dataset
from .tester import test_dataset
from .results import analyze_results, rejudge_results, convert_results_to_excel
from .list import (
    list_seeds,
    list_datasets,
    list_judges,
    list_targets,
    list_plugins,
    list_attacks,
)


banner = r"""
   _____ _____ _____ _  ________ ______ 
  / ____|  __ \_   _| |/ /  ____|  ____|
 | (___ | |__) || | | ' /| |__  | |__   
  \___ \|  ___/ | | |  < |  __| |  __|  
  ____) | |    _| |_| . \| |____| |____ 
 |_____/|_|   |_____|_|\_\______|______|
"""

# Explicitly load the .env file
env_loaded = load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def convert_to_new_args(args):
    """
    Normalizes deprecated flags/values to the new canonical ones and emits warnings.
    Keeps cli.py clean and readable.
    """
    # --format=document -> user-input
    if getattr(args, "format", None) == "document":
        print("DEPRECATION: --format=document → --format=user-input (default).")
        args.format = "user-input"

    # --standalone-attacks <path>  ->  --include-standalone-inputs (ignore supplied path)
    if getattr(args, "standalone_attacks", None) is not None:
        print(
            f"DEPRECATION: --standalone-attacks is ignored. "
            f"Use --include-standalone-inputs instead. "
            f"Ignoring supplied filename '{args.standalone_attacks}'."
        )
        args.include_standalone_inputs = True

    return args


def main():
    print(banner)
    print("SPIKEE - Simple Prompt Injection Kit for Evaluation and Exploitation")
    print(f"Version: {__version__}\n")
    print("Author: Reversec (reversec.com)\n")

    parser = argparse.ArgumentParser(
        description="SPIKEE - Simple Prompt Injection Kit for Evaluation and Exploitation"
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # === [INIT] Sub-command (NEW) ==============================================
    parser_init = subparsers.add_parser(
        "init", help="Initialize a local SPIKEE workspace"
    )
    parser_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing directories if they exist.",
    )
    parser_init.add_argument(
        "--include-builtin",
        choices=["none", "all", "plugins", "judges", "targets", "attacks"],
        default="none",
        help="Copy built-in modules to local workspace (default: none)",
    )

    # === [GENERATE] Sub-command ===============================================
    parser_generate = subparsers.add_parser("generate", help="Generate a dataset")
    parser_generate.add_argument(
        "--seed-folder",
        default="datasets/seeds-mini-test",
        help='Name of seed folder in local "datasets/" or built-in package data',
    )
    parser_generate.add_argument(
        "--positions",
        nargs="+",
        default=["end"],
        help="Positions to insert jailbreaks (start, middle, end). Ignored if the document entry contains a placeholder attribute.",
    )
    parser_generate.add_argument(
        "--injection-delimiters",
        default="\nINJECTION_PAYLOAD\n",
        help="Comma-separated list of patterns to use when injecting payloads",
    )
    parser_generate.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="List of plugin names to modify the jailbreak+instruction text",
    )
    parser_generate.add_argument(
        "--plugin-options",
        default=None,
        help='Plugin-specific options as "plugin1:option1,option2;plugin2:option2"',
    )
    parser_generate.add_argument(
        "--include-standalone-inputs",
        action="store_true",
        help="Include standalone_user_inputs.jsonl (fallback: standalone_attacks.jsonl)",
    )
    parser_generate.add_argument(
        "--standalone-attacks", default=None, help=argparse.SUPPRESS
    )  # hidden legacy alias
    parser_generate.add_argument(
        "--format",
        choices=["user-input", "full-prompt", "burp"],
        default="user-input",
        help="Output format: user-input (default, for apps), full-prompt, or burp",
    )
    parser_generate.add_argument(
        "--spotlighting-data-markers",
        default="\nDOCUMENT\n",
        help='Comma-separated list of data markers (placeholder: "DOCUMENT")',
    )
    parser_generate.add_argument(
        "--languages",
        default=None,
        help="Comma-separated list of languages to filter jailbreaks and instructions",
    )
    parser_generate.add_argument(
        "--match-languages",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Only combine jailbreaks and instructions with matching languages (default: True)",
    )
    parser_generate.add_argument(
        "--instruction-filter",
        default=None,
        help="Comma-separated list of instruction types to include",
    )
    parser_generate.add_argument(
        "--jailbreak-filter",
        default=None,
        help="Comma-separated list of jailbreak types to include",
    )
    parser_generate.add_argument(
        "--include-suffixes",
        action="store_true",
        help="Include advanced suffixes in the dataset generation",
    )
    parser_generate.add_argument(
        "--include-system-message",
        action="store_true",
        help="Include system message based on system_messages.toml",
    )
    parser_generate.add_argument(
        "--tag",
        default=None,
        help="Include a tag at the end of the generated dataset filename",
    )

    # === [TEST] Sub-command ===================================================
    parser_test = subparsers.add_parser(
        "test", help="Test the dataset against a target"
    )
    parser_test.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset file (local workspace)",
    )
    parser_test.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of the target to test (in local or built-in targets/ dir)",
    )
    parser_test.add_argument(
        "--target-options",
        type=str,
        required=False,
        help="Option to pass to the target [Optional]",
    )
    parser_test.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for parallel processing",
    )
    parser_test.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of attempts per payload (default: 1)",
    )
    parser_test.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of retries for failed requests due to API quotas/resources exhusted (default: 3)",
    )
    parser_test.add_argument(
        "--judge-options",
        type=str,
        default=None,
        help="Judge options, typically the name of the LLM to use as a judge",
    )
    parser_test.add_argument(
        "--resume-file",
        type=str,
        default=None,
        help="Path to a results JSONL file to resume from",
    )
    parser_test.add_argument(
        "--throttle",
        type=float,
        default=0,
        help="Time in seconds to wait between requests per thread",
    )
    parser_test.add_argument(
        "--attack",
        default=None,
        help='Name of the attack module (from the "attacks" folder) to use if standard attempts fail',
    )
    parser_test.add_argument(
        "--attack-iterations",
        type=int,
        default=100,
        help="Number of attack iterations per dataset entry (if --attack is provided)",
    )
    parser_test.add_argument(
        "--attack-options",
        default=None,
        help='Options to pass to the attack module (e.g., "mode-X")',
    )
    parser_test.add_argument(
        "--tag", default=None, help="Include a tag at the end of the results filename"
    )
    parser_test.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Sample a percentage of the dataset (e.g., 0.15 for 15%%)",
    )
    parser_test.add_argument(
        "--sample-seed",
        default="42",
        help='Seed for sampling (default: 42, or use "random" for random seed)',
    )
    group_resume = parser_test.add_mutually_exclusive_group()
    group_resume.add_argument(
        "--auto-resume",
        action="store_true",
        help="(non-tty) silently pick the latest matching results file if present; (tty) still prompts",
    )
    group_resume.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="(tty) disable interactive auto-resume prompt",
    )

    # === [RESULTS] Sub-command ================================================
    parser_results = subparsers.add_parser("results", help="Analyze or convert results")
    subparsers_results = parser_results.add_subparsers(
        dest="results_command", help="Results sub-commands"
    )

    # --- analyze
    parser_analyze = subparsers_results.add_parser(
        "analyze", help="Analyze the results JSONL file"
    )
    parser_analyze.add_argument(
        "--result-file", type=str, required=True, help="Path to the results JSONL file"
    )
    parser_analyze.add_argument(
        "--false-positive-checks",
        type=str,
        default=None,
        help="Path to a JSONL file with benign prompts for false positive analysis",
    )
    parser_analyze.add_argument(
        "--output-format",
        choices=["console", "html"],
        default="console",
        help="Output format: console (default) or html",
    )

    # --- rejudge
    parser_rejudge = subparsers_results.add_parser(
        "rejudge", help="Re-judge an offline results JSONL file"
    )
    parser_rejudge.add_argument(
        "--result-file",
        type=str,
        action="append",
        required=True,
        help="Path to the results JSONL file",
    )
    parser_rejudge.add_argument(
        "--judge-options",
        type=str,
        default=None,
        help="Judge options, typically the name of the LLM to use as a judge",
    )
    parser_rejudge.add_argument(
        "--resume",
        action="store_true",
        help="This will attempt to resume a re-judge the most recent results file. (Requires filename to be unmodified and in the same folder.)",
    )

    # --- convert-to-excel
    parser_convert_to_excel = subparsers_results.add_parser(
        "convert-to-excel", help="Convert results JSONL file to Excel"
    )
    parser_convert_to_excel.add_argument(
        "--result-file", type=str, required=True, help="Path to the results JSONL file"
    )

    # === [LIST] Sub-command ================================================
    parser_list = subparsers.add_parser(
        "list", help="List seeds, datasets, targets, or plugins"
    )
    list_subparsers = parser_list.add_subparsers(
        dest="list_command", help="What to list"
    )
    list_subparsers.add_parser("seeds", help="List available seed folders")
    list_subparsers.add_parser("datasets", help="List available dataset .jsonl files")
    list_subparsers.add_parser("judges", help="List available judges")
    list_subparsers.add_parser("targets", help="List available targets")
    list_subparsers.add_parser("plugins", help="List available plugins")
    list_subparsers.add_parser("attacks", help="List available attack scripts")

    args = convert_to_new_args(parser.parse_args())

    if args.command == "init":
        init_workspace(force=args.force, include_builtin=args.include_builtin)

    elif args.command == "generate":
        generate_dataset(args)
    elif args.command == "test":
        test_dataset(args)
    elif args.command == "results":
        if args.results_command == "analyze":
            analyze_results(args)
        elif args.results_command == "rejudge":
            rejudge_results(args)
        elif args.results_command == "convert-to-excel":
            convert_results_to_excel(args)
        else:
            parser_results.print_help()
    elif args.command == "list":
        if args.list_command == "seeds":
            list_seeds(args)
        elif args.list_command == "datasets":
            list_datasets(args)
        elif args.list_command == "judges":
            list_judges(args)
        elif args.list_command == "targets":
            list_targets(args)
        elif args.list_command == "plugins":
            list_plugins(args)
        elif args.list_command == "attacks":
            list_attacks(args)
        else:
            parser_list.print_help()
    else:
        parser.print_help()
        sys.exit(1)


def init_workspace(force=False, include_builtin="none"):
    """
    Copy the entire 'data/workspace' directory from the installed package
    into the user's current working directory. This sets up the local spikee workspace
    (datasets, plugins, targets, env-example, etc.).

    If include_builtin is specified, also copy built-in modules to their respective
    local directories for local modification.
    """
    cwd = Path(os.getcwd())
    workspace_dest = cwd

    src_folder = Path(__file__).parent / "data" / "workspace"

    # If not forcing, ensure we don't overwrite any existing files/folders
    # We'll do this by manually copying sub-items, skipping if they exist.
    for item in src_folder.iterdir():
        destination = workspace_dest / item.name

        if destination.exists() and not force:
            print(f"[init] '{destination}' already exists. Use --force to overwrite.")
            continue

        # If force and destination is a directory, remove it first
        if destination.exists() and force:
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()

        try:
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)
            print(f"[init] Copied {item.name} --> {destination}")
        except Exception as e:
            print(f"[init] Could not copy {item.name} to {destination}: {e}")

    print("[init] Local spikee workspace has been initialized.")

    # Handle copying built-in modules if requested
    if include_builtin != "none":
        copy_builtin_modules(include_builtin, force)


def copy_builtin_modules(include_option, force=False):
    """
    Copy built-in modules to local workspace based on the include_option.
    Uses direct file operations without importing the modules.
    """
    import spikee

    # Get the path to the spikee package
    spikee_path = Path(spikee.__file__).parent

    # Determine which module types to copy
    if include_option == "all":
        module_types = ["plugins", "judges", "targets", "attacks"]
    elif include_option in ["plugins", "targets", "attacks"]:
        module_types = [include_option]
    else:
        module_types = []

    for module_type in module_types:
        try:
            # Path to built-in modules
            module_dir = spikee_path / module_type

            # Ensure the module directory exists in the package
            if not module_dir.is_dir():
                print(
                    f"[init] Warning: Built-in {module_type} directory not found at {module_dir}"
                )
                continue

            # Ensure local directory exists
            local_dir = Path(os.getcwd()) / module_type
            os.makedirs(local_dir, exist_ok=True)

            # Copy each Python file (except __init__.py)
            modules_copied = 0
            for file_path in module_dir.glob("*.py"):
                if file_path.name == "__init__.py":
                    continue

                dest_file = local_dir / file_path.name

                if dest_file.exists() and not force:
                    print(
                        f"[init] '{dest_file}' already exists. Use --force to overwrite."
                    )
                    continue

                try:
                    shutil.copy2(file_path, dest_file)
                    modules_copied += 1
                    print(
                        f"[init] Copied built-in {module_type}/{file_path.name} to local workspace"
                    )
                except Exception as e:
                    print(f"[init] Error copying {module_type}/{file_path.name}: {e}")

            if modules_copied > 0:
                print(
                    f"[init] Copied {modules_copied} built-in {module_type} to local workspace"
                )
            else:
                print(f"[init] No built-in {module_type} were copied")

        except Exception as e:
            print(f"[init] Error processing {module_type}: {e}")
