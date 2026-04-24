"""
LLM-powered spikee command generator with optimized context loading.

This module provides natural language command generation for spikee using an LLM provider
with a two-stage approach:
1. First, classify the query to determine command type
2. Then, load only relevant documentation and module information

Usage:
    spikee docs "test gpt-4o-mini with the cybersec dataset"
    spikee docs "generate dataset with base64 plugin" --model bedrock/claude45-haiku
"""

import sys
import re
from typing import Tuple, Dict, Any

from spikee.utilities.llm import get_llm
from spikee.utilities.llm_message import SystemMessage, HumanMessage
from spikee.utilities.modules import (
    extract_json_or_fail,
    load_module_from_path,
    get_options_from_module,
    get_description_from_module,
    collect_modules,
    collect_seeds,
    collect_datasets,
)

try:
    from rich.console import Console
    from rich.syntax import Syntax
except ImportError:
    Console = None

DEFAULT_MODEL = "openai/gpt-4o"

# === Stage 1: Command Classification ===

CLASSIFIER_PROMPT = """You are a command classifier for the SPIKEE toolkit.

Analyze the user's query and determine which spikee command they want to use.

Available commands:
- generate: Creating test datasets from seed folders
- test: Testing targets with datasets (includes attacks, judges, sampling)
- results: Analyzing, rejudging, or extracting test results
- list: Listing available modules (seeds, datasets, targets, plugins, attacks, judges, providers)
- init: Initializing workspace
- viewer: Launching web viewers
- unknown: Cannot determine or doesn't match spikee commands

Respond with ONLY a JSON object:
{
  "command": "<command_type>",
  "confidence": "<high|medium|low>"
}

Examples:
"test gpt-4o with my dataset" -> {"command": "test", "confidence": "high"}
"generate dataset with plugins" -> {"command": "generate", "confidence": "high"}
"show me all plugins" -> {"command": "list", "confidence": "high"}
"analyze my results" -> {"command": "results", "confidence": "high"}
"setup workspace" -> {"command": "init", "confidence": "medium"}
"""

# === Stage 2: Modular Documentation Sections ===

COMMON_HEADER = """You are an expert assistant for the SPIKEE toolkit - a prompt injection and jailbreaking testing framework.

Your task is to generate valid spikee CLI commands based on natural language descriptions from users.

# Common Patterns:

1. **LLM Provider Format**: Always use "provider/model" format
   - OpenAI: "openai/gpt-4o", "openai/gpt-4o-mini"
   - Bedrock: "bedrock/claude45-sonnet", "bedrock/claude45-haiku"
   - Azure: "azure/gpt-4"
   - Groq: "groq/llama-3.1-70b"
   - DeepSeek: "deepseek/deepseek-chat"

2. **Plugin Piping**: Use | to pipe plugins: "plugin1|plugin2|plugin3"

3. **Options Format**: "module:key1=val1,key2=val2;module2:key3=val3"

4. **Dataset Wildcards**: Can use wildcards in paths: "datasets/cybersec-*.jsonl"
"""

GENERATE_DOCS = """
## GENERATE Command

### Required Arguments:
- `--seed-folder <path>` - REQUIRED: Path to seed folder (e.g., datasets/seeds-cybersec-2026-01)

### Optional Source Arguments:
- `--include-standalone-inputs` - Include standalone_user_inputs.jsonl
- `--include-system-message` - Include system_messages.toml
- `--tag <name>` - Tag for dataset filename

### Optional Transformation Arguments:
- `--plugins <plugins>` - Space-separated list of plugins OR piped plugins with | (e.g., "1337 base64" or "splat|base64")
- `--plugin-options "<opts>"` - Plugin options: "plugin1:option1=value1,option2=value2;plugin2:option2=value2"
- `--plugin-only` - Only output plugin entries
- `--include-fixes <fixes>` - Comma-separated: adv_prefixes, adv_suffixes, prefixes=<filename>, suffixes=<filename>, prefix=<text>, suffix=<text>

### Optional Formatting Arguments:
- `--format <type>` - Output format: user-input (default/apps), full-prompt (LLMs), or burp
- `--languages <langs>` - Comma-separated list of languages to filter (e.g., en)
- `--match-languages` - Only combine jailbreaks/instructions with matching languages (default: True)
- `--positions <positions>` - Position to insert jailbreaks: start, middle, end (ignored if <PLACEHOLDER> present)
- `--injection-delimiters <delims>` - Delimiters for injecting jailbreaks (default: \\nINJECTION_PAYLOAD\\n)
- `--spotlighting-data-markers <markers>` - Comma-separated data markers (placeholder: "DOCUMENT")
- `--instruction-filter <types>` - Comma-separated instruction types to include
- `--jailbreak-filter <types>` - Comma-separated jailbreak types to include

### Examples:
```bash
# Basic generation
spikee generate --seed-folder datasets/seeds-cybersec-2026-01

# With plugins
spikee generate --seed-folder datasets/seeds-toxic-chat --plugins "1337 base64"

# Plugin piping
spikee generate --seed-folder datasets/seeds-cybersec-2026-01 --plugins "splat|base64"

# With plugin options
spikee generate --seed-folder datasets/seeds-example --plugins best_of_n --plugin-options "best_of_n:variants=50"

# With standalone inputs
spikee generate --seed-folder datasets/seeds-in-the-wild --include-standalone-inputs

# With adversarial fixes
spikee generate --seed-folder datasets/seeds-cybersec-2026-01 --include-fixes "adv_prefixes,adv_suffixes"
```
"""

TEST_DOCS = """
## TEST Command

### Required Dataset Arguments (at least one required):
- `--dataset <path>` - Path to dataset JSONL file (can be used multiple times)
- `--dataset-folder <path>` - Path to folder with multiple JSONL files (can be used multiple times)

### Required Module Arguments:
- `--target <name>` - REQUIRED: Target module name (e.g., llm_provider, aws_bedrock_guardrail)

### Optional Module Arguments:
- `--target-options "<provider/model>"` - Target options, typically "provider/model" format
  - Examples: "openai/gpt-4o-mini", "bedrock/claude45-sonnet", "azure/gpt-4"
- `--judge-options "<provider/model>"` - LLM judge model (format: "model=provider/model" or just "provider/model")
  - Examples: "bedrock/claude45-haiku", "openai/gpt-4o"
  - Only needed for datasets requiring semantic evaluation (not canary-based)

### Optional Testing Arguments:
- `--threads <n>` - Number of parallel threads (default: 4)
- `--attempts <n>` - Number of attempts per entry (default: 1)
- `--max-retries <n>` - Number of retries for rate-limiting/429 errors (default: 3)
- `--throttle <seconds>` - Time to wait between entries per thread (default: 0)
- `--sample <percentage>` - Sample percentage of dataset (e.g., 0.15 for 15%, default: 1)
- `--sample-seed <n>` - Seed for random sampling (default: 42)
- `--tag <name>` - Tag for results filename

### Optional Attack Arguments:
- `--attack <name>` - Attack module to use
- `--attack-iterations <n>` - Number of attack iterations/turns per entry
- `--attack-options "<attack:opt=val>"` - Attack-specific options
- `--attack-only` - Only run attack module, skip standard attempts

### Optional Resume Arguments:
- `--resume-file <path>` - Resume from specific results JSONL file (single dataset only)
- `--auto-resume` - Silently resume from latest matching results file
- `--no-auto-resume` - Create new results file, don't resume

### Examples:
```bash
# Basic test
spikee test --dataset datasets/cybersec-2026-01.jsonl --target llm_provider --target-options "openai/gpt-4o-mini"

# With LLM judge
spikee test --dataset datasets/harmful.jsonl --target llm_provider --target-options "openai/gpt-4o-mini" --judge-options "bedrock/claude45-haiku"

# Multiple datasets
spikee test --dataset datasets/dataset1.jsonl --dataset datasets/dataset2.jsonl --target llm_provider --target-options "bedrock/claude45-sonnet"

# With attack
spikee test --dataset datasets/example.jsonl --target llm_provider --target-options "openai/gpt-4o" --attack best_of_n --attack-iterations 25

# With sampling
spikee test --dataset datasets/large.jsonl --target llm_provider --target-options "openai/gpt-4o" --sample 0.1 --sample-seed 123
```
"""

RESULTS_DOCS = """
## RESULTS Command

### Subcommands:
1. `results analyze` - Analyze test results with statistics and visualizations
2. `results rejudge` - Re-judge results with different judge
3. `results extract` - Extract specific results by category or search term
4. `results dataset-comparison` - Compare datasets across multiple targets
5. `results convert-to-excel` - Convert results JSONL to Excel format

### analyze Arguments:
- `--results-file <path>` - Path to results JSONL file (can be used multiple times)
- `--results-folder <path>` - Path to folder with results files (can be used multiple times)
- `--false-positive-checks <path>` - JSONL file with benign prompts for FP analysis (single dataset only)
- `--output-format <type>` - Output format: console (default) or html
- `--overview` - Only output general statistics
- `--combine` - Combine multiple results files into single analysis

### rejudge Arguments:
- `--results-file <path>` - Path to results JSONL file (can be used multiple times)
- `--results-folder <path>` - Path to folder with results files (can be used multiple times)
- `--judge-options <opts>` - Options to pass to the judge
- `--resume` - Resume from most recent re-judge file

### extract Arguments:
- `--results-file <path>` - Path to results JSONL file (can be used multiple times)
- `--results-folder <path>` - Path to folder with results files (can be used multiple times)
- `--category <cat>` - Category: success (default), failure, error, guardrail, no-guardrail, custom
- `--custom-search <search>` - Custom search: 'string', 'field:string', or '!string' to invert
- `--tag <name>` - Tag for results filename

### convert-to-excel Arguments:
- `--result-file <path>` - Path to results JSONL file (required)

### Examples:
```bash
# Analyze results
spikee results analyze --results-file results/test-run.jsonl

# Rejudge with different judge
spikee results rejudge --results-file results/test.jsonl --judge-options "openai/gpt-4o"

# Extract successful prompts
spikee results extract --results-file results/test.jsonl --category success
```
"""

LIST_DOCS = """
## LIST Command

### Subcommands:
- `list seeds` - List available seed folders
- `list datasets` - List available dataset JSONL files
- `list targets` - List available targets
- `list judges` - List available judges
- `list plugins` - List available plugins
- `list attacks` - List available attack scripts
- `list providers` - List available LLM providers

### Optional Arguments (for targets, judges, plugins, attacks, providers):
- `-d`, `--description` - Include module descriptions

### Examples:
```bash
spikee list seeds
spikee list targets --description
spikee list plugins -d
```
"""

INIT_DOCS = """
## INIT Command

### Arguments:
- `--force` - Overwrite existing directories
- `--include-builtin <type>` - Copy built-in modules to local workspace
- `--include-viewer` - Include built-in web viewer in local workspace

### Examples:
```bash
# Basic workspace initialization
spikee init

# With built-in modules
spikee init --include-builtin all

# Force overwrite
spikee init --force
```
"""

VIEWER_DOCS = """
## VIEWER Command

### Subcommands:
- `viewer results` - Launch results viewer

### Common Arguments:
- `-h`, `--host <address>` - Host address (default: 127.0.0.1)
- `-p`, `--port <n>` - Port number (default: 8080)
- `-d`, `--debug` - Enable debug mode with hot-reloading (default: False)
- `--truncate <n>` - Truncate long fields (default: 500 chars, 0 to disable)

### results Viewer Arguments:
- `--result-file <path>` - Path to results JSONL file (can be used multiple times)
- `--result-folder <path>` - Path to results folder (can be used multiple times)
- `--allow-ast` - Allow AST parsing (use with caution)

### Examples:
```bash
# Launch results viewer
spikee viewer results --result-folder results/

# Custom port
spikee viewer -p 8081 results --result-file results/test.jsonl
```
"""

RESPONSE_FORMAT = """
# Your Response Format:

You MUST respond with ONLY valid JSON in this exact format:

{
  "command": "spikee <full command here>",
  "explanation": "Clear explanation of what this command does",
  "options": {
    "useful_module_options": ["List of 2-4 useful module-specific options (e.g., --plugin-options, --attack-options, --target-options) that could enhance this command. ONLY include if the command uses modules like plugins, attacks, targets, or judges"]
  }
}

# Important Guidelines:

1. Generate VALID spikee commands only - use exact argument names and formats shown above
2. ONLY use modules from the available modules list provided
3. Use real seed folders and datasets from the available lists when provided
4. If paths are not specified, use appropriate placeholders from the available lists
5. Use appropriate defaults (e.g., openai/gpt-4o-mini for testing if not specified)
6. Include clear explanations that help users understand what the command does
7. If user mentions a specific LLM provider/model, use it in the command
8. When using LLM-based modules, always include model in options
9. In the "options" field, ONLY suggest useful module-specific options if the command uses modules (plugins, attacks, targets, judges). Do NOT include general command arguments like --threads or --sample
10. Return ONLY the JSON - no additional text before or after
"""


def parse_query_for_model(query: str) -> Tuple[str, str]:
    """
    Extract model specification from query if present.
    
    Patterns detected:
    - "using openai/gpt-4o"
    - "with bedrock/claude45-sonnet"
    - "model=openai/gpt-4"
    
    Args:
        query: Natural language query string
        
    Returns:
        Tuple of (cleaned_query, model_name_or_none)
    """
    pattern = r'\b(using|with|model=)\s*([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)\b'
    match = re.search(pattern, query, re.IGNORECASE)
    
    if match:
        model = match.group(2)
        cleaned = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned, model
    
    return query, ""


def classify_command(query: str, model: str = DEFAULT_MODEL) -> Dict[str, str]:
    """
    Classify the user's query to determine which spikee command they want.
    
    Args:
        query: Natural language query
        model: LLM model to use for classification
        
    Returns:
        Dictionary with command type and confidence level
    """
    try:
        # Use faster, cheaper model with minimal tokens for classification
        llm = get_llm(model, max_tokens=50, temperature=0)
        if llm is None:
            raise ValueError("Failed to initialize LLM")
        
        messages = [
            SystemMessage(content=CLASSIFIER_PROMPT),
            HumanMessage(content=query)
        ]
        
        response = llm.invoke(messages).content

        if not isinstance(response, str) or response.strip() == "":
            raise ValueError("LLM returned empty response for classification")
        
        result = extract_json_or_fail(response)
        
        return result
    except Exception as e:
        # Fallback to general command if classification fails
        return {"command": "unknown", "confidence": "low"}


def get_seeds_info() -> str:
    """
    Get information about available seed folders in the workspace.
    
    Returns:
        Formatted string with seed folder names
    """
    try:
        seeds = collect_seeds()
        
        if not seeds:
            return "\n## Available Seed Folders:\n\nNo seed folders found in datasets/. Use 'spikee init' to initialize workspace."
        
        lines = ["\n## Available Seed Folders:\n"]
        lines.append("Found in datasets/ directory:")
        for seed in seeds:
            lines.append(f"- datasets/{seed}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"\n## Available Seed Folders:\n\nError loading seeds: {str(e)}"


def get_datasets_info() -> str:
    """
    Get information about available datasets in the workspace.
    
    Returns:
        Formatted string with dataset names
    """
    try:
        datasets = collect_datasets()
        
        if not datasets:
            return "\n## Available Datasets:\n\nNo datasets found in datasets/. Generate datasets using 'spikee generate'."
        
        lines = ["\n## Available Datasets:\n"]
        lines.append("Found in datasets/ directory:")
        for dataset in datasets:
            lines.append(f"- datasets/{dataset}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"\n## Available Datasets:\n\nError loading datasets: {str(e)}"


def get_module_info(module_type: str) -> str:
    """
    Get information about available modules from local and built-in sources.
    
    Args:
        module_type: Type of module (plugins, targets, attacks, judges, providers)
        
    Returns:
        Formatted string with module names and options
    """
    try:
        # Use collect_modules from utilities
        all_names, local_names, builtin_names = collect_modules(module_type)
        
        if not all_names:
            return f"\n## Available {module_type.title()}:\n\nNo {module_type} available."
        
        lines = [f"\n## Available {module_type.title()}:\n"]
        
        # Track if any module requires LLM
        any_llm_required = False
        
        for name in all_names:
            try:
                # Load module to get info
                module = load_module_from_path(name, module_type)
                
                # Get options
                options_result = get_options_from_module(module, module_type)
                util_llm = False
                options = []
                
                if options_result is not None:
                    if isinstance(options_result, tuple) and len(options_result) == 2:
                        options, util_llm = options_result
                    else:
                        options = options_result if isinstance(options_result, list) else [options_result]
                
                if util_llm:
                    any_llm_required = True
                
                # Get description
                description_result = get_description_from_module(module, module_type)
                tags = []
                description = ""
                
                if description_result is not None:
                    if isinstance(description_result, tuple) and len(description_result) == 2:
                        tags, description = description_result
                    else:
                        description = description_result if isinstance(description_result, str) else ""
                
                # Build module line
                line = f"- **{name}**"
                
                # Add source indicator
                if name in local_names:
                    line += " [Local]"
                
                # Add tags if available
                if tags:
                    if hasattr(tags[0], 'value'):  # ModuleTag enum
                        tags_str = ", ".join([tag.value for tag in tags])
                    else:
                        tags_str = ", ".join(str(tag) for tag in tags)
                    line += f" [{tags_str}]"
                
                # Add options if available
                if options and len(options) > 0 and not (isinstance(options[0], str) and options[0].startswith("<error")):
                    opts_str = ", ".join(str(opt) for opt in options)
                    line += f" (options: {opts_str})"
                
                # Add description if available
                if description:
                    line += f"\n  {description}"
                
                # Add LLM note if needed
                if util_llm:
                    line += "\n  [LLM] Requires model=provider/model in options"
                
                lines.append(line)
                
            except Exception as e:
                # If module fails to load, still list it but with error
                line = f"- **{name}**"
                if name in local_names:
                    line += " [Local]"
                line += f" [Error loading: {str(e)[:50]}]"
                lines.append(line)
        
        # Add general LLM note if any module requires it
        if any_llm_required:
            lines.insert(1, "\n**Note**: Modules marked with [LLM] require a model parameter in options.")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"\n## Available {module_type.title()}:\n\nError loading {module_type}: {str(e)}"


def build_context_for_command(command_type: str) -> str:
    """
    Build contextual documentation based on detected command type.
    
    Args:
        command_type: The type of command (generate, test, results, etc.)
        
    Returns:
        Relevant documentation string
    """
    context = COMMON_HEADER
    
    if command_type == "generate":
        context += GENERATE_DOCS
        context += get_seeds_info()  # Add available seed folders
        context += get_module_info("plugins")
        
    elif command_type == "test":
        context += TEST_DOCS
        context += get_datasets_info()  # Add available datasets
        context += get_module_info("targets")
        context += get_module_info("attacks")
        context += get_module_info("judges")
        
    elif command_type == "results":
        context += RESULTS_DOCS
        context += get_module_info("judges")  # For rejudge
        
    elif command_type == "list":
        context += LIST_DOCS
        
    elif command_type == "init":
        context += INIT_DOCS
        
    elif command_type == "viewer":
        context += VIEWER_DOCS
        
    else:  # unknown - provide general overview
        context += "\n## All Commands Available:\n"
        context += "- generate: Create test datasets\n"
        context += "- test: Test targets with datasets\n"
        context += "- results: Analyze, rejudge, or extract results\n"
        context += "- list: List available modules\n"
        context += "- init: Initialize workspace\n"
        context += "- viewer: Launch web viewers\n"
        context += "\nPlease rephrase your query to be more specific.\n"
    
    context += RESPONSE_FORMAT
    
    return context


def build_explanation_context(command_type: str) -> str:
    """
    Build contextual documentation for explaining spikee commands.
    
    Args:
        command_type: The type of command being explained (generate, test, results, etc.)
        
    Returns:
        Relevant documentation string for explanations
    """
    # Start with common header
    context = COMMON_HEADER
    
    # Add command-specific explanation context
    context += "\n## Command Explanation Context\n"
    context += "This section provides detailed information to explain spikee commands based on user queries.\n"
    
    if command_type == "generate":
        context += "\n### Generate Command Context\n"
        context += "The generate command creates test datasets from seed folders with optional transformations.\n"
        context += GENERATE_DOCS
        context += get_seeds_info()
        context += get_module_info("plugins")
    
    elif command_type == "test":
        context += "\n### Test Command Context\n"
        context += "The test command evaluates targets with datasets, optionally using attacks and judges.\n"
        context += TEST_DOCS
        context += get_datasets_info()
        context += get_module_info("targets")
        context += get_module_info("attacks")
        context += get_module_info("judges")
    
    elif command_type == "results":
        context += "\n### Results Command Context\n"
        context += "The results command analyzes, rejudges, or extracts test results.\n"
        context += RESULTS_DOCS
        context += get_module_info("judges")
    
    elif command_type == "list":
        context += "\n### List Command Context\n"
        context += "The list command shows available modules for various categories.\n"
        context += LIST_DOCS
    
    elif command_type == "init":
        context += "\n### Init Command Context\n"
        context += "The init command initializes a new spikee workspace.\n"
        context += INIT_DOCS
    
    elif command_type == "viewer":
        context += "\n### Viewer Command Context\n"
        context += "The viewer command launches web viewers for results.\n"
        context += VIEWER_DOCS
    
    else:  # unknown - provide general overview
        context += "\n### General Command Context\n"
        context += "When the command type is unknown, here's a general overview of all spikee commands:\n"
        context += "- generate: Create test datasets\n"
        context += "- test: Test targets with datasets\n"
        context += "- results: Analyze, rejudge, or extract results\n"
        context += "- list: List available modules\n"
        context += "- init: Initialize workspace\n"
        context += "- viewer: Launch web viewers\n"
        context += "\nPlease rephrase your query to be more specific.\n"
    
    # Add explanation-specific response format
    context += "\n# Your Response Format:\n"
    context += "You MUST respond with ONLY valid JSON in this exact format:\n"
    context += "{\n"
    context += "  \"explanation\": \"Clear explanation of the spikee command(s) requested\"\n"
    context += "}\n"
    
    return context


def display_explanation(explanation_dict: Dict[str, Any]) -> None:
    """
    Display generated explanation with formatting.
    
    Args:
        explanation_dict: Dictionary containing explanation
    """
    if Console is None:
        # Fallback to plain text if rich is not available
        print("\nExplanation:")
        print(explanation_dict["explanation"])
        print()
        return
    
    console = Console()
    
    # Display header
    console.print()
    console.print("[bold cyan]Explanation:[/bold cyan]")
    
    # Display explanation
    console.print()
    console.print("[bold green]" + explanation_dict["explanation"] + "[/bold green]")
    console.print()


def generate_command(query: str, model: str = DEFAULT_MODEL, verbose: bool = False) -> Dict[str, Any]:
    """
    Generate a spikee command from natural language query using an LLM with optimized context.
    
    This uses a two-stage approach:
    1. Classify the query to determine command type (fast, minimal tokens)
    2. Load only relevant documentation and generate command (focused context)
    
    Args:
        query: Natural language description of desired command
        model: LLM model to use (format: provider/model)
        verbose: If True, print classification info
        
    Returns:
        Dictionary with keys: command, explanation, options (containing useful_module_options)
        
    Raises:
        Exception: If LLM call fails or response parsing fails
    """
    try:
        # Stage 1: Classify command type
        classification = classify_command(query, model)
        command_type = classification.get("command", "unknown")
        confidence = classification.get("confidence", "low")
        
        if verbose:
            print(f"[Debug] Classified as: {command_type} (confidence: {confidence})")
        
        # Stage 2: Build context and generate command
        context = build_context_for_command(command_type)
        
        if verbose:
            print(f"[Debug] Context size: {len(context)} chars")
        
        # Initialize LLM with appropriate token limit (increased for options)
        llm = get_llm(model, max_tokens=1200, temperature=0.3)
        
        # Create messages
        messages = [
            SystemMessage(content=context),
            HumanMessage(content=query)
        ]
        
        # Get response
        response = llm.invoke(messages)
        
        # Parse JSON response
        result = extract_json_or_fail(response.content)
        
        # Validate required fields
        if "command" not in result or "explanation" not in result:
            raise ValueError("LLM response missing required fields (command, explanation)")
        
        # Options field is optional but should have default structure if missing
        if "options" not in result:
            result["options"] = {
                "useful_module_options": []
            }
        
        return result
        
    except ImportError as e:
        raise Exception(f"Invalid LLM provider specification: {e}\n"
                       f"Use format: provider/model (e.g., openai/gpt-4o)")
    except Exception as e:
        raise Exception(f"Error generating command: {e}\n"
                       "Please try rephrasing your query or check your API credentials in .env file")


def explain_command(query: str, model: str = DEFAULT_MODEL, verbose: bool = False) -> Dict[str, Any]:
    """
    Generate an explanation of spikee commands for natural language queries.
    
    Args:
        query: Natural language query about spikee commands
        model: LLM model to use (format: provider/model)
        verbose: If True, print classification info
        
    Returns:
        Dictionary with key: explanation
        
    Raises:
        Exception: If LLM call fails or response parsing fails
    """
    try:
        # Stage 1: Classify command type
        classification = classify_command(query, model)
        command_type = classification.get("command", "unknown")
        confidence = classification.get("confidence", "low")
        
        if verbose:
            print(f"[Debug] Classified as: {command_type} (confidence: {confidence})")
        
        # Stage 2: Build context for explanation
        context = build_explanation_context(command_type)
        
        if verbose:
            print(f"[Debug] Context size: {len(context)} chars")
        
        # Initialize LLM with appropriate token limit
        llm = get_llm(model, max_tokens=800, temperature=0.3)
        
        # Create messages
        messages = [
            SystemMessage(content=context),
            HumanMessage(content=query)
        ]
        
        # Get response
        response = llm.invoke(messages)
        
        # Parse JSON response
        result = extract_json_or_fail(response.content)
        
        # Validate required fields
        if "explanation" not in result:
            raise ValueError("LLM response missing required field (explanation)")
        
        return result
        
    except ImportError as e:
        raise Exception(f"Invalid LLM provider specification: {e}\n"
                       f"Use format: provider/model (e.g., openai/gpt-4o)")
    except Exception as e:
        raise Exception(f"Error generating explanation: {e}\n"
                       "Please try rephrasing your query or check your API credentials in .env file")


def display_command(command_dict: Dict[str, Any]) -> None:
    """
    Display generated command with formatting.
    
    Args:
        command_dict: Dictionary containing command, explanation, and options
    """
    if Console is None:
        # Fallback to plain text if rich is not available
        print("\nGenerated Command:")
        print(command_dict["command"])
        print("\nExplanation:")
        print(command_dict["explanation"])
        
        # Display options if available
        if "options" in command_dict:
            options = command_dict["options"]
            
            if options.get("useful_module_options"):
                print("\nUseful Module Options:")
                for opt in options["useful_module_options"]:
                    print(f"  • {opt}")
        
        print()
        return
    
    console = Console()
    
    # Display header
    console.print()
    console.print("[bold cyan]Generated Command:[/bold cyan]")
    
    # Display command with syntax highlighting
    syntax = Syntax(command_dict["command"], "bash", theme="monokai", line_numbers=False)
    console.print(syntax)
    
    # Display explanation
    console.print()
    console.print("[bold green]Explanation:[/bold green]")
    console.print(command_dict["explanation"])
    
    # Display options if available
    if "options" in command_dict:
        options = command_dict["options"]
        
        if options.get("useful_module_options"):
            console.print()
            console.print("[bold yellow]Useful Module Options:[/bold yellow]")
            for opt in options["useful_module_options"]:
                console.print(f"  [dim]•[/dim] {opt}")
    
    console.print()


def docs_command(args) -> None:
    """
    Entry point for the 'spikee docs' command.
    
    Args:
        args: Parsed command-line arguments
    """
    
    # Check if any subcommand was provided
    if hasattr(args, 'subcommand') and args.subcommand:
        if args.subcommand == 'generate':
            docs_generate(args)
        elif args.subcommand == 'explain':
            docs_explain(args)
        else:
            print(f"Error: Unknown subcommand '{args.subcommand}'. Use 'spikee docs --help' for available subcommands.")
            sys.exit(1)
    else:
        # Default behavior: run generate mode
        docs_generate(args)


def docs_generate(args) -> None:
    """
    Generate a spikee command from a natural language query.
    
    Args:
        args: Parsed command-line arguments
    """
    # Join query parts into single string
    query = " ".join(args.query)
    
    if not query.strip():
        print("Error: Please provide a query describing the spikee command you want to generate.")
        print("\nExample: spikee docs generate \"test gpt-4o-mini with my dataset\"")
        sys.exit(1)
    
    # Determine which model to use
    model = None
    
    # First priority: --model flag
    if hasattr(args, 'model') and args.model:
        model = args.model
    
    # Second priority: model specified in query
    if model is None:
        cleaned_query, parsed_model = parse_query_for_model(query)
        if parsed_model:
            query = cleaned_query
            model = parsed_model
    
    # Third priority: default model
    if model is None:
        model = DEFAULT_MODEL
    
    # Check for verbose mode
    verbose = hasattr(args, 'verbose') and args.verbose
    
    # Show what we're doing
    if Console:
        console = Console()
        console.print(f"[dim]Generating spikee command using {model}...[/dim]")
    else:
        print(f"Generating spikee command using {model}...")
    
    # Generate command
    try:
        result = generate_command(query, model, verbose=verbose)
        display_command(result)
    except Exception as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)


def docs_explain(args) -> None:
    """
    Explain spikee commands or provide information about them.
    
    Args:
        args: Parsed command-line arguments
    """
    # Join query parts into single string
    query = " ".join(args.query)
    
    if not query.strip():
        print("Error: Please provide a query about spikee commands to explain.")
        print("\nExample: spikee docs explain \"how to test with a custom model\"")
        sys.exit(1)
    
    # Determine which model to use
    model = None
    
    # First priority: --model flag
    if hasattr(args, 'model') and args.model:
        model = args.model
    
    # Second priority: model specified in query
    if model is None:
        cleaned_query, parsed_model = parse_query_for_model(query)
        if parsed_model:
            query = cleaned_query
            model = parsed_model
    
    # Third priority: default model
    if model is None:
        model = DEFAULT_MODEL
    
    # Check for verbose mode
    verbose = hasattr(args, 'verbose') and args.verbose
    
    # Show what we're doing
    if Console:
        console = Console()
        console.print(f"[dim]Explaining spikee commands using {model}...[/dim]")
    else:
        print(f"Explaining spikee commands using {model}...")
    
    # Generate explanation
    try:
        result = explain_command(query, model, verbose=verbose)
        display_explanation(result)
    except Exception as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)
