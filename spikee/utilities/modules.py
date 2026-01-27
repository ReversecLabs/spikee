import importlib
import inspect
import os

from spikee.templates.target import Target
from spikee.templates.judge import Judge
from spikee.templates.llm_judge import LLMJudge
from spikee.templates.plugin import Plugin
from spikee.templates.attack import Attack


BASE_CLASS_MAP = {
    "targets": (Target,),
    "judges": (Judge, LLMJudge),
    "plugins": (Plugin,),
    "attacks": (Attack,),
}


def _resolve_impl_class(module, module_type):
    """Return the first concrete implementation class for the given module."""
    base_classes = BASE_CLASS_MAP.get(module_type)
    if not base_classes or not inspect.ismodule(module):
        return None

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj in base_classes:
            continue
        if obj.__module__ != module.__name__:
            continue
        if issubclass(obj, base_classes):
            return obj
    return None


def _instantiate_impl(module, module_type):
    impl_class = _resolve_impl_class(module, module_type)
    return impl_class() if impl_class else None


# ==== Loading Modules ====
def load_module_from_path(name, module_type):
    """Loads a module either from a local path or from the spikee package."""
    local_path = os.path.join(os.getcwd(), module_type, f"{name}.py")
    if os.path.isfile(local_path):
        spec = importlib.util.spec_from_file_location(name, local_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            raise ImportError(f"Could not load module {name} from {local_path}")
    else:
        try:
            mod = importlib.import_module(f"spikee.{module_type}.{name}")

        except ModuleNotFoundError:
            raise ValueError(
                f"Module '{name}' not found locally or in spikee.{module_type}"
            )

    instance = _instantiate_impl(mod, module_type)
    if instance is not None:
        return instance

    return mod


def get_options_from_module(module, module_type=None):
    """
    Return the option values advertised by the given module or instance.

    Args:
        module: Either an instantiated module (new OOP) or the imported module.
        module_type: Optional str specifying the module category. Required when
            `module` is a module rather than an instance.
    """
    if module and hasattr(module, "get_available_option_values"):
        return module.get_available_option_values()

    if inspect.ismodule(module) and module_type:
        instance = _instantiate_impl(module, module_type)
        if instance and hasattr(instance, "get_available_option_values"):
            return instance.get_available_option_values()

    return None


def get_prefix_from_module(module, module_type=None):
    """
    Return the prefix values advertised by the given module or instance.

    Args:
        module: Either an instantiated module (new OOP) or the imported module.
        module_type: Optional str specifying the module category. Required when
            `module` is a module rather than an instance.
    """
    if module and hasattr(module, "get_available_prefixes"):
        return module.get_available_prefixes()

    if inspect.ismodule(module) and module_type:
        instance = _instantiate_impl(module, module_type)
        if instance and hasattr(instance, "get_available_prefixes"):
            return instance.get_available_prefixes()

    return None


def get_description_from_module(module, module_type=None):
    """
    Return the description advertised by the given module or instance.

    Args:
        module: Either an instantiated module (new OOP) or the imported module.
        module_type: Optional str specifying the module category. Required when
            `module` is a module rather than an instance.
    """
    if module and hasattr(module, "get_description"):
        return module.get_description()

    if inspect.ismodule(module) and module_type:
        instance = _instantiate_impl(module, module_type)
        if instance and hasattr(instance, "get_description"):
            return instance.get_description()

    return None


def get_default_option(module, module_type=None):
    available = get_options_from_module(module, module_type)
    return available[0] if available else None


if __name__ == "__main__":
    print(load_module_from_path("1337", "plugins"))
