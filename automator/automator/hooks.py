import os
import glob
import importlib.util


_HOOKS = {}


def register_hook(name):
    """Register a hook with the given name."""
    def decorator(func):
        _HOOKS[name] = func
    return decorator


def load_hooks():
    """Import all hooks from python files in ~/.automator/hooks."""
    hooks_dir = os.path.expanduser('~/.automator/hooks')
    os.makedirs(hooks_dir, exist_ok=True)
    for filepath in glob.glob(os.path.join(hooks_dir, '*.py')):
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
