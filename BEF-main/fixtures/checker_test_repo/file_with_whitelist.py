"""File that uses allowlist pattern - should PASS allowlist check."""

ALLOWED_MODULES = frozenset(["json", "os", "sys", "math"])

def safe_import(module_name):
    """Import a module, but only if it's on the allowlist."""
    if module_name not in ALLOWED_MODULES:
        raise ValueError(f"Module {module_name} not in allowlist")
    import importlib
    return importlib.import_module(module_name)


def process_data(data):
    """Process data safely."""
    # Use safe_import to load allowed modules
    json_mod = safe_import("json")
    return json_mod.dumps(data)
