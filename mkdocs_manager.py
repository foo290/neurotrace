import os

SRC_DIR = "neurotrace"
DOCS_DIR = "docs/reference/documentation"
IMPORT_PREFIX = "neurotrace"


def get_module_path(file_path):
    """Convert a file path to its corresponding Python module import path.

    Args:
        file_path: Path to the Python file relative to the project root.

    Returns:
        A string representing the Python import path for the module.

    Example:
        >>> get_module_path("eventflow/core/process.py")
        'eventflow.core.process'
    """
    rel_path = os.path.relpath(file_path, SRC_DIR)
    no_ext = os.path.splitext(rel_path)[0]
    return f"{IMPORT_PREFIX}." + ".".join(no_ext.split(os.sep))


def collect_modules(base_dir):
    """Collect all Python module files in the given directory and its subdirectories.

    Args:
        base_dir: Base directory to start the search from.

    Returns:
        A list of file paths to Python modules, excluding __init__.py files.

    Example:
        >>> modules = collect_modules("eventflow")
        >>> "eventflow/core/process.py" in modules
        True
    """
    modules = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".py") and not f.startswith("__init__"):
                full_path = os.path.join(root, f)
                modules.append(full_path)
    return modules


def write_markdown(module_path, import_path):
    """Write a Markdown documentation file for the given module.

    Args:
        module_path: File path to the Python module.
        import_path: Python import path for the module.

    Example:
        >>> write_markdown("eventflow/core/process.py", "eventflow.core.process")
        # Creates a file at docs/reference/documentation/core/process.md
    """
    rel_path = os.path.relpath(module_path, SRC_DIR)
    md_filename = os.path.splitext(rel_path)[0] + ".md"
    md_filepath = os.path.join(DOCS_DIR, md_filename)

    os.makedirs(os.path.dirname(md_filepath), exist_ok=True)
    with open(md_filepath, "w") as f:
        f.write(f"# `{import_path}`\n\n")
        f.write(f"::: {import_path}\n")
        f.write("    options:\n")
        f.write("        show_source: true\n")
        f.write("        show_inline: true\n")


def generate_docs():
    """Generate Markdown documentation files for all modules in the framework.

    Walks through the source directory, identifies Python modules, and creates
    corresponding Markdown files with MkDocs documentation directives.

    Example:
        >>> generate_docs()
        ✅ Generated 42 Markdown reference files in docs/reference/documentation/
    """
    modules = collect_modules(SRC_DIR)
    for module_path in modules:
        import_path = get_module_path(module_path)
        write_markdown(module_path, import_path)
    print(f"✅ Generated {len(modules)} Markdown reference files in {DOCS_DIR}/")


if __name__ == "__main__":
    generate_docs()
