import os

SRC_DIR = "neurotrace"
DOCS_DIR = "docs/reference/documentation"
IMPORT_PREFIX = "neurotrace"


def get_module_path(file_path):
    rel_path = os.path.relpath(file_path, SRC_DIR)
    no_ext = os.path.splitext(rel_path)[0]
    return f"{IMPORT_PREFIX}." + ".".join(no_ext.split(os.sep))


def collect_modules(base_dir):
    modules = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".py") and not f.startswith("__init__"):
                full_path = os.path.join(root, f)
                modules.append(full_path)
    return modules


def write_markdown(module_path, import_path):
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
    modules = collect_modules(SRC_DIR)
    for module_path in modules:
        import_path = get_module_path(module_path)
        write_markdown(module_path, import_path)
    print(f"✅ Generated {len(modules)} Markdown reference files in {DOCS_DIR}/")


if __name__ == "__main__":
    generate_docs()
