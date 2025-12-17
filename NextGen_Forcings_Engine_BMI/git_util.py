import json


def transform_component(component_git_info):
    """
    Transform a single component dictionary to include only selected Git fields in a specific order:
      - Always include 'release', 'build_date', and 'commit_hash' (in that order).
      - If 'tags' is empty, also include 'commit_date', 'author', and 'message' (in that order) if they exist.

    :param component_git_info: A dictionary containing Git information for a component.
    :return: A new dictionary with only the desired fields.
    """
    new_comp = {}

    if component_git_info.get("tags", "").strip() == "":
        # If tags is empty, include branch, author, message, and commit_date.
        branch = f"dev ({component_git_info.get('branch', '<unknown>')})"
        new_comp["release"] = branch
    else:
        new_comp["release"] = component_git_info.get("tags", "")

    # Insert keys in the desired order: build_date, then commit_hash.
    new_comp["build_date"] = component_git_info.get("build_date", "")
    new_comp["commit_hash"] = component_git_info.get("commit_hash", "")

    # If tags is empty, add commit_date, author, and message in order, if they exist.
    if component_git_info.get("tags", "").strip() == "":
        if "commit_date" in component_git_info:
            new_comp["commit_date"] = component_git_info.get("commit_date", "")
        if "author" in component_git_info:
            new_comp["author"] = component_git_info.get("author", "")
        if "message" in component_git_info:
            new_comp["message"] = component_git_info.get("message", "")

    return new_comp


def recursive_print(d: dict, indent: int = 0) -> None:
    """
    Recursively print all key/value pairs from a dictionary.

    For each key-value pair:
      - If the value is a dictionary, print the key on one line and then recurse into that dictionary.
      - If the value is a list, print the key on one line and then iterate through the list;
        for each element that is a dictionary, recurse into it; otherwise print the element on a separate line.
      - Otherwise (if the value is a string or other non-dict, non-list), print the key and value on one line.

    :param d: The dictionary to print.
    :param indent: The current indentation level (number of spaces).
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            recursive_print(value, indent + 2)
        elif isinstance(value, list):
            print(" " * indent + f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    recursive_print(item, indent + 2)
                else:
                    print(" " * (indent + 2) + str(item))
        else:
            print(" " * indent + f"{key}: {value}")


def print_git_info(git_info_file: str):
    """
    Read the specified git_info JSON file, transform its contents, and log all key/value pairs recursively.

    The output will print top-level keys.

    :param git_info_file: Path to the JSON file containing Git information.
    """
    try:
        with open(git_info_file, 'r') as f:
            git_info = json.load(f)
    except FileNotFoundError:
        print(f'{git_info_file} not found')
        return
    except json.decoder.JSONDecodeError as e:
        print(f"Error reading {git_info_file}: {e}")
        return

    if not git_info:
        print(f"Failed to retrieve git information from {git_info_file}.")
        return

    # Transform each top-level component without removing the keys.
    transformed_git_info = {key: transform_component(value) for key, value in git_info.items()}

    recursive_print(transformed_git_info)


def print_git_info_all():
    """
    Convenience function to print Git information from multiple JSON files.
    """
    print_git_info('/ngen-app/ngen-bmi-forcing_git_info.json')
    print()
