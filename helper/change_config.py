#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml


def set_nested_field(data: dict, field_path: list[str], new_value):
    """
    Set a nested field in a dict given a list of keys.
    Returns the old value if it existed, otherwise "<MISSING>".

    Example:
        field_path = ["geometry", "key_points_file_path"]
        data["geometry"]["key_points_file_path"] = new_value
    """
    cur = data
    for key in field_path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            # If intermediate level is missing or not a dict, create a dict
            cur[key] = {}
        cur = cur[key]

    last_key = field_path[-1]
    old_value = cur.get(last_key, "<MISSING>")
    cur[last_key] = new_value
    return old_value


def update_yaml_field_in_dir(
    dir_path: Path,
    field_name: str,
    new_value,
) -> None:
    """
    Update a (possibly nested) field in all YAML files under a directory.

    - For a top-level field (e.g. "scale"), it updates data["scale"].
    - For a nested field (e.g. "geometry.key_points_file_path"),
      it updates data["geometry"]["key_points_file_path"].

    Args:
        dir_path (Path): Directory containing *.yaml / *.yml files.
        field_name (str): The key to update, can be dotted (e.g. "geometry.key_points_file_path").
        new_value: The new value to assign to that key.
    """
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    yaml_files = sorted(
        list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml"))
    )

    if not yaml_files:
        print(f"[WARN] No YAML files found in {dir_path}")
        return

    # Support nested field via dot notation
    field_path = field_name.split(".")

    print(f"[INFO] Found {len(yaml_files)} YAML files in {dir_path}")
    print(f"[INFO] Updating field '{field_name}' to value: {new_value}")

    for ypath in yaml_files:
        print(f"  -> Processing: {ypath}")
        try:
            with ypath.open("r") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"     [ERROR] Failed to read {ypath}: {e}")
            continue

        try:
            old_value = set_nested_field(data, field_path, new_value)
            print(f"     [INFO] {field_name}: {old_value} -> {new_value}")
        except Exception as e:
            print(f"     [ERROR] Failed to update field '{field_name}' in {ypath}: {e}")
            continue

        try:
            with ypath.open("w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        except Exception as e:
            print(f"     [ERROR] Failed to write {ypath}: {e}")
            continue

    print("[INFO] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Batch update a field (e.g. 'scale' or 'geometry.key_points_file_path') in all YAML files in a directory."
    )
    parser.add_argument(
        "dir",
        type=str,
        help="Path to directory containing *.yaml files.",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="scale",
        help="Name of the field to update (default: scale). "
             "Supports dot notation for nested keys, e.g. 'geometry.key_points_file_path'.",
    )
    parser.add_argument(
        "--value",
        type=str,
        required=True,
        help="New value for the field. Will be parsed as float if possible, "
             "otherwise kept as string.",
    )

    args = parser.parse_args()

    dir_path = Path(args.dir).expanduser().resolve()
    field_name = args.field

    # Try to parse the value as float; if it fails, keep as string
    raw_val = args.value
    try:
        new_value = float(raw_val)
    except ValueError:
        new_value = raw_val

    update_yaml_field_in_dir(dir_path, field_name, new_value)


if __name__ == "__main__":
    main()
