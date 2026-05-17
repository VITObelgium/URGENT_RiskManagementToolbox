"""CLI entry point for scaffolding a one-file plugin.

Examples
--------
    pixi run create-plugin simulation Eclipse
    pixi run create-plugin algorithm Genetic
    pixi run create-plugin wms CompanyWMS

The generated file lives at ``plugins/<kind_dir>/<snake_case_name>.py`` and
contains stub implementations of every required method declared on the live
interface for the chosen plugin kind.
"""

from __future__ import annotations

import argparse
import sys

from urgent_plugins.scaffolder import (
    find_kind_spec_by_alias,
    get_kind_specs,
    write_plugin,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold a one-file plugin from a live interface."
    )
    parser.add_argument(
        "kind",
        choices=[spec.cli_alias for spec in get_kind_specs()],
        help="Plugin kind alias.",
    )
    parser.add_argument(
        "class_name",
        help="CamelCase class name for the new plugin implementation.",
    )
    parser.add_argument(
        "--plugin-name",
        default=None,
        help=(
            "Override the registered plugin name (defaults to snake_case of "
            "class_name). Useful when migrating an existing implementation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the target file if it already exists.",
    )
    args = parser.parse_args(argv)

    spec = find_kind_spec_by_alias(args.kind)
    try:
        target = write_plugin(
            spec,
            args.class_name,
            plugin_name=args.plugin_name,
            overwrite=args.overwrite,
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Created {spec.kind.value} plugin stub at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
