import os
import subprocess
from pathlib import Path
from typing import Sequence


def find_proto_file_names(
    directory: str | Path, full_path: bool = False
) -> list[str | Path]:
    """
    Find all `.proto` file names in a directory and its subdirectories.

    Args:
        directory (str | Path): The directory to search in.
        full_path (bool): Whether to return full file paths. Defaults to False.

    Returns:
        list[str | Path]: A list of `.proto` file names or paths.
    """
    proto_files = []
    directory = os.fspath(directory)  # Convert Path to str if needed
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".proto"):
                proto_files.append(os.path.join(root, file) if full_path else file)
    return sorted(proto_files)


def compile_protos(proto_dir: str | Path, python_out: str) -> None:
    """
    Compile `.proto` files using `protol` and `protoc`.

    Args:
        proto_dir (str | Path): Directory containing `.proto` files.
        python_out (str): Output directory for Python code.
    """
    proto_files = [str(p) for p in find_proto_file_names(proto_dir, True)]
    os.makedirs(python_out, exist_ok=True)

    if not proto_files:
        print("No .proto files found.")
        return

    print("Compiling protos file")
    protoc_cmd = [
        "python",
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={python_out}",
        f"--mypy_out={python_out}",
        f"--grpc_python_out={python_out}",
        *proto_files,
    ]
    process = subprocess.run(protoc_cmd)
    print(f"Command executed: {' '.join(protoc_cmd)}")
    if process.returncode != 0:
        print("Error compiling proto")

    print("Formatting with protol...")
    protol_cmd: Sequence[str]
    protol_cmd = [
        "protol",
        "--create-package",
        "--in-place",
        f"--python-out={python_out}",
        "protoc",
        f"--proto-path={proto_dir}",
        *proto_files,
        "--protoc-path",
        "python3 -m grpc_tools.protoc",
    ]

    process = subprocess.run(protol_cmd)
    print(f"Command executed: {' '.join(protol_cmd)}")
    if process.returncode != 0:
        print("Error formatting .proto files.")


def main() -> None:
    compile_protos(proto_dir="proto", python_out="./generated")


if __name__ == "__main__":
    main()
