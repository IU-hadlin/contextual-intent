#!/usr/bin/env python3
"""
Universal protocol buffer generation script that works across Windows, macOS, and Linux.
This script detects the operating system and runs the appropriate commands.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


def check_command_exists(command):
    """Check if a command exists in the system PATH."""
    return shutil.which(command) is not None


def check_python_package(package_name):
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def print_error(message):
    """Print an error message in red."""
    print(f"❌ Error: {message}")


def print_success(message):
    """Print a success message in green."""
    print(f"✅ {message}")


def print_info(message):
    """Print an info message."""
    print(f"ℹ️  {message}")


def check_dependencies():
    """Check if all required dependencies are installed."""
    missing_deps = []

    # Check protoc
    if not check_command_exists("protoc"):
        missing_deps.append("protoc (Protocol Buffer Compiler)")

    # Check grpc_tools
    if not check_python_package("grpc_tools"):
        missing_deps.append("grpcio-tools (Python gRPC tools)")

    # Optional: Yarn and Node for frontend TS generation
    # We don't fail the whole script if they're missing; we just skip TS generation later
    if not check_command_exists("yarn"):
        print_info("Yarn not found; frontend TypeScript generation will be skipped.")
    if not check_command_exists("node"):
        print_info("Node.js not found; frontend TypeScript generation will be skipped.")

    if missing_deps:
        print_error("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease refer to README_SETUP.md for installation instructions.")
        return False

    return True


def run_command(command, description):
    """Run a command and handle errors."""
    print_info(f"{description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to {description.lower()}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def main():
    """Main function to generate protocol buffer files."""
    print("🚀 Starting protocol buffer generation...")

    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    print_info(f"Working directory: {project_root}")
    print_info(f"Operating System: {platform.system()} {platform.release()}")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Define directories
    proto_dir = "proto"
    gen_py = "generated_proto/python"

    # Create directories
    Path(gen_py).mkdir(parents=True, exist_ok=True)

    # Generate Python files
    python_cmd = f"python -m grpc_tools.protoc -I{proto_dir} --python_out={gen_py} --grpc_python_out={gen_py} {proto_dir}/*.proto"
    if not run_command(python_cmd, "Generate Python protobuf files"):
        sys.exit(1)

    print_success("Python proto files generated successfully!")

    # List generated files
    print_info("Generated files:")
    for root, dirs, files in os.walk("generated_proto"):
        for file in files:
            if file.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, file))
                print(f"  - {rel_path}")

    # Optionally generate TypeScript types for the frontend using ts-proto
    frontend_dir = project_root / "result_visualizer"
    if frontend_dir.exists():
        # Allow skipping TS generation explicitly
        if os.environ.get("SKIP_TS_PROTO", "").lower() in {"1", "true", "yes"}:
            print_info("SKIP_TS_PROTO set; skipping TypeScript proto generation.")
            return

        print_info("Detected frontend at result_visualizer; preparing to generate TypeScript types with ts-proto")
        ts_out_dir = frontend_dir / "src" / "proto"
        ts_out_dir.mkdir(parents=True, exist_ok=True)

        # Resolve plugin path without invoking yarn (which can hang in some environments)
        plugin_path = frontend_dir / "node_modules" / ".bin" / "protoc-gen-ts_proto"
        if not plugin_path.exists():
            print_info(
                "ts-proto plugin not found at node_modules/.bin. "
                "Skipping TS generation. Run 'yarn --cwd result_visualizer install' to install dependencies, "
                "then re-run this script."
            )
            return

        # ts-proto options: snake_case field names preserved to match stored JSON; no services
        ts_proto_opts = [
            "useProtoFieldName=true",
            "outputServices=none",
            "useOptionals=messages",
            "esModuleInterop=true",
            "exportCommonSymbols=false",
        ]
        opts = ",".join(ts_proto_opts)
        ts_cmd = (
            f"protoc -I{proto_dir} "
            f"--plugin=protoc-gen-ts_proto={plugin_path} "
            f"--ts_proto_out={ts_out_dir} --ts_proto_opt={opts} {proto_dir}/*.proto"
        )
        if run_command(ts_cmd, "Generate TypeScript protobuf files for frontend"):
            print_success(f"TypeScript proto files generated at {ts_out_dir}")
        else:
            print_error("Failed to generate TypeScript proto files")


if __name__ == "__main__":
    main()
