# Protocol Buffers Setup Guide

This project uses [Protocol Buffers (Protobuf)](https://protobuf.dev/) to define data structures (Turns, Questions, Answers) and configuration schemas. This ensures type safety and consistent data formats across the entire pipeline, from data generation to evaluation.

Before running **any** Python scripts in this repository (whether for CAME-Bench evaluation or the STITCH method), you must generate the Python code from these `.proto` definitions.

## 1. Why do I need this?

- **Data Structures**: The `Turn` and `Question` objects used in our JSONL files are defined in `proto/project_dataset_uniform.proto`.
- **Configurations**: All config files (e.g., for retrieval, answer generation) map directly to Proto messages like `AnswerGenerationConfig` or `RetrievalConfig`.
- **Interoperability**: The generated Python code (`generated_proto/python/*`) is imported by our scripts to read/write these formats.

## 2. Prerequisites

You need two things installed:

1.  **Protocol Buffer Compiler (`protoc`)**: The core binary that compiles `.proto` files.
2.  **Python gRPC Tools**: The Python package that provides the runtime libraries.

### Installing `protoc`

**macOS (Homebrew):**
```bash
brew install protobuf
```

**Linux (apt):**
```bash
sudo apt-get update
sudo apt-get install -y protobuf-compiler
```

**Windows:**
Download the binary from the [GitHub releases page](https://github.com/protocolbuffers/protobuf/releases), extract it, and add the `bin` directory to your system `PATH`.

### Installing Python Dependencies

```bash
pip install grpcio-tools
# OR if you installed requirements.txt, it's already there:
pip install -r requirements.txt
```

## 3. How to Generate

We provide a universal script that works on Windows, macOS, and Linux.

1.  Navigate to the project root.
2.  Run the generation script:

```bash
python3 scripts/generate_proto_universal.py
```

### Verification
If successful, you will see a `generated_proto/python/` directory created in the project root containing files like:
- `project_dataset_uniform_pb2.py`
- `answer_generation_pb2.py`
- ...and others.

## 4. Troubleshooting

**"ModuleNotFoundError: No module named 'generated_proto'..."**
- This means the protos haven't been generated yet. Run the script above.

**"Error: protoc is not installed or not in PATH"**
- You are missing the `protoc` binary. See [Installing protoc](#installing-protoc) above.

**"ImportError: cannot import name '...' from '..._pb2'"**
- If you pulled recent changes, the `.proto` definitions might have changed. Re-run `python3 scripts/generate_proto_universal.py` to regenerate the Python files.
