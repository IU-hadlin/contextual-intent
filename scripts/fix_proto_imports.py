import os
import re


def fix_imports(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                content = f.read()

            new_content = re.sub(
                r"^import (\w+_pb2) as",
                r"from . import \1 as",
                content,
                flags=re.MULTILINE
            )

            new_content = re.sub(
                r"^import (\w+_pb2)\s*$",
                r"from . import \1",
                new_content,
                flags=re.MULTILINE
            )

            if content != new_content:
                print(f"Fixing imports in {filename}")
                with open(filepath, "w") as f:
                    f.write(new_content)


if __name__ == "__main__":
    fix_imports("came_bench/proto")
