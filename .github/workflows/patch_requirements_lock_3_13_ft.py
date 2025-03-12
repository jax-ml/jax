import os
import shutil
from pathlib import Path


jax_repo_path = Path(__file__).absolute().parent.parent.parent
assert jax_repo_path.exists()

gh_ws = os.environ.get("GITHUB_WORKSPACE")
assert gh_ws is not None

req_filepath = jax_repo_path / "build" / "requirements_lock_3_13_ft.txt"
assert req_filepath.exists(), str(req_filepath)
copy_req_filepath = jax_repo_path / "build" / "requirements_lock_3_13_ft.txt.copy"

# make a copy
shutil.copy(req_filepath, copy_req_filepath)

with copy_req_filepath.open("r") as src, req_filepath.open("w") as dst:
    line_number = -1
    skip_line = False
    while True:
        src_line = src.readline()
        line_number += 1

        if len(src_line) == 0:
            break

        # Insert
        # --pre
        # --extra-index-url file://${GITHUB_WORKSPACE}/wheelhouse/
        # numpy
        # just after the header comments
        if line_number == 6:
            dst.write("\n--pre\n")
            dst.write(f"--extra-index-url file://{gh_ws}/wheelhouse/\n")
            dst.write("numpy\n\n")

        # Remove numpy dependency
        if src_line.startswith("numpy=="):
            skip_line = True
            continue

        # skip lines after "numpy==" like "    --hash=" or "    # " just before the next dependency
        if skip_line and (src_line.startswith("    --hash=") or src_line.startswith("    # ")):
            continue
        else:
            # Stop skipping when src_line has the next dependency
            skip_line = False

        dst.write(src_line)
