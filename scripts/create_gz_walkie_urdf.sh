#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Script: create_gz_walkie_urdf.sh
# Purpose:
#   1. Generate gz_walkie.urdf from gz_walkie.urdf.xacro
#   2. Replace package://walkie_description with file:// absolute path
#   3. Save as gz_walkie_absolute_path.urdf
# -----------------------------------------------------------------------------

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If script is in scripts/ directory, package root is one level up;
# if script is in package root, package root is SCRIPT_DIR.
if [ -f "${SCRIPT_DIR}/package.xml" ]; then
    PACKAGE_DIR="${SCRIPT_DIR}"
else
    PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

ROBOTS_DIR="${PACKAGE_DIR}/robots"
XACRO_INPUT="${ROBOTS_DIR}/gz_walkie.urdf.xacro"
URDF_OUTPUT="${ROBOTS_DIR}/gz_walkie.urdf"
ABS_URDF_OUTPUT="${ROBOTS_DIR}/gz_walkie_absolute_path.urdf"

echo "=== Walkie URDF Generator ==="
echo "Package directory : ${PACKAGE_DIR}"
echo "Input xacro       : ${XACRO_INPUT}"
echo "Output URDF       : ${URDF_OUTPUT}"
echo "Output Abs URDF   : ${ABS_URDF_OUTPUT}"
echo "============================="

# 1. Check for xacro command (use system xacro or fallback to pixi)
XACRO_CMD=""
if command -v xacro >/dev/null 2>&1; then
    XACRO_CMD="xacro"
elif command -v pixi >/dev/null 2>&1; then
    XACRO_CMD="pixi run xacro"
else
    echo "Error: Neither 'xacro' nor 'pixi' could be found in PATH." >&2
    echo "Please activate your ROS 2 environment or run 'pixi shell'." >&2
    exit 1
fi

echo "[1/3] Generating ${URDF_OUTPUT} from Xacro..."
${XACRO_CMD} "${XACRO_INPUT}" -o "${URDF_OUTPUT}"
echo "  -> Created ${URDF_OUTPUT}"

echo "[2/3] Replacing package:// references with file://${PACKAGE_DIR}..."
echo "[3/3] Saving absolute path URDF to ${ABS_URDF_OUTPUT}..."

python3 - <<EOF
import os
import re

package_dir = "${PACKAGE_DIR}"
input_urdf = "${URDF_OUTPUT}"
output_urdf = "${ABS_URDF_OUTPUT}"

with open(input_urdf, "r", encoding="utf-8") as f:
    content = f.read()

# Replace package URI with file:// absolute path
# Target format: file:///path/to/robot-description/...
replacements = [
    ("package://walkie_description", f"file://{package_dir}"),
    ("package://walkie-description", f"file://{package_dir}"),
]

for old_pkg, new_pkg in replacements:
    content = content.replace(old_pkg, new_pkg)

# Write output file with standard spelling
with open(output_urdf, "w", encoding="utf-8") as f:
    f.write(content)

# Count and verify replaced paths
file_refs = re.findall(r'file://(/[^"\' <]+)', content)
missing = [path for path in file_refs if not os.path.exists(path)]

print(f"  -> Total mesh references converted: {len(file_refs)}")
if missing:
    print(f"  [WARNING] {len(missing)} referenced files not found on disk:")
    for m in missing[:5]:
        print(f"     - {m}")
else:
    print("  -> All referenced mesh files successfully verified on disk!")

print(f"  -> Successfully generated: {output_urdf}")
EOF

echo "All steps completed successfully!"
