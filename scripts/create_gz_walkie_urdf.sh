#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Script: create_gz_walkie_urdf.sh
# Purpose:
#   Generate simple wheel and roller versions of Walkie URDF:
#   1. gz_walkie_simple.urdf & gz_walkie_simple_absolute_path.urdf (simple wheel)
#   2. gz_walkie_roller.urdf & gz_walkie_roller_absolute_path.urdf (roller tags with collision)
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/package.xml" ]; then
    PACKAGE_DIR="${SCRIPT_DIR}"
else
    PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

ROBOTS_DIR="${PACKAGE_DIR}/robots"
XACRO_INPUT="${ROBOTS_DIR}/gz_walkie.urdf.xacro"

# Mode: all (default), simple, roller
MODE="${1:-all}"
# Lift resting position: true (top), false (bottom, default)
LIFT_AT_TOP="${2:-false}"

echo "=== Walkie URDF Generator ==="
echo "Package directory : ${PACKAGE_DIR}"
echo "Input xacro       : ${XACRO_INPUT}"
echo "Target mode       : ${MODE}"
echo "Lift at top       : ${LIFT_AT_TOP}"
echo "============================="

# 1. Check for xacro command
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

generate_variant() {
    local variant_name="$1"
    local roller_val="$2"
    local urdf_file="${ROBOTS_DIR}/gz_walkie_${variant_name}.urdf"
    local abs_urdf_file="${ROBOTS_DIR}/gz_walkie_${variant_name}_absolute_path.urdf"

    echo "--- Generating [${variant_name}] (roller:=${roller_val}, lift_at_top:=${LIFT_AT_TOP}) ---"
    ${XACRO_CMD} "${XACRO_INPUT}" "roller:=${roller_val}" "lift_at_top:=${LIFT_AT_TOP}" -o "${urdf_file}"
    echo "  -> Created ${urdf_file}"

    # Replace package:// with file:// absolute path
    python3 - <<EOF
import os
import re

package_dir = "${PACKAGE_DIR}"
input_file = "${urdf_file}"
output_file = "${abs_urdf_file}"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

replacements = [
    ("package://walkie_description", f"file://{package_dir}"),
    ("package://walkie-description", f"file://{package_dir}"),
]
for old_pkg, new_pkg in replacements:
    content = content.replace(old_pkg, new_pkg)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(content)

file_refs = re.findall(r'file://(/[^"\' <]+)', content)
missing = [path for path in file_refs if not os.path.exists(path)]
if missing:
    print(f"  [WARNING] {len(missing)} referenced files not found on disk:")
    for m in missing[:3]:
        print(f"     - {m}")
else:
    print(f"  -> Converted {len(file_refs)} mesh paths to absolute file:// URIs (all verified on disk)")

print(f"  -> Created ${abs_urdf_file}")
EOF
}

case "${MODE}" in
    simple)
        generate_variant "simple" "false"
        ;;
    roller)
        generate_variant "roller" "true"
        ;;
    all)
        generate_variant "simple" "false"
        generate_variant "roller" "true"
        ;;
    *)
        echo "Unknown mode: ${MODE}. Available options: all, simple, roller" >&2
        exit 1
        ;;
esac

echo ""
echo "Summary of generated URDF files:"
ls -lh "${ROBOTS_DIR}"/gz_walkie*.urdf
echo "All steps completed successfully!"
