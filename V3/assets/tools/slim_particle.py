import json
import argparse
from pathlib import Path

def slim_json(input_path, output_path, precision=4):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # 1. Round all floating point numbers in the 'tet' and 'vis' sections
    # This is the biggest saver for file size.
    def round_list(lst):
        return [round(x, precision) if isinstance(x, float) else x for x in lst]

    if "tet" in data:
        if "verts" in data["tet"]:
            data["tet"]["verts"] = round_list(data["tet"]["verts"])
    
    if "vis" in data:
        if "verts" in data["vis"]:
            data["vis"]["verts"] = round_list(data["vis"]["verts"])
        if "skinning" in data["vis"]:
            data["vis"]["skinning"] = round_list(data["vis"]["skinning"])

    # 2. Remove unnecessary whitespace/indentation
    # indent=None creates a single-line minified JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

    original_size = Path(input_path).stat().st_size / 1024
    new_size = Path(output_path).stat().st_size / 1024
    
    print(f"Success!")
    print(f"Original Size: {original_size:.2f} KB")
    print(f"Slimmed Size: {new_size:.2f} KB")
    print(f"Reduction: {((original_size - new_size) / original_size) * 100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Heavy JSON file")
    parser.add_argument("output", help="Output slim JSON file")
    parser.add_argument("--prec", type=int, default=4, help="Decimal places (default 4)")
    args = parser.parse_args()
    
    slim_json(args.input, args.output, args.prec)