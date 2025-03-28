import json
import os

input_file = os.path.join("WP", "grasp_apple_5.indy7.json")
output_file = os.path.join("WP", "grasp_apple_edit_5.json")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

program = data.get("Program", [])
single_line_program = [json.dumps(cmd, separators=(",", ":")) for cmd in program]

with open(output_file, "w", encoding="utf-8") as f:
    f.write('{\n')
    f.write(f'  "Name": "{data.get("Name", "")}",\n')
    f.write('  "Program": [\n')
    for i, line in enumerate(single_line_program):
        comma = "," if i < len(single_line_program) - 1 else ""
        f.write(f"    {line}{comma}\n")
    f.write('  ]\n')
    f.write('}\n')

print("Transformation complete: Each command is now on a single line.")
