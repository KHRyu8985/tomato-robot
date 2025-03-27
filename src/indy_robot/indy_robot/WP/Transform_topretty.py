import json
import os

input_file = os.path.join("WP", "grasp_apple_edit_5.json")
output_file = os.path.join("WP", "grasp_apple_5.json")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Transformation complete")
