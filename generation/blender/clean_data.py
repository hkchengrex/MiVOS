import sys
import os
from os import path
import shutil

"""
Look at a "rendered" folder, move the rendered to the output path
Keep the empty folders in place (don't delete since it might still be rendered right)
Also copy the corresponding yaml in there
"""

input_path = sys.argv[1]
output_path = sys.argv[2]
yaml_path = sys.argv[3]

# This overrides the softlink
# os.makedirs(output_path, exist_ok=True)

renders = os.listdir(input_path)
is_rendered = [len(os.listdir(path.join(input_path, r, 'segmentation')))==160 for r in renders]

updated = 0
for i, r in enumerate(renders):
    if is_rendered[i]:
        if not path.exists(path.join(output_path, r)):
            shutil.move(path.join(input_path, r), output_path)
            prefix = r[:3]
        
            shutil.copy2(path.join(yaml_path, 'yaml_%s'%prefix, '%s.yaml'%r), path.join(output_path, r))
            updated += 1
        else:
            print('path exist')
    else:
        # Nothing for now. Can do something later
        pass

print('Number of completed renders: ', len(os.listdir(output_path)))
print('Number of updated renders: ', updated)