# Run in ipython notebook
# %%javascript
# IPython.OutputArea.prototype._should_scroll = function(lines) {
# return false;
# }
import os

from IPython.core.display import display
from wand.image import Image as WImage

base_dir = "../plots/"
all_files = os.listdir(base_dir)
relevant_files = list(filter(lambda x: x.startswith("Variance"), all_files))
print(relevant_files)
for i in relevant_files:
    img = WImage(filename=f"{base_dir}{i}", resolution=100)  # bigger
display(img)
