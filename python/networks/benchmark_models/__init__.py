import os
import importlib

# Get the directory of the current file
package_dir = os.path.dirname(__file__)

# Loop through all files in the directory
for module in os.listdir(package_dir):
    # Check if the file is a Python file and not __init__.py
    if module.endswith('.py') and module != '__init__.py':
        # Import the module
        importlib.import_module(f'{__name__}.{module[:-3]}')