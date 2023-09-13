# Step 1: install packages for creating and uploading package to pypi
"""
pip install --upgrade twine
pip install --upgrade build
"""

# Step 2: create and upload package to pypi
"""
edit code in yaqq/src/yaqq.py
in yaqq/setup.py, change version number
from yaqq folder, run: 
    python -m build
    twine upload dist/*
use token and key
"""

# Step 3: install YAQQ package from pypi
"""
https://pypi.org/project/yaqq/
pip install --upgrade yaqq
"""

# Step 4: run YAQQ package from pypi
import yaqq
yaqq.run()

# Step 5: execute code
"""
from YAQQ folder, run:
    python pypi_usage.py
"""