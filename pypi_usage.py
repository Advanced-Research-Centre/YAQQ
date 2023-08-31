# Step 1
"""
pip install --upgrade twine
pip install --upgrade build
"""

# Step 2
"""
edit code in yaqq/src/yaqq.py
in yaqq/setup.py, change version number
from yaqq folder, run: 
    python -m build
    twine upload dist/*
"""

# Step 3
"""
https://pypi.org/project/yaqq/
pip install --upgrade yaqq
"""

# Step 4
import yaqq
yaqq.run()

# Step 5
"""
from YAQQ folder, run:
    python pypi_usage.py
"""