import pandas as pd
from pathlib import Path

# Leer el último log
log_dir = Path("logs/enhancement")
latest_log = max(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime)

with open(latest_log) as f:
    log_content = f.read()
print(log_content)