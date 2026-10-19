"""Write the API's OpenAPI schema to a file (input for openapi-typescript)."""

import json
import sys
import tempfile
from pathlib import Path

from looklab_api.config import Settings
from looklab_api.main import create_app

out = Path(sys.argv[1] if len(sys.argv) > 1 else "openapi.json")
with tempfile.TemporaryDirectory() as d:
    app = create_app(Settings(data_dir=Path(d), worker_enabled=False))
    out.write_text(json.dumps(app.openapi(), indent=1))
print("wrote", out)
