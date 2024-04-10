"""This file is to aid in the switch from secrets/default.json to config/secrets.toml."""
import json
import os
from modelgauge.config import DEFAULT_CONFIG_DIR, DEFAULT_SECRETS

if __name__ == "__main__":
    with open(os.path.join("secrets", "default.json"), "r") as f:
        raw = json.load(f)

    os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
    with open(os.path.join(DEFAULT_CONFIG_DIR, DEFAULT_SECRETS), "w") as f:
        for scope, values in raw.items():
            print(f"[{scope}]", file=f)
            for key, value in values.items():
                print(f'{key} = "{value}"', file=f)
            print("", file=f)  # Blank line between scopes
