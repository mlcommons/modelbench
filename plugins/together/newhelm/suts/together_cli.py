from newhelm.command_line import (
    SECRETS_FILE_OPTION,
    display_header,
    display_list_item,
    newhelm_cli,
)
from newhelm.general import get_or_create_json_file
from newhelm.secrets_registry import SECRETS
import together  # type: ignore
from collections import defaultdict


@newhelm_cli.command()
@SECRETS_FILE_OPTION
def together_list(secrets: str):
    """List all models available in together.ai."""
    SECRETS.set_values(get_or_create_json_file(secrets))

    together.api_key = SECRETS.get_required("together", "api_key")
    model_list = together.Models.list()

    # Group by display_type, which seems to be the model's style.
    by_display_type = defaultdict(list)
    for model in model_list:
        try:
            display_type = model["display_type"]
        except KeyError:
            display_type = "unknown"
        display_name = model["display_name"]
        by_display_type[display_type].append(f"{display_name}: {model['name']}")

    for display_name, models in by_display_type.items():
        display_header(f"{display_name}: {len(models)}")
        for model in sorted(models):
            display_list_item(model)
    display_header(f"Total: {len(model_list)}")
