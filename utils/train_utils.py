import constants as c
import json

def get_default_params():
    with open(c.DEFAULT_PARAMS_DIR) as f:
        return json.load(f)
    