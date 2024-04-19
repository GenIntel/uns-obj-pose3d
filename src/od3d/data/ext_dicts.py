

def unroll_nested_dict(nested_dict, parent_key='', separator='/'):
    items = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(unroll_nested_dict(value, new_key, separator))
        else:
            items[new_key] = value
    return items

def rollup_flattened_dict(flattened_dict, separator='/'):
    nested_dict = {}
    for key, value in flattened_dict.items():
        keys = key.split(separator)
        current_dict = nested_dict
        for k in keys[:-1]:
            current_dict = current_dict.setdefault(k, {})
        current_dict[keys[-1]] = value
    return nested_dict