import time


def retry(times, exceptions=Exception):
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f'Attempt {attempt} out of {times}')
                    print(f'Exception thrown when attempting to run {func}: {e}')
                    time.sleep(min(2**attempt, 30))
                    attempt += 1
            return func(*args, **kwargs)

        return newfn

    return decorator


def flatten_dict(input_dict, parent_key='', sep='.'):
    flattened = {}
    for key, value in input_dict.items():
        new_key = f'{parent_key}{sep}{key}' if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, sep=sep))
        else:
            flattened[new_key] = value
    return flattened


def get_dict_value(input_dict, keys, default=None):
    for key in keys:
        if key in input_dict:
            return input_dict[key]
    return default
