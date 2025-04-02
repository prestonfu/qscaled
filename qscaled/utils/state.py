import os
import shutil
from qscaled.constants import suppress_overwrite_prompt


def remove_with_prompt(*paths):
    """Removes files or directories with a prompt if they already exist."""
    existing_paths = [path for path in paths if os.path.exists(path)]

    if existing_paths:
        if suppress_overwrite_prompt:
            should_remove = True
        else:
            description = '\n'.join(['  ' + path for path in existing_paths])
            if len(existing_paths) > 1:
                response = input(
                    f'Paths:\n{description}\nalready exist. Remove and overwrite? (y/N): '
                ).lower()
            else:
                response = input(
                    f'Path:\n{description}\nalready exists. Remove and overwrite? (y/N): '
                ).lower()
            should_remove = response in ('y', 'yes')

        if should_remove:
            for path in existing_paths:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        else:
            print('Cancelling operation.')
            exit(0)
