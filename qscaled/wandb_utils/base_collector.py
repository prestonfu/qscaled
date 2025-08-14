import os
import abc
import numpy as np
import pandas as pd

from copy import deepcopy
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Any
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

from qscaled.constants import QSCALED_PATH
from qscaled.utils.state import remove_with_prompt


class BaseCollector(abc.ABC):
    """Base class for efficiently collecting run data using tags from WandB."""
    MISSING_DATA_LABEL = 'missing_wandb_data'
    NAN_DATA_LABEL = 'nan_wandb_data'

    def __init__(
        self,
        wandb_entity: str,
        wandb_project: str,
        wandb_tags: Union[List[str], str] = [],
        use_cached: bool = True,
        parallel: bool = True,
    ):
        """
        Creates a new BaseCollector object.

        Args:
        * `wandb_tags`: List of wandb tags to filter runs by. If empty list, all
          runs are collected from wandb. If `None`, no runs are collected.
        * `use_cached`: If true, loads pre-existing data from memory. Otherwise,
          fetches data from wandb.
        * `parallel`: If true, fetches data from wandb in parallel.

        The data are stored in two dictionaries, `_metadatas` and `_rundatas`.
        * Keys are tuples of hparam values and are shared between te two dictionaries.
        * `_metadatas` maps keys to lists of metadata dictionaries.
        * `_rundatas` maps keys to lists of dataframes.
        """
        import wandb

        self._wandb_entity = wandb_entity
        self._wandb_project = wandb_project
        self._wandb_api = wandb.Api(timeout=120)
        self._path = os.path.join(QSCALED_PATH, 'collector', f'{wandb_entity}:{wandb_project}')
        os.makedirs(self._path, exist_ok=True)
        self._metadatas = defaultdict(list)
        self._rundatas = defaultdict(list)
        self._hparams = []
        self._set_env_step_key()
        self._set_hparams()
        self._set_wandb_metrics()
        if wandb_tags is not None:
            self._fetch_data(wandb_tags, use_cached, parallel)

    @abc.abstractmethod
    def _set_hparams(self):
        """Specifies run hparams and their order."""
        raise NotImplementedError

    @abc.abstractmethod
    def _set_wandb_metrics(self):
        """Specifies all metrics to fetch from Wandb."""
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_key(self, run):
        """
        Generates a key for a run based on its config, in the same order
        specified by `_set_hparams`. To check a run's config, go to
        "Overview" -> "Config" in the Wandb dashboard.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def wandb_fetch(self, run) -> Union[Tuple[Dict[str, Any], pd.DataFrame], None]:
        """
        If the run fails to fetch from the wandb api returns `None`. If the run
        is fetched successfully, returns a tuple of metadata and history.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_zip_export_data(self, metric, logging_freq=None):
        """
        Prepares output for `ZipHandler`. Returns a dictionary mapping
        collector keys to a single `pd.DataFrame`, filtered on the given env
        and utd.

        If `logging_freq is not None`, it will round the step up to the nearest
        multiple of `logging_freq`.
        """
        raise NotImplementedError

    def _set_env_step_key(self):
        self._env_step_key = '_step'

    def keys(self):
        return list(self._rundatas.keys())
    
    def items(self):
        return {key: (self._metadatas[key], self._rundatas[key]) for key in self.keys()}.items()

    def __len__(self):
        """Number of distinct keys."""
        return len(self.keys())

    def size(self):
        """Number of distinct runs."""
        return sum(len(rundatas) for rundatas in self._rundatas.values())

    @property
    def num_hparams(self):
        return len(self._hparams)

    @property
    def hparam_index(self):
        return {hparam: i for i, hparam in enumerate(self._hparams)}

    def _insert_wandb_run(self, run, verbose=False):
        result = self.wandb_fetch(run)
        if result is not None:
            metadata, df = result
            self._metadatas[self._generate_key(run)].append(metadata)
            self._rundatas[self._generate_key(run)].append(df)
        elif verbose:
            print(f'Failed to fetch {run.name} from Wandb')

    def copy_state(self, collector):
        self._metadatas = deepcopy(collector._metadatas)
        self._rundatas = deepcopy(collector._rundatas)
        self._wandb_metrics = deepcopy(collector._wandb_metrics)

    def load_state(self, tag):
        state = np.load(os.path.join(self._path, tag + '.npy'), allow_pickle=True)
        new_metadatas, new_rundatas = state
        assert (
            isinstance(new_metadatas, dict)
            and isinstance(new_rundatas, dict)
            and set(new_metadatas.keys()) == set(new_rundatas.keys())
        )
        self._metadatas.update(new_metadatas)
        self._rundatas.update(new_rundatas)

    def save_state(self, tag):
        """Turns out pickle/gzip is very slow, so we use numpy."""
        state = [self._metadatas, self._rundatas]
        save_path = os.path.join(self._path, tag + '.npy')
        remove_with_prompt(save_path)
        np.save(save_path, state)

    def sample(self):
        """
        Samples a key uniformly at random. Returns the key with its corresponding
        metadata and rundata.

        Useful e.g. if your collector only contains one key.
        """
        keys = self.keys()
        sampled_key = keys[np.random.choice(len(keys))]
        return sampled_key, self._metadatas[sampled_key], self._rundatas[sampled_key]

    def get_unique(self, hparam, filter_str: str = None):
        """
        Find unique keys in hparam subject to constraints (default none).
        See `filter` for more details.
        """
        if hparam not in self.hparam_index:
            raise ValueError(
                f'Invalid hparam: {hparam}. Must be one of {list(self.hparam_index.keys())}.'
            )
        filtered_rundatas = self.filter(filter_str)
        idx = self.hparam_index[hparam]
        unique = set(key[idx] for key in filtered_rundatas.keys())
        return sorted(list(unique))

    def get_metadatas(self):
        return deepcopy(self._metadatas)

    def get_rundatas(self):
        return deepcopy(self._rundatas)

    def _get_filtered_keys(self, filter_str: str = None):
        keys = self.keys()
        if not filter_str:
            return keys
        if not keys:
            df = pd.DataFrame(columns=self._hparams)
        else:
            df = pd.DataFrame(keys, columns=self._hparams)
        
        try:            
            return [tuple(row) for _, row in df.query(filter_str).iterrows()]
        except pd.errors.UndefinedVariableError:
            raise ValueError(
                f'Invalid filter string: {filter_str}. '
                'Make sure all variables are in the form of `hparam==value`.'
            )

    def _get_filtered_datas(self, filter_str: str = None) -> Tuple[Dict, Dict]:
        """Returns two dictionaries: metadata and rundata."""
        filtered_keys = self._get_filtered_keys(filter_str)
        metadatas = {key: self._metadatas[key] for key in filtered_keys}
        rundatas = {key: self._rundatas[key] for key in filtered_keys}
        return metadatas, rundatas

    def get_filtered_metadatas(self, filter_str: str = None):
        """Returns a dictionary of metadata that satisfy the constraints specified by args, kw."""
        return deepcopy(self._get_filtered_datas(filter_str)[0])

    def get_filtered_rundatas(self, filter_str: str = None):
        """Returns a dictionary of rundata that satisfy the constraints specified by args, kw."""
        return deepcopy(self._get_filtered_datas(filter_str)[1])

    def filter(self, filter_str: str):
        """
        Filters the collector by the constraints specified by `filter_str`,
        a thin wrapper around `pd.DataFrame.query`. Returns a collector of the
        same class.

        For example, if the collector has keys `(a, b, c)` and the user calls
        `filter('a>1 and c=="foo"')`, the collector will return all data where
        the key has `a>1` and `c=="foo"`.
        """
        metadatas, rundatas = self._get_filtered_datas(filter_str)
        collector = self.__class__(self._wandb_entity, self._wandb_project, wandb_tags=None)
        collector._metadatas = metadatas
        collector._rundatas = rundatas
        return collector

    def remove_short(self, thresh=1.0):
        """Removes runs with less than `thresh` fraction of the maximum number of steps."""
        for key in self.keys():
            metadatas = self._metadatas[key]
            rundatas = self._rundatas[key]
            max_steps = max(metadata['last_step'] for metadata in metadatas)
            short_runs = [
                i
                for i, metadata in enumerate(metadatas)
                if metadata['last_step'] < max_steps * thresh
            ]
            self._metadatas[key] = [
                metadata for i, metadata in enumerate(metadatas) if i not in short_runs
            ]
            self._rundatas[key] = [
                rundata for i, rundata in enumerate(rundatas) if i not in short_runs
            ]

    def trim(self, num_seeds: int, compare_metric: str, compare_how: str, verbose: bool = False):
        """
        Trims the collector to only the best `num_seeds` keys according to the
        `compare_metric`, ranked by `compare_how` (either `'max'` or `'min'`).
        """
        assert compare_how in ['max', 'min']
        comparator = lambda x: -x[1] if compare_how == 'max' else x[1]
        for key in self.keys():
            metadatas = self._metadatas[key]
            rundatas = self._rundatas[key]
            if len(rundatas) > num_seeds:
                metric_vals = [
                    (i, rundata[compare_metric].mean()) for i, rundata in enumerate(rundatas)
                ]
                metric_vals = sorted(metric_vals, key=comparator)
                idx = [i for i, rundata in metric_vals]
                self._metadatas[key] = [metadatas[idx[j]] for j in range(num_seeds)]
                self._rundatas[key] = [rundatas[idx[j]] for j in range(num_seeds)]
            elif len(rundatas) < num_seeds and verbose:
                print(f'Warning: key {key} contains {len(rundatas)} < {num_seeds} runs')

    def merge(self, collector):
        return self.__class__.merge(self, collector)

    @staticmethod
    def merge(*collectors):
        """
        Merge multiple BaseCollector-inherited objects of the same class.

        Merging collectors from different projects or entities is possible,
        but the resulting collector will have `None` for entity and
        project. Thus, `_fetch_data` will not work for the merged collector.
        """
        assert len(collectors) > 0
        shared_cls = collectors[0].__class__
        assert all(collector.__class__ == shared_cls for collector in collectors), (
            'All collectors must be of the same class.'
        )

        collector = deepcopy(collectors[0])
        for other in collectors[1:]:
            if (
                other._wandb_entity != collector._wandb_entity
                or other._wandb_project != collector._wandb_project
            ):
                collector._wandb_entity = None
                collector._wandb_project = None
            for key in set(collector.keys()) | set(other.keys()):
                collector._metadatas[key].extend(other._metadatas[key])
                collector._rundatas[key].extend(other._rundatas[key])
        return collector

    def _fetch_data(
        self,
        wandb_tags: Union[List[str], str] = [],
        use_cached: bool = True,
        parallel: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
        * `wandb_tags`: List of wandb tags to filter runs by. If empty, all runs
          are collected from wandb.
        * `use_cached`: If true, loads pre-existing data from memory. Otherwise, fetches
          data from wandb.
        * `parallel`: If true, fetches data from wandb in parallel with half the
          number of cores.
        """
        UNSPECIFIED_TAG_NAME = 'QSCALED_ALL_RUNS'
        if not wandb_tags:
            wandb_tags = [UNSPECIFIED_TAG_NAME]
        elif isinstance(wandb_tags, str):
            wandb_tags = [wandb_tags]

        collector_factory = lambda: self.__class__(
            self._wandb_entity, self._wandb_project, wandb_tags=None
        )
        collectors = []
        
        num_threads = min(int(os.cpu_count() * 0.5), 10)  # Wandb connection pool size
        if num_threads < 1:
            parallel = False

        for tag in wandb_tags:
            collector = collector_factory()

            if use_cached and os.path.exists(os.path.join(self._path, tag + '.npy')):
                collector.load_state(tag)
            else:
                wandb_str = f'{self._wandb_entity}/{self._wandb_project}'
                if tag is UNSPECIFIED_TAG_NAME:
                    tag_filter = {}
                    tqdm_desc = f'{wandb_str}: all runs'
                else:
                    tag_filter = {'tags': {'$in': [tag]}}
                    tqdm_desc = f'{wandb_str}: {tag}'

                runs = self._wandb_api.runs(
                    f'{collector._wandb_entity}/{collector._wandb_project}', filters=tag_filter
                )
                runs = [r for r in tqdm(runs, desc=f'{tqdm_desc}: fetching')]
                if len(runs) == 0:
                    print(f'No runs found for {tqdm_desc}')
                    continue
                insert_verbose = lambda run: collector._insert_wandb_run(run, verbose)
                if parallel:
                    with ThreadPool() as pool:
                        list(tqdm(pool.imap(insert_verbose, runs), total=len(runs), desc=f'{tqdm_desc}: processing'))
                else:
                    for run in tqdm(runs, total=len(runs), desc=f'{tqdm_desc}: processing'):
                        insert_verbose(run)

                collector.save_state(tag)

            collectors.append(collector)

        combined_collector = self.__class__.merge(*collectors)
        self.copy_state(combined_collector)

    def _resolve_logging_freq(self, df: pd.DataFrame, logging_freq: int):
        """
        Resolves non-uniform logging frequencies. Rounds step up to nearest
        multiple of `logging_freq`, then averages over potential duplicates.
        Since performance is usually increasing, in general the output
        will be more conservative than the input.

        Useful in cases where merging data from multiple runs with different
        logging frequencies.
        """
        df = df.copy()
        df['rounded_step'] = np.ceil(df[self._env_step_key] / logging_freq) * logging_freq
        agg_dict = {
            self._env_step_key: 'first',
            **{col: 'mean' for col in df.columns if col.startswith('seed')},
        }
        return df.groupby('rounded_step').agg(agg_dict).dropna().reset_index(drop=True)
