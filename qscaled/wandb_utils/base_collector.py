import os
import abc
import numpy as np
import wandb
import pandas as pd

from copy import deepcopy
from functools import reduce
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

from qscaled.constants import QSCALED_PATH

np.random.seed(42)

api = wandb.Api(timeout=120)


class BaseCollector(abc.ABC):
    """Base class for efficiently collecting run data using tags from WandB."""

    DUMMY_VALUE = 'Wandb collector dummy value'

    def __init__(
        self,
        wandb_entity: str,
        wandb_project: str,
        wandb_tags: List[str] | str = [],
        use_cache: bool = True,
        parallel: bool = True,
    ):
        """
        Creates a new BaseCollector object.

        Args:
        * `wandb_tags`: List of wandb tags to filter runs by. If empty list, all
          runs are collected from wandb. If `None`, no runs are collected.
        * `use_cache`: If true, loads pre-existing data from memory. Otherwise,
          fetches data from wandb.
        * `parallel`: If true, fetches data from wandb in parallel.

        The data are stored in two dictionaries, `_metadatas` and `_rundatas`.
        * Keys are tuples of hparam values and are shared between te two dictionaries.
        * `_metadatas` maps keys to lists of metadata dictionaries.
        * `_rundatas` maps keys to lists of dataframes.
        """
        self._wandb_entity = wandb_entity
        self._wandb_project = wandb_project
        self._path = os.path.join(QSCALED_PATH, 'collector', f'{wandb_entity}:{wandb_project}')
        os.makedirs(self._path, exist_ok=True)
        self._metadatas = defaultdict(list)
        self._rundatas = defaultdict(list)
        self._hparams = []
        self._set_hparams()
        self._set_wandb_metrics()
        if wandb_tags is not None:
            self._fetch_data(wandb_tags, use_cache, parallel)

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
    def wandb_fetch(self, run, num_tries=5) -> Tuple[Dict[str, Any], pd.DataFrame] | None:
        """
        If the run fails to fetch from the wandb api after `num_tries` attempts,
        returns `None`. If the run is fetched successfully, returns a tuple of
        metadata and history.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_zip_export_data(self, metric, logging_freq=None) -> Dict[Any, np.typing.NDArray]:
        """
        Prepares output for `ZipHandler`. Returns a dictionary mapping
        collector keys to a single `pd.DataFrame`, filtered on the given env
        and utd.

        If `logging_freq is not None`, it will round the step up to the nearest
        multiple of `logging_freq`.
        """
        raise NotImplementedError

    def keys(self):
        return list(self._rundatas.keys())

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
        np.save(os.path.join(self._path, tag + '.npy'), state)

    def sample(self):
        """
        Samples a key uniformly at random. Returns the key with its corresponding
        metadata and rundata.
        """
        sampled_key = self.keys[np.random.choice(len(self.keys))]
        return sampled_key, self._metadatas[sampled_key], self._rundatas[sampled_key]

    def get_unique(self, hparam, *args, **kw):
        """
        Find unique keys in hparam subject to constraints (default none).
        See `filter` for more details.
        """
        if hparam not in self._hparam_index:
            raise ValueError(f'Invalid hparam: {hparam}. Must be one of {list(self._hparam_index.keys())}.')
        filtered_rundatas = self.filter(*args, **kw)
        idx = self._hparam_index[hparam]
        unique = set(key[idx] for key in filtered_rundatas.keys())
        return sorted(list(unique))

    def get_metadatas(self):
        return deepcopy(self._metadatas)

    def get_rundatas(self):
        return deepcopy(self._rundatas)

    def _dummy_key_factory(self, *args, **kw):
        """Create a dummy key with DUMMY_VALUE for unspecified hparams."""
        params = [BaseCollector.DUMMY_VALUE] * self.num_hparams
        for i, value in enumerate(args):
            params[i] = value
        for hparam, value in kw.items():
            assert hparam in self._hparams and self._hparam_index[hparam] >= len(args)
            params[self._hparam_index[hparam]] = value
        return tuple(params)

    def _check_key(self, key: Tuple, *args, **kw):
        """Check whether check_key satisfies the constraints specified by args, kw."""
        template_key = self._dummy_key_factory(*args, **kw)
        for check, template in zip(key, template_key):
            if template not in (BaseCollector.DUMMY_VALUE, check):
                return False
        return True

    def _get_filtered_keys(self, *args, **kw):
        """Returns a list of keys that satisfy the constraints specified by args, kw."""
        return [key for key in self.keys if self._check_key(key, *args, **kw)]

    def _get_filtered_datas(self, *args, **kw) -> Tuple[Dict, Dict]:
        """Returns two dictionaries: metadata and rundata."""
        filtered_keys = self._get_filtered_keys(*args, **kw)
        metadatas = {key: self._metadatas[key] for key in filtered_keys}
        rundatas = {key: self._rundatas[key] for key in filtered_keys}
        return metadatas, rundatas

    def get_filtered_metadatas(self, *args, **kw):
        """Returns a dictionary of metadata that satisfy the constraints specified by args, kw."""
        return self._get_filtered_datas(*args, **kw)[0]

    def get_filtered_rundatas(self, *args, **kw):
        """Returns a dictionary of rundata that satisfy the constraints specified by args, kw."""
        return self._get_filtered_datas(*args, **kw)[1]

    def filter(self, *args, **kw):
        """
        Filters the collector by the constraints specified by args and kw.
        Returns a collector of the same class.

        For example, if the collector has keys `(a, b, c)` and the user calls
        `filter('foo', c='bar')`, the collector will return all data where
        the key has `a='foo'` and `c='bar'`.
        """
        metadatas, rundatas = self._get_filtered_datas(*args, **kw)
        collector = self.__class__(self.entity, self.project, wandb_tags=None)
        collector._metadatas = metadatas
        collector._rundatas = rundatas
        return collector

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
                metric_vals = [(i, rundata[compare_metric].mean()) for i, rundata in enumerate(rundatas)]
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

        Warning: Merging collectors from different projects or entities is
        possible, but the resulting collector will have None for entity and
        project. Thus, `_insert` will not work for the merged collector.
        """
        assert len(collectors) > 0
        shared_cls = collectors[0].__class__
        assert all(collector.__class__ == shared_cls for collector in collectors), (
            'All collectors must be of the same class.'
        )

        collector = deepcopy(collectors[0])
        for other in collectors[1:]:
            if other._wandb_entity != collector._wandb_entity or other._wandb_project != collector._wandb_project:
                collector._wandb_entity = None
                collector._wandb_project = None
            for key in collector.keys():
                collector._metadatas[key].extend(other._metadatas[key])
                collector._rundatas[key].extend(other._rundatas[key])
        return collector

    def _fetch_data(
        self, wandb_tags: List[str] | str = [], use_cache: bool = True, parallel: bool = True, verbose: bool = False
    ):
        """
        Args:
        * `wandb_tags`: List of wandb tags to filter runs by. If empty, all runs
          are collected from wandb.
        * `use_cache`: If true, loads pre-existing data from memory. Otherwise, fetches
          data from wandb.
        * `parallel`: If true, fetches data from wandb in parallel with half the
          number of cores.
        """
        UNSPECIFIED_TAG_NAME = 'QSCALED_ALL_RUNS'
        if not wandb_tags:
            wandb_tags = [UNSPECIFIED_TAG_NAME]
        elif isinstance(wandb_tags, str):
            wandb_tags = [wandb_tags]

        collector_factory = lambda: self.__class__(self._wandb_entity, self._wandb_project, wandb_tags=None)
        collectors = []

        for tag in wandb_tags:
            collector = collector_factory()

            if use_cache and os.path.exists(os.path.join(self._path, tag + '.npy')):
                collector.load_state(tag)
            else:
                wandb_str = f'{self._wandb_entity}/{self._wandb_project}'
                if tag is UNSPECIFIED_TAG_NAME:
                    tag_filter = {}
                    tqdm_desc = f'{wandb_str}: all runs'
                else:
                    tag_filter = {'tags': {'$in': [tag]}}
                    tqdm_desc = f'{wandb_str}: {tag}'

                runs = api.runs(f'{collector._wandb_entity}/{collector._wandb_project}', tag_filter)
                insert_verbose = lambda run: collector._insert_wandb_run(run, verbose)

                if parallel:
                    with ThreadPool(int(os.cpu_count() * 0.5)) as pool:
                        list(tqdm(pool.imap(insert_verbose, runs), total=len(runs), desc=tqdm_desc))
                else:
                    for run in tqdm(runs, total=len(runs), desc=tqdm_desc):
                        insert_verbose(run)

            collector.save_state(tag)
            collectors.append(collector)

        combined_collector = self.__class__.merge(*collectors)
        self.copy_state(combined_collector)

    def _resolve_logging_freq(self, df: pd.DataFrame, logging_freq):
        """
        Resolves non-uniform logging frequencies. Rounds step up to nearest
        multiple of `logging_freq`, then averages over potential duplicates.
        Since performance is usually increasing, in general the output
        will be more conservative than the input.
        """
        df = df.copy()
        df['rounded_step'] = np.ceil(df['_step'] / logging_freq) * logging_freq
        agg_dict = {'_step': 'first', **{col: 'mean' for col in df.columns if col.startswith('seed')}}
        return df.groupby('rounded_step').agg(agg_dict).dropna().reset_index(drop=True)
