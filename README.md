# Value-Based Deep RL Scales Predictably
### [Preprint](https://arxiv.org/abs/2502.04327)

Implementation of a workflow for evaluating trade-offs between data efficiency,
compute efficiency, and performance for online RL, validated across multiple
environments.

 [Oleh Rybkin](https://people.eecs.berkeley.edu/~oleh/)<sup>1</sup>,
 [Michal Nauman](https://scholar.google.com/citations?user=GnEVRtQAAAAJ&hl=en)<sup>1,2</sup>,
 [Preston Fu](https://prestonfu.com/)<sup>1</sup>,
 [Charlie Snell](https://sea-snell.github.io/)<sup>1</sup>,
 [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/)<sup>1</sup>,
 [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>1</sup>,
 [Aviral Kumar](https://aviralkumar2907.github.io/)<sup>3</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>University of Warsaw, <sup>3</sup>Carnegie Mellon University

<img src='assets/scaling.png'/>

## Installation

```
pip install -e .
```

It requires Python 3.10+.

## Running code

### Workflow
1. Run a hyperparameter grid search over UTD $\sigma$, batch size $B$, learning rate $\eta$, with logging to [Wandb](https://wandb.ai/).
2. Run [`analyze_grid_search.ipynb`](analyze_grid_search.ipynb). This produces fits 
   $B^* (\sigma)$, $\eta^* (\sigma)$, as well as a baseline using the best $(B, \eta)$ 
   setting for some $\sigma$ on each environment.
3. Run the proposed fit and the baseline.
4. Run [`analyze_fitted.ipynb`](analyze_fitted.ipynb).

### Replicating paper results
We have provided sample `zip` files containing run data fetched from Wandb in 
[`data/zip`](data/zip). Running the aforementioned notebooks above will use those 
zip` files by default.

## Citation
```
@misc{rybkin2025valuebaseddeeprlscales,
      title={Value-Based Deep RL Scales Predictably}, 
      author={Oleh Rybkin and Michal Nauman and Preston Fu and Charlie Snell and Pieter Abbeel and Sergey Levine and Aviral Kumar},
      year={2025},
      eprint={2502.04327},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.04327}, 
}
```