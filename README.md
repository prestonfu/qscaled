# Value-Based Deep RL Scales Predictably
### [Paper](https://arxiv.org/pdf/2502.04327)
Implementation of a workflow for evaluating trade-offs between data efficiency,
compute efficiency, and performance for online RL, validated across multiple
environments.

 [Oleh Rybkin](https://people.eecs.berkeley.edu/~oleh/)\*<sup>1</sup>,
 [Michal Nauman](https://scholar.google.com/citations?user=GnEVRtQAAAAJ&hl=en)\*<sup>1,2</sup>,
 [Preston Fu](https://prestonfu.com/)\*<sup>1</sup>,
 [Charlie Snell](https://sea-snell.github.io/)\*<sup>1</sup>,
 [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/)<sup>1</sup>,
 [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>1</sup>,
 [Aviral Kumar](https://aviralkumar2907.github.io/)<sup>3</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>University of Warsaw, <sup>3</sup>Carnegie Mellon University

In submission.


<img src='assets/scaling.png'/>

## Setup

To setup a conda environment,
```
 conda env -create -f environment.yml
```

## Running code

### Workflow
* Run a hyperparameter grid search over UTD $\sigma$, batch size $B$, learning rate $\eta$.
* Run [`analyze_grid_search.ipynb`](analyze_grid_search.ipynb). This produces fits 
  $B^* (\sigma)$, $\eta^* (\sigma)$, as well as a baseline using the best $(B, \eta)$ 
  setting for some $\sigma$ on each environment.
* Run the proposed fit and the baseline.
* Run [`analyze_fitted.ipynb`](analyze_fitted.ipynb).

### Replicating paper results
We have provided sample `zip` files containing run data fetched from Wandb in 
[`cache/zip`](cache/zip). Running the aforementioned notebooks above will use those 
zip` files by default.

## Citation
```
@article{rybkin2025value,
  title={Value-Based Deep RL Scales Predictably},
  author={Rybkin, Oleh and Nauman, Michal and Fu, Preston and Snell, Charlie and Abbeel, Pieter and Levine, Sergey and Kumar, Aviral},
  journal={arXiv preprint arXiv:2502.04327},
  year={2025}
}
```