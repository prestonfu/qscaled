# title

## Setup

## Workflow
* Run a hyperparameter grid search over UTD $\sigma$, batch size $B$, learning rate $\eta$.
* Run [analyze_grid_search.ipynb](analyze_grid_search.ipynb). This produces fits 
  $B^*(\sigma)$, $\eta^*(\sigma)$, as well as a baseline using the best $(B, \eta)$ 
  setting for some $\sigma$ on each environment.
* Run the proposed fit and the baseline.
* Run [analyze_fitted.ipynb](analyze_fitted.ipynb).