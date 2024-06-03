from itertools import product
import numpy as np
from model_selection.cross_validation import cross_validate

def grid_search(model, param_grid, X, y):
    """Perform grid search to find the best hyperparameters."""
    best_score = -np.inf
    best_params = {}
    
    # Create a cartesian product of all parameters
    keys, values = zip(*param_grid.items())
    for param_combination in product(*values):
        params = dict(zip(keys, param_combination))
        model.set_params(**params)
        _, score = cross_validate(model, X, y)  # Assuming cross_validate returns mean score
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score
