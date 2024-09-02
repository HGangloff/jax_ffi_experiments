import os
os.environ["JAX_PLATFORMS"] = "cpu"
import time

import jax

key = jax.random.PRNGKey(0)

from mrfx.models import Potts
from mrfx.samplers import GibbsSampler
from mrfx.experiments import time_complete_sampling

rows = 200
cols = 200
Q = 3
beta = 1
n_iter = 1000

key, subkey = jax.random.split(key, 2)
times, n_iterations = time_complete_sampling(
    GibbsSampler,
    Potts,
    subkey,
    [Q],
    [(rows, cols)],
    5,
    kwargs_sampler={"eps": 0.01, "max_iter":n_iter, "cv_type":"iter_only"},
    kwargs_model={"beta":1.}
)
print("JAX on CPU", times)
