# Neural tangent kernel

## Preliminary

- We know that at init, ANN can be represented as GP
  - Maybe show recursive formula of MLP as GP
- Define a dataset
- Show how we define SGD with a time step t on this gp
- Now, What if we put it in function space
- SGD becomes linear kernel func -> that's how we get the NTK

## What's the NTK ?

- Give ntk formula
- What 2028 paper brought : NTK is deterministic when n_1, ..., n_L-1 -> infinity + Doesn't change during training
- Kernel doesn't change with t, so SGD becomes ODE with NTK in it   
- So : the model function f after SGD (when t -> infinity) can be computed.
- The compute finally gives kernel ridge regression under NTK gram matrix, which happens to be GP with kernel NTK
- Maybe show infinite-width recursive formula in the case of MLP

## Lazy training regime

Question : How much does the above statements hold when we have realistic widths ?


