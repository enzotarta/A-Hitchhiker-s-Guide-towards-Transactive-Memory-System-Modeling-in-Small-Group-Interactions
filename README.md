# A Hitchhiker's Guide towards Transactive Memory System Modeling in Small Group Interactions

[![DOI](https://zenodo.org/badge/doi/10.1145/3461615.3485414.svg)](http://dx.doi.org/10.1145/3461615.3485414)

The official repository.

Dependencies to execute the pipeline:
* torch (>= 1.8)
* torchvision (>= 0.9)
* numpy (>= 1.20)
* sklearn (>= 0.23)
* scipy (>= 1.6)
* matplotlib (>= 3.4)

## Performance

Features granularity| Features normalization| Dimensionality reduction | Data augmentation| Trained model | Specialization | Credibility  | Coordination
------------|---------------|-----------|--------------|-------|----------------|--------------|-------------
            |               |           |              |Random guess |20.0| |20.0|20.0
        High  & \cmark & \xmark  & \cmark  & CNN & $46.8\pm 20.9$ & $ 46.5\pm 20.3$ & $25.7 \pm 16.6$\\
        High  & \cmark & \xmark  & \xmark  & CNN &$43.3\pm 10.1$ & $ 38.9\pm 10.8$ & $16.7 \pm 8.6$\\
        Low   & \cmark & \xmark  & \xmark  & MLP & $37.0 \pm 17.0$ & $55.6\pm 12.0$  & $27.6 \pm 17.1$\\
        Low   & \cmark & \xmark  & \cmark  & MLP & $30.8 \pm 3.8$ & $28.8 \pm 5.7$ & $50.0*$\\
        Low   & \cmark & \xmark  & \xmark  & LR & $25.0*$ & $25.0*$  & $27.5*$ &\\
        Low   & \cmark & \cmark  & \xmark  & LR & $43.3*$ & $40.0*$& $33.3*$\\
        Low   & \cmark & \cmark  & \cmark  & LR & $ 50.5*$ & $ 51.7*$& $58.3*$\\
        Low   & \xmark & \xmark  & \xmark  & DT & $45.7\pm 7.3$&$57.1\pm 2.6$ & $59.1\pm 7.2$ \\
        Low   & \cmark & \xmark  & \xmark  & DT & $44.8\pm 7.7$  &$41.2 \pm 7.3$  &$59.2\pm 6.8$\\
        Low   & \xmark & \xmark  & \cmark  & DT & $42.8\pm 3.8$&$54.2\pm 2.6$ & $80.0*$ \\
        Low   & \cmark & \xmark  & \cmark  & DT & $42.8\pm 3.8$  &$ 54.2\pm 2.6$  &$80.0\pm 1.1$\\
        Low   & \cmark & \cmark  & \cmark  & DT & $\mathbf{51.8\pm 2.1}$&$\mathbf{58.3*}$ & $\mathbf{83.3*}$ \\
