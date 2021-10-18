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

|Features granularity| Features normalization| Dimensionality reduction | Data augmentation| Trained model | Specialization | Credibility  | Coordination
|--------------------|-----------------------|--------------------------|------------------|---------------|----------------|--------------|-------------
|                    |                       |                          |                  |Random guess   |20.0            |20.0          |20.0
|High  | ✅| ❌ |  ✅  | CNN | **46.8 ± 20.9** | 46.5 ± 20.3|25.7 ± 16.6
|High  |  ✅ | ❌ | ❌  | CNN | 43.3 ± 10.1 | 38.9 ± 10.8 | 16.7 ± 8.6
|Low   |  ✅ | ❌ | ❌  | MLP | 37.0 ± 17.0 | 55.6 ± 12.0 | 27.6 ± 17.1
|Low   |  ✅ | ❌ |  ✅  | MLP | 30.8 ± 3.8 | 28.8 ± 5.7 | 50.0\*
|Low   |  ✅ | ❌ | ❌  | LR  | 25.0\* | 25.0\* | 27.5\*
|Low   |  ✅ |  ✅ | ❌  | LR | 43.3\* | 40.0\* | 33.3\*
|Low   |  ✅ |  ✅ |  ✅  | LR | 50.5\* | 51.7\* | 58.3\*
|Low   | ❌ | ❌ | ❌  | DT  | 45.7 ± 7.3 | 57.1 ± 2.6 | 59.1 ± 7.2
|Low   |  ✅ | ❌ | ❌  | DT |44.8 ± 7.7 | 41.2 ± 7.3$ | 59.2 ± 6.8
|Low   | ❌ | ❌ |  ✅  | DT |42.8 ± 3.8 | 54.2 ± 2.6 |80.0\*
|Low   |  ✅ | ❌ |  ✅  | DT |42.8 ± 3.8 | 54.2 ± 2.6 | 80.0 ± 1.1
|Low   |  ✅ |  ✅ |  ✅  | DT | **51.8 ± 2.1**|**58.3\*\ ** | **83.3\*\ **
