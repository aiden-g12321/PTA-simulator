## PTA-simulator

Simulate Pulsar Timing Array (PTA) datasets. As opposed to existing packages (e.g. `libstempo`, `pint`, `pta_replicator`) this package simulates "high-level" data. That is, rather than loading .par/.tim files and fitting a timing model, we directly simulate timing residuals and project them into a space orthogonal to an artificial timing model.

These simulations are useful for testing parameter estimation codes, as it is easy to verify our injections are consistent with models used in the posterior probability density. The timing residuals and TOA errors per pulsar are output. Alternatively, we can output a custom `ptasimulator.data.SimulatedData` which is compatible with `Prometheus` data objects.

Install with
```
pip install git+https://github.com/aiden-g12321/PTA-simulator.git
```


