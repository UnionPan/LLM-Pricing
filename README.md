# PACT Framework: Contract Theory for Agentic AI Services

A Python implementation of the **PACT** framework for pricing agentic AI services. This project uses contract theory to find the optimal menu of Quality of Service (QoS) and Price pairs that a service provider can offer to users with unknown preferences, maximizing the provider's expected profit.

***
## Files

* **`contract_optimizer.py`**: Contains the `ContractOptimizer` class. This is the core engine that solves the baseline PACT contract design problem.
* **`liability_comparison.py`**: Contains the `LiabilityCostComparison` class. This tool runs and visualizes comparative analyses, showing how optimal contracts change under different provider cost scenarios (e.g., with/without liability costs).

***
## Requirements

The project requires `numpy`, `scipy`, and `matplotlib`.

```bash
pip install numpy scipy matplotlib