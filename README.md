# Water Supply Portfolio Optimization with Multi-stage Stochastic Mixed-Integer Linear Programming

Requires: 
- [Pyomo](https://github.com/Pyomo/pyomo)
- [networkx](https://networkx.org/)

Solver(s):
- [Gurobi](https://www.gurobi.com/)

Two-stage stochastic programming with Pyomo's [PySP](https://pyomo.readthedocs.io/en/stable/modeling_extensions/pysp.html) modeling extension.

## Run
Run two-stage model [/models/two_stage_deterministic](./models/two_stage_deterministic):
```bash
runef --solve -m=two_stage_concrete.py --solver=gurobi --solution-writer=pyomo.pysp.plugins.csvsolutionwriter 
```

Run three-stage model [/models/three_stage_scenarios](./models/three_stage_scenarios):
```bash
runef --solve -m=three_stage_scenario.py --solver=gurobi  --solver-options="NonConvex=2" --solution-writer=pyomo.pysp.plugins.csvsolutionwriter
```

## References
Lund, J. R. (1995). Derived Estimation of Willingness to Pay to Avoid Probabilistic Shortage. Water Resources Research, 31(5), 1367–1372.

Wilchfort Orit, & Lund Jay R. (1997). Shortage Management Modeling for Urban Water Supply Systems. Journal of Water Resources Planning and Management, 123(4), 250–258.

Rosenberg, D. E., & Lund, J. R. (2009). Modeling Integrated Decisions for a Municipal Water System with Recourse and Uncertainties: Amman, Jordan. Water Resources Management, 23(1), 85.
