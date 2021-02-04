# Water Supply Portfolio Optimization with a Two-stage Stochastic Mixed Integer Linear Program

Requires: 
- [Pyomo](https://github.com/Pyomo/pyomo)
- [networkx](https://networkx.org/)

Two-stage stochastic programming with Pyomo's [PySP](https://pyomo.readthedocs.io/en/stable/modeling_extensions/pysp.html) modeling extension.

## Run
Run abstract with data defined in node/scenario files: 
```bash
runph --model=two_stage_default.py --solver=gurobi --instance-directory=nodedata --default-rho=1 --solution-writer=pyomo.pysp.plugins.csvsolutionwriter --termdiff-threshold=0.01 --max-iterations=20
```

Run concrete model with data defined in a json:
```bash
runph --model=two_stage_concrete.py --solver=gurobi --default-rho=1 --solution-writer=pyomo.pysp.plugins.csvsolutionwriter --termdiff-threshold=0.01 --max-iterations=20
```


## References
Lund, J. R. (1995). Derived Estimation of Willingness to Pay to Avoid Probabilistic Shortage. Water Resources Research, 31(5), 1367–1372.

Wilchfort Orit, & Lund Jay R. (1997). Shortage Management Modeling for Urban Water Supply Systems. Journal of Water Resources Planning and Management, 123(4), 250–258.

Rosenberg, D. E., & Lund, J. R. (2009). Modeling Integrated Decisions for a Municipal Water System with Recourse and Uncertainties: Amman, Jordan. Water Resources Management, 23(1), 85.