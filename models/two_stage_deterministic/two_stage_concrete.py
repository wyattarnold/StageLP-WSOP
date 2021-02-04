#  ___________________________________________________________________________
#
#  Two-stage stochastic linear optimization of water supply portfolio
#  Concrete Pyomo model
#  
#  Author: Wyatt Arnold
#  References: Lund 1995, Wilchfort & Lund 1997, Rosenberg 2009
#  ___________________________________________________________________________

#
# Imports
#

import json
import networkx 
import pyomo.environ as pyo

#
# Model
#

model = pyo.ConcreteModel()

#
# Parameters
#

with open('two_stage_data_dict.json') as f:
  data = json.load(f)

model.LT = pyo.Set(initialize=list(data['LT_MAX'].keys()))

model.LT_MAX = data['LT_MAX']

model.LT_QF = data['LT_QF']

model.C_LT = data['C_LT']

model.ST = pyo.Set(initialize=list(data['ST_MAX'].keys()))

model.ST_MAX = data['ST_MAX']

model.C_ST = data['C_ST']

model.SHORTAGE = pyo.Set(initialize=['SH'])

model.SHORTAGE_Q = pyo.Param(model.SHORTAGE,
                             within=pyo.NonNegativeReals,
                             initialize=0.0,
                             mutable=True)

#
# Variables
#

model.LT_ACTION = pyo.Var(model.LT, 
                          bounds=(0, None), 
                          within=pyo.NonNegativeIntegers)

model.ST_Q = pyo.Var(model.ST, 
                     bounds=(0, None),
                     within=pyo.NonNegativeReals)

#
# Constraints
#

def MeetShortage_rule(model):
    st_q = pyo.quicksum(model.ST_Q[j] for j in model.ST)
    return pyo.sum_product(model.LT_QF, model.LT_ACTION) + st_q >= model.SHORTAGE_Q['SH']
model.MeetShortage = pyo.Constraint(rule=MeetShortage_rule)

def LongTermMax_rule(model, i):
    return model.LT_ACTION[i] <= model.LT_MAX[i]
model.LongTermMax = pyo.Constraint(model.LT, rule=LongTermMax_rule)

def ShortTermMax_rule(model, j):
    return model.ST_Q[j] <= model.ST_MAX[j]
model.ShortTermMax = pyo.Constraint(model.ST, rule=ShortTermMax_rule)

def ShortTermRestrict_rule(model):
    return model.ST_Q['LS_RESTRICT'] <= model.LT_MAX['LS_RETRO'] - model.LT_ACTION['LS_RETRO']
model.ShortTermRestrict = pyo.Constraint(rule=ShortTermRestrict_rule)

def ShortTermOption_rule(model):
    return model.ST_Q['EX_OPTION'] <= model.LT_ACTION['OPTION']
model.ShortTermOption = pyo.Constraint(rule=ShortTermOption_rule)

def ShortTermNonNegativity_rule(model, j):
    return model.ST_Q[j] >= 0 
model.ShortTermNonNegativity = pyo.Constraint(model.ST, rule=ShortTermNonNegativity_rule)

def LongTermNonNegativity_rule(model, i):
    return model.LT_ACTION[i] >= 0
model.LongTermNonNegativity = pyo.Constraint(model.LT, rule=LongTermNonNegativity_rule)

#
# Stage-specific cost computations
#

def ComputeFirstStageCost_rule(model):
    return pyo.sum_product(model.C_LT, model.LT_ACTION)

model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

def ComputeSecondStageCost_rule(model):
    return pyo.sum_product(model.C_ST, model.ST_Q)

model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

#
# minimize: sum of StageCosts
#

StageSet = pyo.RangeSet(2)
def cost_rule(m, stage):
    # Just assign the expressions to the right stage
    if stage == 1:
        return model.FirstStageCost
    if stage == 2:
        return model.SecondStageCost
model.CostExpressions = pyo.Expression(StageSet, rule=cost_rule)

def total_cost_rule(model):
    return model.FirstStageCost + model.SecondStageCost
model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

#
# Stochastic Data
#

SHORTAGE_Q = data['SHORTAGE_Q']

SHORTAGE_P = data['SHORTAGE_P']

def pysp_instance_creation_callback(scenario_name, node_names):

    instance = model.clone()
    instance.SHORTAGE_Q.store_values(SHORTAGE_Q[scenario_name])

    return instance

def pysp_scenario_tree_model_callback():
    # Return a NetworkX scenario tree.
    g = networkx.DiGraph()

    ce1 = "CostExpressions[1]"
    ce2 = "CostExpressions[2]"

    g.add_node("Root",
               cost = ce1,
               variables = ["LT_ACTION[*]"],
               derived_variables = [])

    for shortage in SHORTAGE_Q:
        g.add_node(shortage,
                cost = ce2,
                variables = ["ST_Q[*]"],
                derived_variables = [])
        g.add_edge("Root", shortage, weight=SHORTAGE_P[shortage])

    return g
