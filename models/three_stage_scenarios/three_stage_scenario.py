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

import os,json
import networkx 
import pyomo.environ as pyo

#
# Model
#

model = pyo.ConcreteModel()

#
# Parameters
#

with open(os.path.join(os.path.dirname(__file__),
          'three_stage_scenario_data_dict.json')) as f:
  data = json.load(f)

model.LT = pyo.Set(initialize=list(data['LT_MAX'].keys()))

model.LT_MAX = data['LT_MAX']

model.LT_QF = data['LT_QF']

model.C_LT = data['C_LT']

model.MT = pyo.Set(initialize=list(data['MT_MAX'].keys()))

model.MT_MAX = data['MT_MAX']

model.C_MT = data['C_MT']

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

model.MT_EXP = pyo.Var(model.MT, 
                          bounds=(0, 1), 
                          within=pyo.PercentFraction)

model.ST_Q = pyo.Var(model.ST, 
                     bounds=(0, None),
                     within=pyo.NonNegativeReals)

#
# Constraints
#

def MeetShortage_rule(model):
    lt_q = pyo.sum_product(model.LT_QF, model.LT_ACTION)
    mt_q = pyo.quicksum(model.LT_QF[i]*model.LT_ACTION[i]*model.MT_EXP[i] for i in model.LT)
    st_q = pyo.quicksum(model.ST_Q[k] for k in model.ST)
    tot_q = lt_q + mt_q + st_q
    return  tot_q >= model.SHORTAGE_Q['SH']
model.MeetShortage = pyo.Constraint(rule=MeetShortage_rule)

def LongTermMax_rule(model, i):
    return model.LT_ACTION[i] <= model.LT_MAX[i]
model.LongTermMax = pyo.Constraint(model.LT, rule=LongTermMax_rule)

def MidTermMax_rule(model, j):
    return model.MT_EXP[j] <= model.MT_MAX[j]
model.MidTermMax = pyo.Constraint(model.MT, rule=MidTermMax_rule)

def MidTermLSRetro_rule(model):
    lt_retro = model.LT_ACTION['LS_RETRO']
    return lt_retro * model.MT_EXP['LS_RETRO'] + lt_retro <= model.LT_MAX['LS_RETRO']
model.MidTermLSRetro = pyo.Constraint(rule=MidTermLSRetro_rule)

def ShortTermMax_rule(model, k):
    return model.ST_Q[k] <= model.ST_MAX[k]
model.ShortTermMax = pyo.Constraint(model.ST, rule=ShortTermMax_rule)

def ShortTermRestrict_rule(model):
    lt_retro = model.LT_ACTION['LS_RETRO']
    st_q_ = model.ST_Q['LS_RESTRICT'] / model.LT_QF['LS_RETRO']
    return st_q_ + lt_retro * model.MT_EXP['LS_RETRO'] + lt_retro  <= model.LT_MAX['LS_RETRO']
model.ShortTermRestrict = pyo.Constraint(rule=ShortTermRestrict_rule)

def LTOption_rule(model):
    return model.ST_Q['EX_LT_OPTION'] <= model.LT_ACTION['OPTION']
model.LTOption = pyo.Constraint(rule=LTOption_rule)

def MTOption_rule(model):
    mt_option = model.LT_ACTION['OPTION'] * model.MT_EXP['OPTION']
    return model.ST_Q['EX_MT_OPTION'] <= mt_option
model.MTOption = pyo.Constraint(rule=MTOption_rule)

def ShortTermNonNegativity_rule(model, k):
    return model.ST_Q[k] >= 0 
model.ShortTermNonNegativity = pyo.Constraint(model.ST, rule=ShortTermNonNegativity_rule)

def MidTermNonNegativity_rule(model, j):
    return model.MT_EXP[j] >= 0 
model.MidTermNonNegativity = pyo.Constraint(model.MT, rule=MidTermNonNegativity_rule)

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
    return pyo.quicksum(model.C_MT[i]*model.MT_EXP[i]*model.LT_ACTION[i]*model.LT_QF[i] for i in model.LT) + pyo.quicksum(1000*model.MT_EXP[i] for i in model.LT)
model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

def ComputeThirdStageCost_rule(model):
    return pyo.sum_product(model.C_ST, model.ST_Q)
model.ThirdStageCost = pyo.Expression(rule=ComputeThirdStageCost_rule)

#
# minimize: sum of StageCosts
#

StageSet = pyo.RangeSet(3)
def cost_rule(m, stage):
    # Just assign the expressions to the right stage
    if stage == 1:
        return model.FirstStageCost
    if stage == 2:
        return model.SecondStageCost
    if stage == 3:
        return model.ThirdStageCost
model.CostExpressions = pyo.Expression(StageSet, rule=cost_rule)

def total_cost_rule(model):
    return model.FirstStageCost + model.SecondStageCost + model.ThirdStageCost
model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

#
# Stochastic Data
#

SHORTAGE_Q = {}
for projection in data['SHORTAGE_Q']: 
    SHORTAGE_Q.update(data['SHORTAGE_Q'][projection])


SHORTAGE_P = {}
for projection in data['SHORTAGE_P']: 
    SHORTAGE_P.update(data['SHORTAGE_P'][projection])

def pysp_instance_creation_callback(scenario_name, node_names):

    instance = model.clone()
    instance.SHORTAGE_Q.store_values(SHORTAGE_Q[scenario_name])

    return instance

def pysp_scenario_tree_model_callback():
    # Return a NetworkX scenario tree.
    g = networkx.DiGraph()

    ce1 = "CostExpressions[1]"
    ce2 = "CostExpressions[2]"
    ce3 = "CostExpressions[3]"

    g.add_node("Root",
               cost = ce1,
               variables = ["LT_ACTION[*]"],
               derived_variables = [])

    for projection in data['PROJECTION_P']:

        g.add_node(projection,
                    cost = ce2,
                    variables = ["MT_EXP[*]"],
                    derived_variables = [])
        g.add_edge("Root", projection, weight=data['PROJECTION_P'][projection])

        for shortage in data['SHORTAGE_P'][projection]:

            g.add_node(shortage,
                    cost = ce3,
                    variables = ["ST_Q[*]"],
                    derived_variables = [])
            g.add_edge(projection, shortage, weight=data['SHORTAGE_P'][projection][shortage])

    return g
