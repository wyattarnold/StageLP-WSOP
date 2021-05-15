#  ___________________________________________________________________________
#
#  Two-stage stochastic linear optimization of water supply portfolio
#  Concrete Pyomo model
#  
#  Author: Wyatt Arnold
#  References: Lund 1995, Wilchfort & Lund 1997, Rosenberg 2009
#  ___________________________________________________________________________

#
# sh command
#

# runef --solve -m=three_stage.py --solver=gurobi --solver-options="NonConvex=2 TimeLimit=60" --solution-writer=pyomo.pysp.plugins.csvsolutionwriter
# runef --solve -m=three_stage.py --solver=gurobi --solver-options="NonConvex=2" --solution-writer=pyomo.pysp.plugins.csvsolutionwriter --generate-weighted-cvar --cvar-weight=0.1 --risk-alpha=0.95



#
# Imports
#

import os, json
import numpy as np
import networkx 
import pyomo.environ as pyo



#
# Model
#

model = pyo.ConcreteModel()



#
# data
#
 
with open(os.path.join(os.path.dirname(__file__), 'model_data.json')) as f:
  data = json.load(f)



#
# Sets
#

model.LT = pyo.Set(initialize=list(data['LT_MAX'].keys()))
model.LT_EXP = pyo.Set(initialize=list(data['LT_EXP'].keys()))
model.ST = pyo.Set(initialize=list(data['ST_MAX'].keys()))
model.SHORTAGE = pyo.Set(initialize=['SH'])
model.SHORT = pyo.Set(initialize=list(data['SHORT_Q_MAX']['P3'].keys()))



#
# Parameters
#

def lt_min(model, i):
    return data['LT_MIN'][i]
model.LT_MIN = pyo.Param(model.LT, within=pyo.NonNegativeReals, initialize=lt_min)

def lt_max(model, i):
    return data['LT_MAX'][i]
model.LT_MAX = pyo.Param(model.LT, within=pyo.NonNegativeReals, initialize=lt_max)

def c_lt(model, i):
    return data['C_LT'][i]
model.C_LT = pyo.Param(model.LT, within=pyo.NonNegativeReals, initialize=c_lt)

def st_min(model, i):
    return data['ST_MIN'][i]
model.ST_MIN = pyo.Param(model.ST, within=pyo.NonNegativeReals, initialize=st_min)

def st_max(model, i):
    return data['ST_MAX'][i]
model.ST_MAX = pyo.Param(model.ST, within=pyo.NonNegativeReals, initialize=st_max)

def c_st(model, i):
    return data['C_ST'][i]
model.C_ST = pyo.Param(model.ST, within=pyo.NonNegativeReals, initialize=c_st)

def exp_permit_cost(model, i):
    return data['LT_EXP'][i]['permit_cost']
model.LT_EXP_PERMIT_COST = pyo.Param(model.LT_EXP, within=pyo.NonNegativeReals, initialize=exp_permit_cost)


# mutable parameters (per scenario)
model.SHORTAGE_Q = pyo.Param(model.SHORTAGE, within=pyo.NonNegativeReals, initialize=0.0, mutable=True)
model.SHORT_Q_MAX = pyo.Param(model.SHORT, within=pyo.NonNegativeReals, initialize=0.0,  mutable=True)                                              
model.SHORT_COST = pyo.Param(model.SHORT, within=pyo.NonNegativeReals, initialize=0.0, mutable=True)



#
# Variables
#

model.LT_ACTION = pyo.Var(model.LT, bounds=(0, None), within=pyo.NonNegativeReals, initialize=0)

model.ST_ACTION = pyo.Var(model.ST, bounds=(0, None), within=pyo.NonNegativeReals, initialize=0)

model.LT_EXP_PERMIT_ACTION = pyo.Var(model.LT_EXP, bounds=(0, 1), within=pyo.Binary, initialize=0)

model.LT_EXP_ACTION = pyo.Var(model.LT_EXP, bounds=(0, 1e6), within=pyo.NonNegativeReals, initialize=0)

model.LT_EXP_COST = pyo.Var(model.LT_EXP, bounds=(0, 1e6), within=pyo.NonNegativeReals, initialize=0)

model.EXP_ACTION = pyo.Var(model.LT_EXP, bounds=(0, None), within=pyo.NonNegativeReals, initialize=0)

model.EXP_BOP_ACTION = pyo.Var(model.LT_EXP, bounds=(0, None), within=pyo.NonNegativeReals, initialize=0)

model.EXP_VOP_ACTION = pyo.Var(model.LT_EXP, bounds=(0, None), within=pyo.NonNegativeReals, initialize=0)

model.SHORT_ACTION = pyo.Var(model.SHORT, 
                          bounds=(0, None),
                          within=pyo.NonNegativeReals, 
                          initialize=0)



#
# LT Expansion Piecewise functions
#

piecewise_representation = 'INC'

bkpts = [0] + (np.arange(5e3,1.5e5,1e4)).tolist() + \
        (np.arange(1.5e5,4.5e5,5e4)).tolist() + \
        (np.arange(4.5e5,1.1e6,2e5)).tolist() 

def scale_marginal(model, LT_EXP, LT_EXP_ACTION):
    if LT_EXP_ACTION==0:
        return 0
    else:
        p = data['LT_EXP'][LT_EXP]['p']
        mult = data['LT_EXP'][LT_EXP]['multiplier']
        return p * mult * LT_EXP_ACTION**(p-1)

model.LT_EXP_pw = pyo.Piecewise(model.LT_EXP, model.LT_EXP_COST, model.LT_EXP_ACTION,
                              pw_pts=bkpts, pw_constr_type='EQ', f_rule=scale_marginal,
                              pw_repn=piecewise_representation)



#
# Constraints
#

def MeetShortage_rule(model):
    lt_q = model.LT_ACTION['RETRO']
    exp_vop_q = pyo.quicksum(model.EXP_VOP_ACTION[i]*model.LT_EXP_PERMIT_ACTION[i] for i in model.LT_EXP)
    st_q = pyo.quicksum(model.ST_ACTION[i] for i in model.ST)
    short_q = pyo.quicksum(model.SHORT_ACTION[i] for i in model.SHORT_ACTION)
    return  lt_q + st_q + exp_vop_q + short_q >= model.SHORTAGE_Q['SH']
model.MeetShortage = pyo.Constraint(rule=MeetShortage_rule)

def LongTermMax_rule(model, i):
    return model.LT_ACTION[i] <= model.LT_MAX[i]
model.LongTermMax = pyo.Constraint(model.LT, rule=LongTermMax_rule)

def LongTermExp_rule(model, i):
    return model.EXP_ACTION[i] / data['LT_EXP'][i]['exp_max'] <= model.LT_EXP_ACTION[i]
model.LongTermExp = pyo.Constraint(model.LT_EXP, rule=LongTermExp_rule)

def BaselineMinOp_rule(model, i):
    total_lt_exp = model.EXP_ACTION[i] + model.LT_EXP_ACTION[i]
    min_ratio = data['LT_EXP'][i]['baseline_op_min_ratio']
    return model.EXP_BOP_ACTION[i] >= min_ratio*total_lt_exp
model.BaselineMinOp = pyo.Constraint(model.LT_EXP, rule=BaselineMinOp_rule)

def BaselineMaxOp_rule(model, i):
    total_lt_exp = model.EXP_ACTION[i] + model.LT_EXP_ACTION[i]
    return model.EXP_BOP_ACTION[i] <= total_lt_exp
model.BaselineMaxOp = pyo.Constraint(model.LT_EXP, rule=BaselineMaxOp_rule)

def VariableMinOp_rule(model, i):
    return model.EXP_VOP_ACTION[i] >= model.EXP_BOP_ACTION[i]
model.VariableMinOp = pyo.Constraint(model.LT_EXP, rule=VariableMinOp_rule)

def VariableMaxOp_rule(model, i):
    total_lt_exp = model.EXP_ACTION[i] + model.LT_EXP_ACTION[i]
    return model.EXP_VOP_ACTION[i] <= total_lt_exp
model.VariableMaxOp = pyo.Constraint(model.LT_EXP, rule=VariableMaxOp_rule)

def ShortTermMax_rule(model, k):
    return model.ST_ACTION[k] <= model.ST_MAX[k]
model.ShortTermMax = pyo.Constraint(model.ST, rule=ShortTermMax_rule)

def ShortTermRestrict_rule(model):
    return model.LT_ACTION['RETRO'] + model.ST_ACTION['RESTRICT'] <= model.LT_MAX['RETRO']
model.ShortTermRestrict = pyo.Constraint(rule=ShortTermRestrict_rule)

def LTOption_rule(model):
    return model.ST_ACTION['OPTION'] <= model.LT_ACTION['OPTION']
model.LTOption = pyo.Constraint(rule=LTOption_rule)

def ShortMax_rule(model, i):
    return model.SHORT_ACTION[i] <= model.SHORT_Q_MAX[i]
model.ShortMax = pyo.Constraint(model.SHORT, rule=ShortMax_rule)



#
# Stage-specific cost computations
#

def ComputeFirstStageCost_rule(model):

    lt_actions = pyo.sum_product(model.C_LT, model.LT_ACTION)

    lt_exp_actions = pyo.sum_product(model.LT_EXP_ACTION, model.LT_EXP_COST)

    lt_exp_permit_action = pyo.quicksum(model.LT_EXP_PERMIT_ACTION[i]*model.LT_EXP_PERMIT_COST[i] for i in model.LT_EXP)

    return  lt_actions + lt_exp_actions + lt_exp_permit_action



def ComputeSecondStageCost_rule(model):

    expansion_cost = pyo.quicksum(model.EXP_ACTION[i]*model.LT_EXP_COST[i] for i in model.LT_EXP)

    baseline_op_cost = pyo.quicksum(model.EXP_BOP_ACTION[i]*data['LT_EXP'][i]['baseline_op_cost'] for i in model.LT_EXP)

    return  expansion_cost + baseline_op_cost



def ComputeThirdStageCost_rule(model):

    st_actions = pyo.sum_product(model.C_ST, model.ST_ACTION)

    short_cost = pyo.quicksum(model.SHORT_ACTION[i]*model.SHORT_COST[i] for i in model.SHORT_ACTION)

    exp_op_cost = pyo.quicksum(model.EXP_VOP_ACTION[i]*data['LT_EXP'][i]['variable_op_cost'] for i in model.LT_EXP) 

    return st_actions + short_cost + exp_op_cost


model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)
model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)
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
    for i in instance.SHORT:
        instance.SHORT_Q_MAX[i] = data['SHORT_Q_MAX'][node_names[1]][i]
        instance.SHORT_COST[i] = data['SHORT_COST'][node_names[1]][i]
    return instance



#
# Decision Tree
#

def pysp_scenario_tree_model_callback():
    # Return a NetworkX scenario tree.
    g = networkx.DiGraph()

    ce1 = "CostExpressions[1]"
    ce2 = "CostExpressions[2]"
    ce3 = "CostExpressions[3]"

    g.add_node("Root",
               cost = ce1,
               variables = ["LT_ACTION[*]","LT_EXP_ACTION[*]","LT_EXP_PERMIT_ACTION[*]"],
               derived_variables = ["LT_EXP_COST[*]"])

    for projection in data['PROJECTION_P']:

        g.add_node(projection,
                    cost = ce2,
                    variables = ["EXP_ACTION[*]","EXP_BOP_ACTION[*]"],
                    derived_variables = [])
        g.add_edge("Root", projection, weight=data['PROJECTION_P'][projection])

        for shortage in data['SHORTAGE_P'][projection]:

            g.add_node(shortage,
                    cost = ce3,
                    variables = ["ST_ACTION[*]","SHORT_ACTION[*]","EXP_VOP_ACTION[*]"],
                    derived_variables = [])
            g.add_edge(projection, shortage, weight=data['SHORTAGE_P'][projection][shortage])

    return g
