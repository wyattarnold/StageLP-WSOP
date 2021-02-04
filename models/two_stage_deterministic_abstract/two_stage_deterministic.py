#  ___________________________________________________________________________
#
#  Two-stage stochastic linear optimization of water supply portfolio
#  
#  Author: Wyatt Arnold
#  References: Lund 1995, Wilchfort & Lund 1997, Rosenberg 2009
#  ___________________________________________________________________________

#
# Imports
#

from pyomo.core import (AbstractModel, Set, Param, Var, sum_product, quicksum,
                        PositiveReals, NonNegativeReals, NonNegativeIntegers, Integers, Constraint,
                        Objective, Expression, minimize)


#
# Model
#

model = AbstractModel()

#
# Parameters
#

model.LT = Set()

model.LT_MAX = Param(model.LT, within=NonNegativeIntegers)

model.LT_QF = Param(model.LT, within=NonNegativeIntegers)

model.C_LT = Param(model.LT, within=NonNegativeReals)

model.ST = Set()

model.ST_MAX = Param(model.ST, within=NonNegativeReals)

model.C_ST = Param(model.ST, within=NonNegativeReals)

# model.G = Param(model.ST, model.LT, within=Integers)

model.SHORTAGE = Set()

model.SHORTAGE_Q = Param(model.SHORTAGE, within=NonNegativeReals)

#
# Variables
#

model.LT_ACTION = Var(model.LT, bounds=(0, None), within=NonNegativeIntegers)

model.ST_Q = Var(model.ST, bounds=(0, None), within=NonNegativeReals)

#
# Constraints
#

def MeetShortageRequirement_rule(model):
    st_q = sum(model.ST_Q[j] for j in model.ST)
    return sum_product(model.LT_QF, model.LT_ACTION) + st_q >= model.SHORTAGE_Q['SH']
model.MeetShortageRequirement = Constraint(rule=MeetShortageRequirement_rule)

def LongTermMax_rule(model, i):
    return model.LT_ACTION[i] <= model.LT_MAX[i]
model.LongTermMax = Constraint(model.LT, rule=LongTermMax_rule)

def ShortTermMax_rule(model, j):
    return model.ST_Q[j] <= model.ST_MAX[j]
model.ShortTermMax = Constraint(model.ST, rule=ShortTermMax_rule)

def ShortTermRestrict_rule(model):
    return model.ST_Q['RESTRICT'] <= model.LT_MAX['LSRETRO'] - model.LT_ACTION['LSRETRO']
model.ShortTermRestrict = Constraint(rule=ShortTermRestrict_rule)

def ShortTermOption_rule(model):
    return model.ST_Q['EX_OPTION'] <= model.LT_ACTION['OPTION']
model.ShortTermOption = Constraint(rule=ShortTermOption_rule)

def ShortTermNonNegativity_rule(model, j):
    return model.ST_Q[j] >= 0 
model.ShortTermNonNegativity = Constraint(model.ST, rule=ShortTermNonNegativity_rule)

def LongTermNonNegativity_rule(model, i):
    return model.LT_ACTION[i] >= 0
model.LongTermNonNegativity = Constraint(model.LT, rule=LongTermNonNegativity_rule)

#
# Stage-specific cost computations
#

def ComputeFirstStageCost_rule(model):
    return sum_product(model.C_LT, model.LT_ACTION)

model.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)

def ComputeSecondStageCost_rule(model):
    return sum_product(model.C_ST, model.ST_Q)

model.SecondStageCost = Expression(rule=ComputeSecondStageCost_rule)


#
# minimize: sum of StageCosts
#

def total_cost_rule(model):
    return model.FirstStageCost + model.SecondStageCost
model.Total_Cost_Objective = Objective(rule=total_cost_rule, sense=minimize)
