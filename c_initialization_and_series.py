import pandas as pd
import numpy as np
import time, datetime

case_list = [[1, 1], [0, 1], [1, 0], [0, 0]]  # default
consumption_level_list = list(range(2, 11))  # default
consumption_level_list = [3, 6, 10]  # default

simulation_time = int(10401)

# Auxillary data
stockRP_data, mass_differential_data, sys_mass = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

EETI_dem_data, ESIRP_data = [np.zeros(simulation_time, dtype=float) for _ in range(2)]

inflows_P2, outflows_P2 = [np.zeros(simulation_time, dtype=float) for _ in range(2)]

inflows_P3, outflows_P3 = [np.zeros(simulation_time, dtype=float) for _ in range(2)]

mass_ecological_RP, IS_in, IS_out = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

IRP_in, IRP_out = [np.zeros(simulation_time, dtype=float) for _ in range(2)]

EEHH_dem_data, food_consumed_data, food_waste_data = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

CLI_inventory, IS_inventory, CLI_processing = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

# Compartment variables/ stocks declaration
P1, P2, P3 = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

H1, H2, H3 = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

C1, C2, HH = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

RP, IRP, ES = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

IS_total, TI_inventory = [np.zeros(simulation_time, dtype=float) for _ in range(2)]

# Flows: Intercompartmental
P1H2_data, P2H1_data, P2H2_data, P2H3_data, P3H3_data = [np.zeros(simulation_time, dtype=float) for _ in range(5)]

H1C1_data, H2C1_data, H2C2_data, H3C2_data = [np.zeros(simulation_time, dtype=float) for _ in range(4)]

ISHH_total_supply, TIHH, ISHH_total_demand_data, EECLI_data, EEHH_data = [np.zeros(simulation_time, dtype=float)
                                                                   for _ in range(5)]

# Flows: mortality
P1RP_data, P2RP_data, P3RP_data = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

H1RP_data, H2RP_data, H3RP_data = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

C1RP_data, C2RP_data, HHRP_data = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

# Flows: Recycling and RP flows
IRPP2_data, IRPP3_data, IRPRP_data = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

RPP1_data, RPP2_data, RPP3_data, RPIS_data = [np.zeros(simulation_time, dtype=float) for _ in range(4)]

# Economic variables: wages, and weighted price
wages_data, weightedprice_data = [np.zeros(simulation_time, dtype=float) for _ in range(2)]

# Economic variables: prices
P1_price_data, H1_price_data, TI_price_data, EE_price_data = [np.zeros(simulation_time, dtype=float) for _ in range(4)]

# Economic variables: production
P1_prod_data, H1_prod_data, TI_prod_data = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

# Economic variables: deficits
P1H1_def, P1TI_def, P1HH_def = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

P1_def_total, H1_def, TI_def = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

# Human population related variables
N_HH, per_capita_mass, percapbirths_data = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

# Economic dimension
P1HH_per_cap_demand_data, H1HH_per_cap_demand_data, ISHH_per_cap_demand_data = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

EEHH_per_capita_demand_data, P1HH_data = [np.zeros(simulation_time, dtype=float) for _ in range(2)]

H1HH_data, EEHH_data = [np.zeros(simulation_time, dtype=float) for _ in range(2)]

raw_material_use_ratio, waste_per_unit_virgin_raw_material, ISHH_fresh_total_demand_data, P1H1_data, P1IS_data = [np.zeros(simulation_time, dtype=float) for _ in range(5)]

a_term_CLI_prod, a_prod_CLI_scaling_local, P1IS_dem_data, RPIS_dem_data = [np.zeros(simulation_time, dtype=float) for _ in range(4)]

# Circulation compartment
circular_material_use_rate, virgin_material_use_rate, IS_price_data, CLI_def, HHCLI, CLIHH, CLI_processed, CLI_price_data, CLIIRP = [np.zeros(simulation_time, dtype=float)
                                                                for _ in range(9)]
finished_goods_supply_CLI_data, finished_goods_supply_TI_data, CLI_premium_data, CLI_in, HHCLI_processing_deficit, \
CLI_out, CLI_processed, CLI_price_data, \
TI_def, EETI_dem_data, IS_def = [np.zeros(simulation_time, dtype=float)
          for _ in range(9+2)]

raw_material_CLI_def_data, price_difference_data, RPIS_dem_data, HHCLI_processing_available_data, \
HHCLI_processing_demand_data = [np.zeros(simulation_time, dtype=float) for _ in range(5)]

# additional information for analysis
P1_demand_humans_data, P1_supply_humans_data, P1_supply_stocks, P1_created \
    = [np.zeros(simulation_time, dtype=float) for _ in range(4)]
P1_demand_total_data, P1_supply_total_data, circulation_fraction, \
CLI_prod_data, TI_prod_data = [np.zeros(simulation_time, dtype=float) for _ in range(5)]
availability_for_wages_data, total_income, total_expenditure \
    = [np.zeros(simulation_time, dtype=float) for _ in range(3)]

percent_production_growth_CLI_data, production_cap_CLI_data, production_growth_CLI_data, availability_of_finished_goods_CLI_data, ISIRP, a_CLI_prod_growth_data, a_CLI_growth_multiplier \
    = [np.zeros(simulation_time, dtype=float) for _ in range(3+4)]

# special cases.
ratio_c_d_wages_data,\
uncirculated_used_goods_data,\
CLI_prod_min_data, \
CLI_prod_local_data, \
IS_flow_additional_P1, \
IS_flow_additional_P2, \
price_ratio_data,\
willingness_to_pay_CLI_data, \
finished_goods_demand_CLI_data, \
prod_cap_increase_fraction_data, \
relative_circulation = [np.zeros(simulation_time, dtype=float) for _ in range(7+4)]
# P1_demand_total_data, P1_supply_total_data = [np.zeros(simulation_time, dtype=float) for _ in range(2)]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# START copying readme data here
#  some case related information stored here
# these parameters should not be changed unless intended with thorough understanding of what is being done
period = 52
pi = 3.1416

# save_data = 1
# save_trends = 1

# defaults: DO NOT LEAVE SAVE WITH ALTERED VALUES
case_power_plant = 1
d_fee = 0
case_model_selection = 1
P1H1_dem_min = 0.000  # default 0
case_new_EE_parameters = 1  # set EE parameters based on Rodriguez-Gonzalez et al. (2018) default = 1
# changing 1 to 0 here, won't have any effect
# check series and variables file. search for the term Rodriguez-Gonzalez et al. (2018).
case_limit_pcm = 1  # default = 1
case_new_EEIS_computation = 1  # default = 1
case_case_high_IS_EE_dem = 1  # default = 1
P1IRP = 0
if case_limit_pcm == 1:
    limit_pcm_multiplier_upper = 2
    limit_pcm_multiplier_lower = 0.5

include_timestamp = 1

'''User inputs or manually inserted variables'''
simulation_time = int(10400)
# STOP copying the data here

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


'''
Parameter initialization
Initializing parameters and variables - from files
'''

# accessing file containing initial values
with open('initial.csv') as f:
    # reading the file into pandas-DataFrame
    d = pd.read_csv(f, comment='#', dtype={'Parameter_name': 'str',
                                           'Parameter_value': 'float64'})

initial = dict(zip(d.Parameter_name, d.Parameter_value))

# accessing file containing parameters
with open('parameters.csv') as f:
    # reading the file into pandas-DataFrame
    d = pd.read_csv(f, comment='#', dtype={'Parameter_name': 'str',
                                           'Parameter_value': 'float64'})
parameter = dict(zip(d.Parameter_name, d.Parameter_value))

'''
Setting up initial state of variables
'''
# Setting up initial conditions of the state variables
# Compartment masses
P1[0] = initial['P1_0']
P2[0] = initial['P2_0']
P3[0] = initial['P3_0']
H1[0] = initial['H1_0']
H2[0] = initial['H2_0']
H3[0] = initial['H3_0']
C1[0] = initial['C1_0']
C2[0] = initial['C2_0']
IS_total[0] = initial['IS_0']
RP[0] = initial['RP_0']
HH[0] = 1.0
IRP[0] = initial['IRP_0']
N_HH[0] = initial['N_HH']
ES[0] = initial['ES_0']

# Deficits
P1H1_def[0] = initial['P1H1_def']
P1TI_def[0] = initial['P1TI_def']
P1HH_def[0] = initial['P1HH_def']
H1_def[0] = initial['H1HH_def']
TI_def[0] = initial['ISHH_def']
P1H2_data[0] = 0

# Updating the human mortality rates
# These are used in the population growth case
mHHset = []
mHHset.append(parameter['m_HH'])  # parameter['m_HH'] is mortality per week
for i in range(699):
    mHHset.append(mHHset[-1] - 1e-7)
# First ramp of 699 time steps

for i in range(300):
    mHHset.append(mHHset[-1] - 8e-8)
# Second ramp of 300 time steps (total time 1000)

for i in range(500):
    mHHset.append(mHHset[-1] - 5e-8)
# Third ramp of 500 time steps (total time 1500)

for i in range(1000):
    mHHset.append(mHHset[-1] - 1e-8)
# Fourth ramp of 1000 time steps (total time 2500)

for i in range(7900):
    mHHset.append(mHHset[-1])

# Modifying the birth rates eta_a,2.7139e-04 eta_b,1.0454e-04 eta_c,0.0167
eta_b_set = []
span = 4000
eta_b_step = 9 / span
x = - 4
for i in range(span):
    x += eta_b_step
    eta_b_set.append(1e-4 + (8 - 1) * 1e-4 / (1 + 10 ** (0.6021 - x)))

# Equations in original Matlab code have been modified suit Python programming
# while maintaining the values
# eta_b_step = (eta_b_set[-1] - eta_b_set[-1] * 0.61) / ((span2 + span) - span)  # old value

span2 = 2000
eta_b_step = (eta_b_set[-1] - eta_b_set[-1] * 0.76) / ((span2 + span) - span)  # New default

for i in range(span2):
    eta_b_set.append(eta_b_set[-1] + eta_b_step)

for i in range(4400):
    eta_b_set.append(eta_b_set[-1])

i = 0
rIRPP2 = parameter['r_IRPP2'] * (100 / (100 + (IRP[i]) ** 2)) * 5
rIRPP3 = parameter['r_IRPP3'] * (100 / (100 + (IRP[i]) ** 2))

P1_def_total[i] = P1H1_def[i] + P1TI_def[i] + P1HH_def[i]
pcm_ideal = HH[0] / N_HH[0]
per_capita_mass[0] = pcm_ideal

k_demand = 1

# Alternative option
# k_demand = 2.0/N_HH[0]

zP1HH_initial = parameter['z_P1HH'] * k_demand
zH1HH_initial = parameter['z_H1HH'] * k_demand
zISHH_initial = parameter['z_ISHH'] * k_demand

dP1HH_initial = parameter['d_P1HH'] * k_demand
dH1HH_initial = parameter['d_H1HH'] * k_demand
dISHH_initial = parameter['d_ISHH'] * k_demand

mP1HH_initial = parameter['m_P1HH'] * k_demand
mH1HH_initial = parameter['m_H1HH'] * k_demand
mISHH_initial = parameter['m_ISHH'] * k_demand

nP1HH_initial = parameter['n_P1HH'] * k_demand
nH1HH_initial = parameter['n_H1HH'] * k_demand
nISHH_initial = parameter['n_ISHH'] * k_demand

kP1HH_initial = parameter['k_P1HH'] * k_demand
kH1HH_initial = parameter['k_H1HH'] * k_demand
kISHH_initial = parameter['k_ISHH'] * k_demand

# Creating local parameters for Transfer coefficients from RP to P1-3
gRPP1Base = parameter['g_RPP1']
gRPP2Base = parameter['g_RPP2']
gRPP3Base = parameter['g_RPP3']

EE_price = 0
fuelcost = 0
wagecost = 0
EEproduction = 0  # Energy Produced at time t
EEHH_per_cap_demand = 0  # Amount of Energy demanded by the HH Compartment
EEHHmass = 0  # Amount of fuel that is used to produce energy to satisfy the demand of HH
EEISdemand = 0  # Amount of Energy demand by the IS industry
ESIRP = 0  # Amount of mass used for producing the energy (for both humans and IS) in mass units can be interpreted
# as mass transfer from fuel source to IRP
minimum_wages = 0

coef_cEE = 2000
aEE = parameter['a_price_P1']
bEE = parameter['b_price_P1']
cEE_multiplier = 12e-4  # default 1
cEE = coef_cEE * parameter['c_price_P1'] * cEE_multiplier

Energy_Mass = 15e-4  # default
#   These three parameters will appear in the demand equation of P1, H1 and IS

tP1HH_initial = kP1HH_initial
tH1HH_initial = kH1HH_initial
tISHH_initial = kISHH_initial

# New parameter values based on Rodriguez-Gonzalez et al. (2018)
# supplementary information. please note l_EE = m_EE, and o_EE = t_EE
# other parameter names are same.
EEHH_demand_modifier = 1
dEEHH_initial = 6e-08 * EEHH_demand_modifier
kEEHH_initial = 6e-08 * EEHH_demand_modifier
mEEHH_initial = 4e-08 * EEHH_demand_modifier
nEEHH_initial = 2e-08 * EEHH_demand_modifier
tEEHH_initial = 6e-08 * EEHH_demand_modifier
zEEHH_initial = 5.68e-05 * EEHH_demand_modifier

# STOP copying readme data here
#  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IS_price = 1
# P1_price = 1
# H1_price = 1
# EE_price = 1
# percapdem = - 1 / 3 * (
#         + 1 * zISHH_initial * nEEHH_initial * IS_price
#         + 1 * zISHH_initial * kEEHH_initial * P1_price
#         - 1 * tISHH_initial * EE_price * zEEHH_initial
#         + 3 * nISHH_initial * IS_price * zEEHH_initial
#         - 1 * kISHH_initial * P1_price * zEEHH_initial
#         - 3 * dISHH_initial * zEEHH_initial
#         + 1 * zISHH_initial * mEEHH_initial * H1_price
#         - 3 * zISHH_initial * tEEHH_initial * EE_price
#         + 3 * zISHH_initial * dEEHH_initial
#         - 1 * mISHH_initial * H1_price * zEEHH_initial
#         + 1 * mISHH_initial * H1_price
#         + 1 * zISHH_initial * kH1HH_initial * P1_price
#         - 3 * zISHH_initial * mH1HH_initial * H1_price
#         - 3 * nISHH_initial * IS_price
#         + 3 * dISHH_initial
#         + 1 * kISHH_initial * P1_price
#         + 1 * tISHH_initial * EE_price
#         - 3 * zP1HH_initial * dISHH_initial
#         - 1 * zP1HH_initial * kISHH_initial * P1_price
#         - 1 * zP1HH_initial * mISHH_initial * H1_price
#         + 3 * zP1HH_initial * nISHH_initial * IS_price
#         - 1 * zP1HH_initial * tISHH_initial * EE_price
#         + 3 * zISHH_initial * dP1HH_initial
#         - 3 * zISHH_initial * kP1HH_initial * P1_price
#         + 1 * zISHH_initial * mP1HH_initial * H1_price
#         + 1 * zISHH_initial * nP1HH_initial * IS_price
#         + 1 * zISHH_initial * tP1HH_initial * EE_price
#         - 3 * zH1HH_initial * dISHH_initial
#         - 1 * zH1HH_initial * kISHH_initial * P1_price
#         - 1 * zH1HH_initial * mISHH_initial * H1_price
#         + 3 * zH1HH_initial * nISHH_initial * IS_price
#         - 1 * zH1HH_initial * tISHH_initial * EE_price
#         + 3 * zISHH_initial * dH1HH_initial
#         + 1 * zISHH_initial * nH1HH_initial * IS_price
#         + 1 * zISHH_initial * tH1HH_initial * EE_price) / (-1 +
#                                                            zP1HH_initial +
#                                                            zH1HH_initial +
#                                                            zISHH_initial +
#                                                            zEEHH_initial)
#
#
#
#
#
#
#
#
# for count, ix in enumerate([+ 1 * zISHH_initial * nEEHH_initial * IS_price,
#            + 1 * zISHH_initial * kEEHH_initial * P1_price,
#            - 1 * tISHH_initial * EE_price * zEEHH_initial,
#            + 3 * nISHH_initial * IS_price * zEEHH_initial,
#            - 1 * kISHH_initial * P1_price * zEEHH_initial,
#            - 3 * dISHH_initial * zEEHH_initial,
#            + 1 * zISHH_initial * mEEHH_initial * H1_price,
#            - 3 * zISHH_initial * tEEHH_initial * EE_price,
#            + 3 * zISHH_initial * dEEHH_initial,
#            - 1 * mISHH_initial * H1_price * zEEHH_initial,
#            + 1 * mISHH_initial * H1_price,
#            + 1 * zISHH_initial * kH1HH_initial * P1_price,
#            - 3 * zISHH_initial * mH1HH_initial * H1_price,
#            - 3 * nISHH_initial * IS_price,
#            + 3 * dISHH_initial,
#            + 1 * kISHH_initial * P1_price,
#            + 1 * tISHH_initial * EE_price,
#            - 3 * zP1HH_initial * dISHH_initial,
#            - 1 * zP1HH_initial * kISHH_initial * P1_price,
#            - 1 * zP1HH_initial * mISHH_initial * H1_price,
#            + 3 * zP1HH_initial * nISHH_initial * IS_price,
#            - 1 * zP1HH_initial * tISHH_initial * EE_price,
#            + 3 * zISHH_initial * dP1HH_initial,
#            - 3 * zISHH_initial * kP1HH_initial * P1_price,
#            + 1 * zISHH_initial * mP1HH_initial * H1_price,
#            + 1 * zISHH_initial * nP1HH_initial * IS_price,
#            + 1 * zISHH_initial * tP1HH_initial * EE_price,
#            - 3 * zH1HH_initial * dISHH_initial,
#            - 1 * zH1HH_initial * kISHH_initial * P1_price,
#            - 1 * zH1HH_initial * mISHH_initial * H1_price,
#            + 3 * zH1HH_initial * nISHH_initial * IS_price,
#            - 1 * zH1HH_initial * tISHH_initial * EE_price,
#            + 3 * zISHH_initial * dH1HH_initial,
#            + 1 * zISHH_initial * nH1HH_initial * IS_price,
#            + 1 * zISHH_initial * tH1HH_initial * EE_price,
#            - 1 + zP1HH_initial + zH1HH_initial + zISHH_initial + zEEHH_initial]):
#     if abs(ix) < 0.5:
#         # print(ix)
#         # print('{:0=6.3f}e-12'.format(ix))
#         print('{:0=2d} \t {:0=5.3f}e-12'.format(count + 1, ix * 1e12))
#     else:
#         print('{:0=6.4f}'.format(ix))
