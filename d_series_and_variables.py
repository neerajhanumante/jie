import pandas as pd
import time, datetime

#START copying the data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# some case related information stored here
# these parameters should not be changed unless intended with thorough understanding of what is being done
period = 52
pi = 3.1416

save_data = 1
save_trends = 1
case_power_plant = 1
d_fee = 0
case_model_selection = 1

consumption_increase_factor = 1.1  # old default 1.1, maintained as from original model
consumption_increase_factor = 1.1  # default 1.1, maintained as from original model

P1H1_dem_min = 0.000  # default 0

# defaults: DO NOT LEAVE SAVE WITH ALTERED VALUES
case_new_EE_parameters = 1  # set EE parameters based on Rodriguez-Gonzalez et al. (2018) default = 1
# changing 1 to 0 here, won't have any effect
# check series and variables file. search for the term Rodriguez-Gonzalez et al. (2018).
case_limit_pcm = 1  # default = 1
case_new_EEIS_computation = 1  # default = 1
case_case_high_IS_EE_dem = 1  # default = 1
if case_limit_pcm == 1:
    limit_pcm_multiplier_upper = 2
    limit_pcm_multiplier_lower = 0.5

include_timestamp = 1

'''User inputs or manually inserted variables'''
simulation_time = int(10400)  # also check series and variables
start_time = time.time()
st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#STOP copying the data here
"""
Initialize empty pandas series for various variables -- datatype - float64
"""
# Auxillary data
stockRP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

mass_differential_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

sys_mass = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

EEIS_dem_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

EEHH_dem_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

ESIRP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

EEHH_per_capita_demand_data_uncorrected = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

inflows_P2 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

outflows_P2 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

inflows_P3 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

outflows_P3 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

mass_ecological_RP = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

food_waste_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

food_consumed_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Compartment variables/ stocks declaration
P1 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P2 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P3 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H1 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H2 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H3 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

C1 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

C2 = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

HH = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IS_total = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IS_fresh = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

ES_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

RP = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IRP = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Flows: economic dimension
P1HHdemand_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H1HHdemand_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

ISHHdemand_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

EEHH_per_capita_demand_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P1HH_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H1HH_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

EEHH_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IS_fresh_out_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P1H1_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P1IS_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Flows: Intercompartmental
P1H2_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P2H1_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P2H2_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P2H3_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P3H3_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H1C1_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H2C1_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H2C2_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H3C2_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Flows: mortality
P1RP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P2RP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P3RP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H1RP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H2RP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H3RP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

C1RP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

C2RP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

HHRP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Flows: Recycling and RP flows
IRPP2_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IRPP3_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IRPRP_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

RPP1_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

RPP2_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

RPP3_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

RPIS_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Economic variables: wages, and weighted price
wages_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

weightedprice_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Economic variables: prices
P1_price_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H1_price_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IS_price_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

EE_price_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Economic variables: production
P1_prod_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H1_prod_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IS_prod_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Economic variables: deficits
P1H1_def = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P1IS_def = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P1HH_def = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P1_def_total = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

H1_def = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IS_def = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Human population related variables
N_HH = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

per_capita_mass = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

percapbirths_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Circulation compartment data

CC = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CC_price_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CC_processed = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CC_diff = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

EECC_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

ISCC = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P1CC = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P1CC_def = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

RPCC = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CCIRP_fmcg = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CCIS_fmcg = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CCIRP_smcg = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CCIS_smcg = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CCIS_total = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CC_def = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CC_prod_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IS_out = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

P1CC_demand = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IS_FMCG = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IS_SMCG = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

FMCG_CC = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

SMCG_CC = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

FMCG_IRP = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

SMCG_IRP = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CC_in = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CC_out = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IRP_in = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IRP_out = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

IS_in = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

ISHHflow_data = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CC_inventory = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CC_processing = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

CCIRP = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

# Product life data

SMCG_storage = pd.Series(index=range(simulation_time + 1)).fillna(0.0).astype('float64')

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

#START copying the data



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

'''
Setting up initial state of variables
'''
# Setting up initial conditions of the state variables
# Compartment masses
# P1[0] = initial['P1_0'] /1.2 * 3
P1[0] = initial['P1_0']
# P1[0] = initial['P1_0'] * 2
P2[0] = initial['P2_0']
P3[0] = initial['P3_0']
H1[0] = initial['H1_0']
H2[0] = initial['H2_0']
H3[0] = initial['H3_0']
C1[0] = initial['C1_0']
C2[0] = initial['C2_0']
IS_total[0] = initial['IS_0']
RP[0] = initial['RP_0']
# HH[0] = 1.3
HH[0] = 1.0
# HH[0] = 1.75
IRP[0] = initial['IRP_0']
N_HH[0] = initial['N_HH']

# Deficits
P1H1_def[0] = initial['P1H1_def']
P1IS_def[0] = initial['P1IS_def']
P1HH_def[0] = initial['P1HH_def']
H1_def[0] = initial['H1HH_def']
IS_def[0] = initial['ISHH_def']
P1H2_data[0] = 0

# In[]:
# Updating the human mortality rates
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
# plot_variable = mHHset
# plt.plot(plot_variable)
# plt.show()
#
# sys.exit()

# Modifying the birth rates eta_a,2.7139e-04 eta_b,1.0454e-04 eta_c,0.0167
eta_b_set = []
span = 4000
eta_b_step = 9 / span
x = - 4
for i in range(span):
    x += eta_b_step
    eta_b_set.append(1e-4 + (8 - 1) * 1e-4 / (1 + 10 ** (0.6021 - x)))

# print(eta_b_set[-4:])
#
# Equations in original Matlab code have been modified suit Python programming
# while maintaining the values

span2 = 2000
# eta_b_step = (eta_b_set[-1] - eta_b_set[-1] * 0.61) / ((span2 + span) - span)  # old value
eta_b_step = (eta_b_set[-1] - eta_b_set[-1] * 0.76) / ((span2 + span) - span)  # New default

for i in range(span2):
    eta_b_set.append(eta_b_set[-1] + eta_b_step)

for i in range(4400):
    eta_b_set.append(eta_b_set[-1])

i = 0
rIRPP2 = parameter['r_IRPP2'] * (100 / (100 + (IRP[i]) ** 2)) * 5
rIRPP3 = parameter['r_IRPP3'] * (100 / (100 + (IRP[i]) ** 2))

P1_def_total[i] = P1H1_def[i] + P1IS_def[i] + P1HH_def[i]
pcm_ideal = HH[0] / N_HH[0]
per_capita_mass[0] = pcm_ideal

# K_demand calculation --- due to some reason, multiplication with k_demand does not change coefficients
# This is really deviant behaviour of Matlab code reason of which is unknown right now.
# Won't spend time on this unless required.
# Hence following formula has been commented. Value of k_demand is set to 1

# k_demand = 2.0/N_HH[0]

k_demand = 1

zP1HH = parameter['z_P1HH'] * k_demand
zH1HH = parameter['z_H1HH'] * k_demand
zISHH = parameter['z_ISHH'] * k_demand

dP1HH = parameter['d_P1HH'] * k_demand
dH1HH = parameter['d_H1HH'] * k_demand
dISHH = parameter['d_ISHH'] * k_demand

mP1HH = parameter['m_P1HH'] * k_demand
mH1HH = parameter['m_H1HH'] * k_demand
mISHH = parameter['m_ISHH'] * k_demand

nP1HH = parameter['n_P1HH'] * k_demand
nH1HH = parameter['n_H1HH'] * k_demand
nISHH = parameter['n_ISHH'] * k_demand

kP1HH = parameter['k_P1HH'] * k_demand
kH1HH = parameter['k_H1HH'] * k_demand
kISHH = parameter['k_ISHH'] * k_demand

# Magnitude of coefficient change in demand function
aISp = 0.3109
bISp = 0.0044
cISp = 0.3313

# Creating local parameters for Transfer coefficients from RP to P1-3
gRPP1Base = parameter['g_RPP1']
gRPP2Base = parameter['g_RPP2']
gRPP3Base = parameter['g_RPP3']

rise_coefficient = 1

gRPP1rise = gRPP1Base * rise_coefficient
gRPP2rise = gRPP2Base * rise_coefficient
gRPP3rise = gRPP3Base * rise_coefficient

# pEE = 0 # This declaration is not necessary since now price_variables['EE_price'][0] is used
EE_price = 0
fuelcost = 0
wagecost = 0
EEproduction = 0  # Energy Produced at time t
EEHHdemand = 0  # Amount of Energy demanded by the HH Compartment
EEHHmass = 0  # Amount of fuel that is used to produce energy to satisfy the demand of HH
EEISdemand = 0  # Amount of Energy demand by the IS industry
ESIRP = 0  # Amount of mass used for producing the energy (for both humans and IS) in mass units can be interpreted
# as mass transfer from fuel source to IRP
# Not required delete the following EEHH declarations and t coefficients declarations
# tP1HH = 0
# tH1HH = 0
# tISHH = 0
# tEEHH = 0
# zEEHH = 0
# dEEHH = 0
# kEEHH = 0
# mEEHH = 0
# nEEHH = 0
minimum_wages = 0

coef_cEE = 2000

# earlier values
# if case_consumption_increase == 1:
#     coef_cEE = 13000
# elif case_population_explosion == 1:
#     coef_cEE = 3400
# else:
#     coef_cEE = 5000

aEE = parameter['a_price_P1']
bEE = parameter['b_price_P1']
cEE_multiplier = 12e-4  # default 1
cEE = coef_cEE * parameter['c_price_P1'] * cEE_multiplier

# Energy_Mass = 15e-3  # test
Energy_Mass = 15e-4  # default
ES = initial['ES_0']  # default 800
#   These three parameters will appear in the demand equation of P1, H1 and IS

tP1HH = kP1HH
tH1HH = kH1HH
tISHH = kISHH

#   These six paramters will appear in the demand equation of EE,
# delete these later, when confirmed that these will not be used anymore.

tEEHH = kP1HH
zEEHH = zP1HH
dEEHH = dP1HH
kEEHH = kP1HH
mEEHH = mP1HH
nEEHH = nP1HH

# New parameter values based on Rodriguez-Gonzalez et al. (2018)
# supplementary information. please note l_EE = m_EE, and o_EE = t_EE
# other parameter names are same.

dEEHH = 6e-08
kEEHH = 6e-08
mEEHH = 4e-08
nEEHH = 2e-08
tEEHH = 6e-08
zEEHH = 5.68e-05

eta_a = 0.90462 * 3e-4
eta_b = 1.0454e-4


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#STOP copying the data here