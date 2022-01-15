from inspect import currentframe, getframeinfo
import pandas as pd
import math, itertools, csv
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import shutil
from os import path, getcwd, listdir, mkdir
from os.path import join, expanduser
from b_aux_functions import prod_multiplier_calculation_a_CLI
import pathlib
import numpy as np
import matplotlib

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=24)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

# Importing user-defined modules
from c_initialization_and_series import *
from b_aux_functions import flow_computation, wtp_literature_segments, wtp_literature_linear

consumption_lower_limit = 1  # default 1
case_list = [[1, 1]]  # default 1. 1 consumption increase and population growth both
# case_list = [[0, 0]]  # no consumption increase, no population growth scenario

case_circulation = 0
case_circulation = 1  # default 1

# ----------------------------------------------------------------------------------------------------
# Setting up parameters
# ----------------------------------------------------------------------------------------------------

# scaling factors
HHCLI_threshold_flow = 2000 * 10**-8
prod_cap_increase_period_years = 3  # default 3
kappa = 1  # default 1
a_price_CLI_scaling = 1.25  # default 1.25
years_CLI_development = 30
# a_price_CLI_scaling = 2.5  # default 1.25
# years_CLI_development = 60
assert prod_cap_increase_period_years == 3
assert kappa == 1
assert a_price_CLI_scaling == 1.25
assert years_CLI_development == 30

CLI_initial_scaling_price = a_price_CLI_scaling
CLI_final_scaling_price = 1.0
tolerance_at_end_of_dev = 0.05
# exponential decay function below:
decay_constant = np.log(1/tolerance_at_end_of_dev)/(52 * years_CLI_development)
CLI_a_modifier_price_list = [CLI_final_scaling_price +
                             (CLI_initial_scaling_price -
                              CLI_final_scaling_price) *
                             np.exp(-decay_constant * t) for t in
                             range(simulation_time)]


prod_cap_increase_period = int(prod_cap_increase_period_years * 52)
a_prod_CLI_scaling = 1/2200  # default 1 # earlier 1/2520
a_prod_CLI_scaling_local[i] = a_prod_CLI_scaling
b_prod_CLI_scaling = 1/100  # default 1 # earlier test 1/100
c_prod_CLI_scaling = a_prod_CLI_scaling  # default 1/2500 # earlier test 1/2000
c_prod_CLI_scaling = 1/2500  # default 1/2500 # earlier test 1/2000

b_price_CLI_scaling = 1  # default 1
c_price_CLI_scaling = 1  # default 1

case_wtp_modeling = 1  # default 1
# case_wtp_modeling = 0

size_CLI = 0.17
# size_CLI = 0.7
circulation_efficiency = 0.85  # default 0.85
# size_CLI = 0


max_circulation = 0.9
processing_time_CLI = 10  # default 10 - simulation_time
circulation_energy_savings = 0.7  # default 0.7


# ----------------------------------------------------------------------------------------------------
# Configuration for different output figures
# ----------------------------------------------------------------------------------------------------

# Figure 3
consumption_level_list = [6]  # default 6
[A, B] = [20, -72]  # consumption level 6 CM1 limit
[A, B] = [20, -98]  # consumption level 6 CM1 limit
[A, B] = [20, -70]  # consumption level 6 no collapse


# Figure 5
# consumption_level_list = [6]  # range 4, 6, 8, 10
# [A, B] = [ , ]  # get collapse mode 1 and 2 limits for the entire simulation matrix
# For each point, take A and find limiting values of B and vice versa

# Figure 6
consumption_level_list = [6]  # default 6
[A, B] = [30, -113]  # consumption level 6 CM1 limit
# consumption_level_list = [8]  # default 8
# [A, B] = [60, -113]  # consumption level 8 CM1 limit



save_data = 1
# save_data = 0
assert save_data in [0, 1]

print('consumption = {}'.format(consumption_level_list[0]))
print('kappa = {}'.format(kappa))
print('min WTP = {}\nmax WTP = {}'.format(A, B))
wtp_dict = wtp_literature_linear([B, A])

case_6th_wave = 1
case_6th_wave = 0
if case_6th_wave == 1:
    print('CLI prod growth with 6th wave modelled')
case_overall_limit_CWGR = 1  # compunded weekly growth rate default 1
# case_overall_limit_CWGR = 0  # compunded weekly growth rate
assert case_overall_limit_CWGR in [1]
assert [case_overall_limit_CWGR, case_6th_wave] != [1, 1]
assert case_6th_wave in [0, 1]
if case_overall_limit_CWGR == 1:
    case_6th_wave = 0
    upper_limit_CWGR = (1 + 5/100) ** (1/52)  # 0.0009387127031117437,
    # ratio of production capacities should not exceed this value

case_constant_wtp = 1
case_constant_wtp = 0
if case_constant_wtp == 1:
    percent_population_constant = 14  # default size_CLI * circulation_efficiency * 100
    percent_population_constant = size_CLI * circulation_efficiency * 100  # default size_CLI * circulation_efficiency * 100
    percent_population_constant = 80  # default 14
    # percent_population_constant = 75  # default 14  CM2 lower limit 75
    # percent_population_constant = 41  # default 14  CM1 upper limit 41

    percent_population_constant = min(100, percent_population_constant)
    price_independent = [[-70, percent_population_constant], [35, percent_population_constant]]

    wtp_dict = wtp_literature_segments(price_independent)
    print('This is constant WTP = {}'.format(percent_population_constant))
if case_list == [[0, 0]]:
    wtp_dict = wtp_literature_segments(price_independent)

case_additional_supply_price_independent = 0  # default 0  # only for testing, delete later
case_additional_supply_price_independent = 1  # default 0  # only for testing, delete later

case_reproduce_paper_1 = 1
case_reproduce_paper_1 = 0  # default 0


CLI_bar = size_CLI * IS_total[0] * 0.2
# CLI_bar = 0
if case_reproduce_paper_1 == 1:
    CLI_bar = 0
    circulation_fraction_input = 0.17

# Creating list of cases for which simulations need to be carried out
case_list_all = []
args = [case_list, consumption_level_list]
# Creating an interim list of cases with increased consumption level
case_list_all_interim = [itertools.product(*[[x], consumption_level_list]) for x in case_list if x[1] == 1]
case_list_all = [list(x) for y in case_list_all_interim for x in list(y)]
# Adding cases without consumption increase
case_list_all.extend([x] for x in case_list if x[1] == 0)

innovation_cycle_period = 60  # years default 60
growth_rate_CAGR = 5  # % percent default 5

prod_cap_max = (growth_rate_CAGR/100 + 1) ** innovation_cycle_period
CLI_multiplier_prod_list = prod_multiplier_calculation_a_CLI(x=np.linspace(0,
                                                                           simulation_time + 1,
                                                                           simulation_time + 1),
                                                             center=int(innovation_cycle_period *
                                                                        0.5 * 52),
                                                             width=int(innovation_cycle_period *
                                                                       0.2 * 52),
                                                             prod_cap_local=prod_cap_max)

# Case list loop
for case_index, case in enumerate(case_list_all):
    # STOP copying readme data here
    case_population_explosion, case_consumption_increase = case[0]
    if case_circulation == 1:
        TI_inventory[0] = IS_total[0] * (1 - size_CLI)
        CLI_inventory[0] = IS_total[0] * size_CLI
    else:
        TI_inventory[0] = IS_total[0]

    if case_circulation == 1:
        additional_name_tags = 'with circulation'
    else:
        additional_name_tags = 'no circulation'

    # case specific changes
    if case_population_explosion == 1:
        eta_a = parameter['eta_a'] * 2  # default
    else:
        eta_a = parameter['eta_a']  # default

    start_time_case_float = time.time()
    start_time_case_YmdHMS = datetime.datetime.fromtimestamp(start_time_case_float).strftime('%Y-%m-%d_%H:%M:%S')

    zP1HHSet, zH1HHSet, zISHHSet, zEEHHSet = [], [], [], []

    dP1HHSet, dH1HHSet, dISHHSet, dEEHHSet = [], [], [], []

    mP1HHSet, mH1HHSet, mISHHSet, mEEHHSet = [], [], [], []

    nP1HHSet, nH1HHSet, nISHHSet, nEEHHSet = [], [], [], []

    kP1HHSet, kH1HHSet, kISHHSet, kEEHHSet = [], [], [], []

    tP1HHSet, tH1HHSet, tISHHSet, tEEHHSet = [], [], [], []

    gRPP3_list = []
    gRPP1_list = []

    switch_variable_P1 = 0
    extinction_timestep_P1 = 10401
    switch_variable_P2 = 0
    extinction_timestep_P2 = 10401
    switch_variable_P3 = 0
    extinction_timestep_P3 = 10401
    switch_variable_H1 = 0
    extinction_timestep_H1 = 10401
    switch_variable_H2 = 0
    extinction_timestep_H2 = 10401
    switch_variable_H3 = 0
    extinction_timestep_H3 = 10401
    switch_variable_C1 = 0
    extinction_timestep_C1 = 10401
    switch_variable_C2 = 0
    extinction_timestep_C2 = 10401
    switch_variable_IS = 0
    extinction_timestep_IS = 10401
    switch_variable_ES = 0
    extinction_timestep_ES = 10401

    # implementing changes related to the increased consumption case
    # c_demand = consumption_level_list[consumption_level_index]
    c_demand = 1
    if case_consumption_increase == 1:
        c_demand = case[1]
        consumption_upper_limit = c_demand

        demand_multiplier = [consumption_lower_limit +
                             (consumption_upper_limit -
                              consumption_lower_limit) * i / simulation_time for i in
                             range(simulation_time)]

        zP1HHSet = [zP1HH_initial] * simulation_time
        zH1HHSet = [zH1HH_initial] * simulation_time
        zISHHSet = [initial_value * zISHH_initial for initial_value in demand_multiplier]
        zEEHHSet = [initial_value * zEEHH_initial for initial_value in demand_multiplier]

        dP1HHSet = [dP1HH_initial] * simulation_time
        dH1HHSet = [dH1HH_initial] * simulation_time
        dISHHSet = [initial_value * dISHH_initial for initial_value in demand_multiplier]
        dEEHHSet = [initial_value * dEEHH_initial for initial_value in demand_multiplier]

        mP1HHSet = [mP1HH_initial] * simulation_time
        mH1HHSet = [mH1HH_initial] * simulation_time
        mISHHSet = [initial_value * mISHH_initial for initial_value in demand_multiplier]
        mEEHHSet = [initial_value * mEEHH_initial for initial_value in demand_multiplier]

        nP1HHSet = [nP1HH_initial] * simulation_time
        nH1HHSet = [nH1HH_initial] * simulation_time
        nISHHSet = [initial_value * nISHH_initial for initial_value in demand_multiplier]
        nEEHHSet = [initial_value * nEEHH_initial for initial_value in demand_multiplier]

        kP1HHSet = [kP1HH_initial] * simulation_time
        kH1HHSet = [kH1HH_initial] * simulation_time
        kISHHSet = [initial_value * kISHH_initial for initial_value in demand_multiplier]
        kEEHHSet = [initial_value * kEEHH_initial for initial_value in demand_multiplier]

        tP1HHSet = [tP1HH_initial] * simulation_time
        tH1HHSet = [tH1HH_initial] * simulation_time
        tISHHSet = [initial_value * tISHH_initial for initial_value in demand_multiplier]
        tEEHHSet = [initial_value * tEEHH_initial for initial_value in demand_multiplier]

    # file/directory names updates
    if case_population_explosion == 1 and case_consumption_increase == 0:
        case_name = "Population Growth"
    elif case_consumption_increase == 1 and case_population_explosion == 0:
        case_name = "Consumption Increase"
    elif case_consumption_increase == 0 and case_population_explosion == 0:
        case_name = "Base Case"
    elif case_consumption_increase == 1 and case_population_explosion == 1:
        case_name = "Both cases simultaneously"

    print('this is {} \t consumption {} {} '.format(case_name, c_demand, additional_name_tags))
    print("case {} of {}".format(case_index + 1, len(case_list_all)))

    working_switch_CLI = 0

    # Simulation of each case over the simulation time begins here
    # Step 1
    for i in range(0, simulation_time):

        Q1 = 1 + parameter['A_m'] * math.sin(2 * pi * i / period - pi / 2) * 1
        Q2 = 1 + parameter['A_m'] * math.sin(2 * pi * i / period - pi / 2) * 5
        gRPP1 = gRPP1Base * Q1
        gRPP2 = gRPP2Base * Q1
        gRPP3 = gRPP3Base * Q2
        gRPP3_list.append(gRPP3)
        gRPP1_list.append(gRPP1)

        if case_population_explosion == 1 and i < simulation_time:
            eta_b = eta_b_set[i]
            mHH = mHHset[i]
        elif i < simulation_time:
            eta_b = parameter['eta_b']
            mHH = parameter['m_HH']

        if case_consumption_increase == 1:
            zP1HH = zP1HHSet[i]
            zH1HH = zH1HHSet[i]
            zISHH = zISHHSet[i]
            zEEHH = zEEHHSet[i]

            dP1HH = dP1HHSet[i]
            dH1HH = dH1HHSet[i]
            dISHH = dISHHSet[i]
            dEEHH = dEEHHSet[i]

            mP1HH = mP1HHSet[i]
            mH1HH = mH1HHSet[i]
            mISHH = mISHHSet[i]
            mEEHH = mEEHHSet[i]

            nP1HH = nP1HHSet[i]
            nH1HH = nH1HHSet[i]
            nISHH = nISHHSet[i]
            nEEHH = nEEHHSet[i]

            kP1HH = kP1HHSet[i]
            kH1HH = kH1HHSet[i]
            kISHH = kISHHSet[i]
            kEEHH = kEEHHSet[i]

            tP1HH = tP1HHSet[i]
            tH1HH = tH1HHSet[i]
            tISHH = tISHHSet[i]
            tEEHH = tEEHHSet[i]
        else:
            zP1HH = zP1HH_initial
            zH1HH = zH1HH_initial
            zISHH = zISHH_initial
            zEEHH = zEEHH_initial

            dP1HH = dP1HH_initial
            dH1HH = dH1HH_initial
            dISHH = dISHH_initial
            dEEHH = dEEHH_initial

            mP1HH = mP1HH_initial
            mH1HH = mH1HH_initial
            mISHH = mISHH_initial
            mEEHH = mEEHH_initial

            nP1HH = nP1HH_initial
            nH1HH = nH1HH_initial
            nISHH = nISHH_initial
            nEEHH = nEEHH_initial

            kP1HH = kP1HH_initial
            kH1HH = kH1HH_initial
            kISHH = kISHH_initial
            kEEHH = kEEHH_initial

            tP1HH = tP1HH_initial
            tH1HH = tH1HH_initial
            tISHH = tISHH_initial
            tEEHH = tEEHH_initial

        #   Step 2
        #   Wage calculation Eq 2
        # In case circulation is present the TI inventory is less than the total IS inventory,
        # thus computed here wages are on higher side. Following equation is useful only when circulation is not
        # considered.

        availability_of_finished_goods_total = (+ TI_def[i] +
                                                TI_inventory[i] -
                                                parameter['TI_bar'])/(parameter['lambda_RP'] +
                                                                      parameter['theta_P1'])

        wages = max(parameter['a_wage'] -
                    parameter['c_wage'] * availability_of_finished_goods_total -
                    parameter['d_wage'] * N_HH[i],

                    minimum_wages)
        if case_circulation == 1:
            availability_of_finished_goods_total = (CLI_inventory[i] - CLI_bar +
                                                    TI_inventory[i] - parameter['TI_bar'] +
                                                    IS_def[i]) / (parameter['lambda_RP'] +
                                                                  parameter['theta_P1'])
            wages = max(parameter['a_wage'] -
                        parameter['c_wage'] * availability_of_finished_goods_total -
                        parameter['d_wage'] * N_HH[i],

                        minimum_wages)
            availability_for_wages_data[i] = availability_of_finished_goods_total

        ratio_c_d_wages_data[i] = parameter['c_wage'] * availability_of_finished_goods_total / (parameter['d_wage'] *
                                                                                                N_HH[i])

        # Determination of P1, H1, IS, EE prices and production targets for P1, and H1

        # Step 3
        #   Price calculation of P1 Eq 13 + Production calculation of P1 Eq 14

        if P1[i] <= 0:
            P1_price = 0
            P1_prod = 0
        else:

            availability_of_P1_goods = P1[i] - parameter['P1_bar'] + P1_def_total[i]

            P1_price = max(parameter['a_price_P1'] +
                           parameter['b_price_P1'] * wages -
                           parameter['c_price_P1'] * availability_of_P1_goods,

                           parameter['msv'])

            P1_prod = max(parameter['a_prod_P1'] -
                          parameter['b_prod_P1'] * wages -
                          parameter['c_prod_P1'] * availability_of_P1_goods,

                          parameter['msv'])

        # Step 4
        #   Price calculation of H1 Eq 32 + Production calculation of H1 Eq 33

        if H1[i] <= 0:
            H1_price = 0
            H1_prod = 0
        else:
            availability_of_H1_goods = H1[i] - parameter['H1_bar'] + H1_def[i]

            H1_price = max(parameter['a_price_H1'] +
                           parameter['b_price_H1'] * wages -
                           parameter['c_price_H1'] * availability_of_H1_goods,

                           parameter['msv'])

            H1_prod = max(parameter['a_prod_H1'] -
                          parameter['b_prod_H1'] * wages -
                          parameter['c_prod_H1'] * availability_of_H1_goods,
                          parameter['msv'])

        if ES[i] > 0:
            EE_price = max(aEE + bEE * wages + cEE / ES[i], 0)
            # prices are used in per capita demand computation
        else:
            EE_price = 0

            # There is a statement to compute ISHH demand at i, however it is unused -- line 519 Matlab code
        other_prices_effect_numerator_IS_price = \
            (- dISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] +
                                                                        parameter['theta_P1']) / (-1 +
                                                                                                  zH1HH +
                                                                                                  zISHH +
                                                                                                  zP1HH)
             - mISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] +
                                                                        parameter['theta_P1']) / (-1 +
                                                                                                  zH1HH +
                                                                                                  zISHH +
                                                                                                  zP1HH) * H1_price
             - kISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] +
                                                                        parameter['theta_P1']) / (-1 +
                                                                                                  zH1HH +
                                                                                                  zISHH +
                                                                                                  zP1HH) * P1_price
             + dISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] +
                                                                        parameter['theta_P1']) / (-1 +
                                                                                                  zH1HH +
                                                                                                  zISHH +
                                                                                                  zP1HH) * zH1HH
             + mISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                        parameter['theta_P1']) / (-1 + 
                                                                                                  zH1HH + 
                                                                                                  zISHH + 
                                                                                                  zP1HH) * P1_price * zH1HH
             + kISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                        parameter['theta_P1']) / (-1 + 
                                                                                                  zH1HH + 
                                                                                                  zISHH + 
                                                                                                  zP1HH) * P1_price * zH1HH
             - dH1HH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                        parameter['theta_P1']) / (-1 +  
                                                                                                  zH1HH +  
                                                                                                  zISHH + 
                                                                                                  zP1HH) * zISHH
             - dP1HH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                        parameter['theta_P1']) / (-1 + 
                                                                                                  zH1HH + 
                                                                                                  zISHH + 
                                                                                                  zP1HH) * zISHH
             + mH1HH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                        parameter['theta_P1']) / (-1 + 
                                                                                                  zH1HH + 
                                                                                                  zISHH + 
                                                                                                  zP1HH) * H1_price * zISHH
             - mP1HH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                        parameter['theta_P1']) / (-1 + 
                                                                                                  zH1HH + 
                                                                                                  zISHH + 
                                                                                                  zP1HH) * H1_price * zISHH
             - kH1HH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                        parameter['theta_P1']) / (-1 + 
                                                                                                  zH1HH + 
                                                                                                  zISHH + 
                                                                                                  zP1HH) * P1_price * zISHH
             + kP1HH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                        parameter['theta_P1']) / (-1 + 
                                                                                                  zH1HH + 
                                                                                                  zISHH + 
                                                                                                  zP1HH) * P1_price * zISHH
             + dISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                        parameter['theta_P1']) / (-1 + 
                                                                                                  zH1HH + 
                                                                                                  zISHH + 
                                                                                                  zP1HH) * zP1HH
             + mISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                        parameter['theta_P1']) / (-1 + 
                                                                                                  zH1HH + 
                                                                                                  zISHH + 
                                                                                                  zP1HH) * H1_price * zP1HH
             + kISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                        parameter['theta_P1']) / (-1 + 
                                                                                                  zH1HH + 
                                                                                                  zISHH + 
                                                                                                  zP1HH) * P1_price * zP1HH)

        other_prices_effect_denomenator_IS_price = (
                1 -
                nISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                         parameter['theta_P1']) / (-1 + 
                                                                                                   zH1HH + 
                                                                                                   zISHH + 
                                                                                                   zP1HH) + 
                nISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                         parameter['theta_P1']) / (-1 + 
                                                                                                   zH1HH + 
                                                                                                   zISHH + 
                                                                                                   zP1HH) * zH1HH + 
                nH1HH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                         parameter['theta_P1']) / (-1 + 
                                                                                                   zH1HH + 
                                                                                                   zISHH + 
                                                                                                   zP1HH) * zISHH + 
                nP1HH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                         parameter['theta_P1']) / (-1 + 
                                                                                                   zH1HH + 
                                                                                                   zISHH + 
                                                                                                   zP1HH) * zISHH + 
                nISHH * parameter['d_price_IS'] * parameter['p_ISHH'] * (parameter['lambda_RP'] + 
                                                                         parameter['theta_P1']) / (-1 + 
                                                                                                   zH1HH + 
                                                                                                   zISHH + 
                                                                                                   zP1HH) * zP1HH)

        availability_of_finished_goods_TI = (+ TI_def[i] +
                                             TI_inventory[i] -
                                             parameter['TI_bar']) / (parameter['lambda_RP'] +
                                                                     parameter['theta_P1'])
        p_ISbar = (parameter['a_price_IS'] +
                   parameter['b_price_IS'] * wages -
                   parameter['c_price_IS'] * availability_of_finished_goods_TI +
                   other_prices_effect_numerator_IS_price) / other_prices_effect_denomenator_IS_price

        TI_price = max(p_ISbar, parameter['msv'])
        IS_price = TI_price

        if case_circulation == 1:
            # Prices of other goods do not affect circulation goods price
            # Coefficients of circulation goods price are modified
            availability_of_finished_goods_CLI = (CLI_inventory[i] +
                                                  CLI_def[i] -
                                                  CLI_bar) / (parameter['lambda_RP'] +
                                                              parameter['theta_P1'])
            availability_of_finished_goods_CLI_data[i] = availability_of_finished_goods_CLI

            a_price_CLI_scaling_local = CLI_a_modifier_price_list[i]

            CLI_price = max(parameter['a_price_IS'] * a_price_CLI_scaling_local +
                            parameter['b_price_IS'] * wages * b_price_CLI_scaling -
                            parameter['c_price_IS'] * availability_of_finished_goods_CLI * c_price_CLI_scaling,

                            parameter['msv'])

            assert TI_price > 0
            price_ratio = TI_price/CLI_price
            # willingness_to_pay_TI ==> what fraction of population is ready to buy TI at that price
            if case_wtp_modeling == 0:
                if price_ratio <= 1:
                    willingness_to_pay_CLI = size_CLI * circulation_efficiency
                else:
                    willingness_to_pay_CLI = 1
            if case_wtp_modeling == 1:
                assert price_ratio > 0, 'check price ratio'
                CLI_premium = 100 * (CLI_price - TI_price)/TI_price
                CLI_premium_data[i] = CLI_premium

                if CLI_premium > max(wtp_dict.keys()):  # premium more than nobody is willing to buy
                    willingness_to_pay_CLI = min(wtp_dict.values())
                elif CLI_premium < min(wtp_dict.keys()):  # premium more than everybody is willing to buy
                    willingness_to_pay_CLI = max(wtp_dict.values())
                else:
                    willingness_to_pay_CLI = wtp_dict[
                        min(wtp_dict.keys(), key=lambda x: abs(x - CLI_premium_data[i]))]

                # willingness_to_pay_CLI = max(willingness_to_pay_CLI, size_CLI * circulation_efficiency)
                # willingness_to_pay_CLI = min(willingness_to_pay_CLI, 1)
            willingness_to_pay_CLI = max(willingness_to_pay_CLI, size_CLI * circulation_efficiency)
            # willingness_to_pay_CLI = min(willingness_to_pay_CLI, 0.75)

            if case_constant_wtp == 1:
                willingness_to_pay_CLI = percent_population_constant/100

            willingness_to_pay_TI = 1 - willingness_to_pay_CLI

            willingness_to_pay_CLI_data[i] = willingness_to_pay_CLI
            # price_ratio_data[i] = price_ratio

            # This IS price is used to determine the demand for the consumer goods.

            # IS_price = willingness_to_pay_CLI * CLI_price + willingness_to_pay_TI * TI_price
            if TI_inventory[i] + CLI_inventory[i] > 0:
                # IS price is weighted average of the two subcompartments
                IS_price = (TI_price * TI_inventory[i] +
                            CLI_price * CLI_inventory[i]) / (TI_inventory[i] +
                                                             CLI_inventory[i])
            else:
                IS_price = TI_price_data[i]

            if case_reproduce_paper_1 == 1:
                if TI_inventory[i] + CLI_inventory[i] > 0:
                    # IS price is weighted average of the two subcompartments
                    IS_price = (TI_price * TI_inventory[i] +
                                CLI_price * CLI_inventory[i]) / (TI_inventory[i] +
                                                                 CLI_inventory[i])
                else:
                    IS_price = TI_price_data[i]

        if HH[i] == 0 or N_HH[i] < 2000:
            TI_prod = 0
            TI_price = 0
            CLI_prod = 0
            CLI_price = 0
            IS_price = 0

        # Determination of per capita demand of P1, H1, IS and EE by humans

        P1HH_per_cap_demand = + 1 / 3 * (
                - 1 * tP1HH * EE_price
                - 3 * dP1HH
                + 3 * kP1HH * P1_price
                - 1 * mP1HH * H1_price
                + 1 * zH1HH * nP1HH * IS_price
                + 1 * zH1HH * tP1HH * EE_price
                - 1 * nP1HH * IS_price
                - 3 * zP1HH * dH1HH
                - 1 * zP1HH * kH1HH * P1_price
                + 3 * zP1HH * mH1HH * H1_price
                - 1 * zP1HH * nH1HH * IS_price
                - 1 * zP1HH * tH1HH * EE_price
                + 3 * zH1HH * dP1HH
                - 3 * zH1HH * kP1HH * P1_price
                + 1 * zH1HH * mP1HH * H1_price
                - 3 * zP1HH * dISHH
                - 1 * zP1HH * kISHH * P1_price
                - 1 * zP1HH * mISHH * H1_price
                + 3 * zP1HH * nISHH * IS_price
                - 1 * zP1HH * tISHH * EE_price
                + 3 * zISHH * dP1HH
                - 3 * zISHH * kP1HH * P1_price
                + 1 * zISHH * mP1HH * H1_price
                + 1 * zISHH * nP1HH * IS_price
                + 1 * zISHH * tP1HH * EE_price
                - 3 * zP1HH * dEEHH
                - 1 * zP1HH * kEEHH * P1_price
                - 1 * zP1HH * mEEHH * H1_price
                - 1 * zP1HH * nEEHH * IS_price
                + 3 * zP1HH * tEEHH * EE_price
                + 3 * zEEHH * dP1HH
                - 3 * zEEHH * kP1HH * P1_price
                + 1 * zEEHH * mP1HH * H1_price
                + 1 * zEEHH * nP1HH * IS_price
                + 1 * zEEHH * tP1HH * EE_price) / (- 1 + zP1HH + zH1HH + zISHH + zEEHH)

        H1HH_per_cap_demand = - 1 / 3 * (
                - 3 * mH1HH * H1_price
                + 1 * zH1HH * nP1HH * IS_price
                + 1 * zH1HH * tP1HH * EE_price
                - 1 * zISHH * kH1HH * P1_price
                + 3 * zISHH * mH1HH * H1_price
                + 1 * nH1HH * IS_price
                + 1 * tH1HH * EE_price
                + 3 * dH1HH + kH1HH * P1_price
                - 3 * zP1HH * dH1HH
                - 1 * zP1HH * kH1HH * P1_price
                + 3 * zP1HH * mH1HH * H1_price
                - 1 * zP1HH * nH1HH * IS_price
                - 1 * zP1HH * tH1HH * EE_price
                + 3 * zH1HH * dP1HH
                - 3 * zH1HH * kP1HH * P1_price
                + 1 * zH1HH * mP1HH * H1_price
                + 3 * zH1HH * dISHH
                + 1 * zH1HH * kISHH * P1_price
                + 1 * zH1HH * mISHH * H1_price
                - 3 * zH1HH * nISHH * IS_price
                + 1 * zH1HH * tISHH * EE_price
                - 3 * zISHH * dH1HH
                - 1 * zISHH * nH1HH * IS_price
                - 1 * zISHH * tH1HH * EE_price
                + 1 * zH1HH * kEEHH * P1_price
                - 1 * zEEHH * nH1HH * IS_price
                + 3 * zEEHH * mH1HH * H1_price
                + 1 * zH1HH * mEEHH * H1_price
                - 3 * zH1HH * tEEHH * EE_price
                - 1 * zEEHH * kH1HH * P1_price
                - 1 * zEEHH * tH1HH * EE_price
                + 1 * zH1HH * nEEHH * IS_price
                - 3 * zEEHH * dH1HH
                + 3 * zH1HH * dEEHH) / (-1 + zP1HH + zH1HH + zISHH + zEEHH)

        ISHH_per_cap_demand = - 1 / 3 * (
                + 1 * zISHH * nEEHH * IS_price
                + 1 * zISHH * kEEHH * P1_price
                - 1 * tISHH * EE_price * zEEHH
                + 3 * nISHH * IS_price * zEEHH
                - 1 * kISHH * P1_price * zEEHH
                - 3 * dISHH * zEEHH
                + 1 * zISHH * mEEHH * H1_price
                - 3 * zISHH * tEEHH * EE_price
                + 3 * zISHH * dEEHH
                - 1 * mISHH * H1_price * zEEHH
                + 1 * mISHH * H1_price
                + 1 * zISHH * kH1HH * P1_price
                - 3 * zISHH * mH1HH * H1_price
                - 3 * nISHH * IS_price
                + 3 * dISHH
                + 1 * kISHH * P1_price
                + 1 * tISHH * EE_price
                - 3 * zP1HH * dISHH
                - 1 * zP1HH * kISHH * P1_price
                - 1 * zP1HH * mISHH * H1_price
                + 3 * zP1HH * nISHH * IS_price
                - 1 * zP1HH * tISHH * EE_price
                + 3 * zISHH * dP1HH
                - 3 * zISHH * kP1HH * P1_price
                + 1 * zISHH * mP1HH * H1_price
                + 1 * zISHH * nP1HH * IS_price
                + 1 * zISHH * tP1HH * EE_price
                - 3 * zH1HH * dISHH
                - 1 * zH1HH * kISHH * P1_price
                - 1 * zH1HH * mISHH * H1_price
                + 3 * zH1HH * nISHH * IS_price
                - 1 * zH1HH * tISHH * EE_price
                + 3 * zISHH * dH1HH
                + 1 * zISHH * nH1HH * IS_price
                + 1 * zISHH * tH1HH * EE_price) / (-1 + zP1HH + zH1HH + zISHH + zEEHH)

        EEHH_per_capita_demand = - 1 / 3 * (
                - 1 * zISHH * nEEHH * IS_price
                - 1 * zISHH * kEEHH * P1_price
                + 1 * tISHH * EE_price * zEEHH
                - 3 * nISHH * IS_price * zEEHH
                + 1 * kISHH * P1_price * zEEHH
                + 3 * dISHH * zEEHH
                - 1 * zISHH * mEEHH * H1_price
                + 3 * zISHH * tEEHH * EE_price
                - 3 * zISHH * dEEHH
                + 1 * mISHH * H1_price * zEEHH
                + 3 * dEEHH
                + 1 * kEEHH * P1_price
                + 1 * mEEHH * H1_price
                + 1 * nEEHH * IS_price
                - 3 * tEEHH * EE_price
                - 3 * zP1HH * dEEHH
                - 1 * zP1HH * kEEHH * P1_price
                - 1 * zP1HH * mEEHH * H1_price
                - 1 * zP1HH * nEEHH * IS_price
                + 3 * zP1HH * tEEHH * EE_price
                + 3 * zEEHH * dP1HH
                - 3 * zEEHH * kP1HH * P1_price
                + 1 * zEEHH * mP1HH * H1_price
                + 1 * zEEHH * nP1HH * IS_price
                + 1 * zEEHH * tP1HH * EE_price
                - 1 * zH1HH * kEEHH * P1_price
                + 1 * zEEHH * nH1HH * IS_price
                - 3 * zEEHH * mH1HH * H1_price
                - 1 * zH1HH * mEEHH * H1_price
                + 3 * zH1HH * tEEHH * EE_price
                + 1 * zEEHH * kH1HH * P1_price
                + 1 * zEEHH * tH1HH * EE_price
                - 1 * zH1HH * nEEHH * IS_price
                + 3 * zEEHH * dH1HH
                - 3 * zH1HH * dEEHH) / (-1 + zP1HH + zH1HH + zISHH + zEEHH)

        P1HH_per_cap_demand = max(P1HH_per_cap_demand, 0)
        H1HH_per_cap_demand = max(H1HH_per_cap_demand, 0)
        ISHH_per_cap_demand = max(ISHH_per_cap_demand, 0)
        EEHH_per_capita_demand = max(EEHH_per_capita_demand, 0)

        if HH[i] == 0 or N_HH[i] < 2000:
            P1HH_per_cap_demand = 0
            H1HH_per_cap_demand = 0
            ISHH_per_cap_demand = 0
            if case_power_plant == 1:
                EEHH_per_capita_demand = 0

        ISHH_total_demand = max(parameter['msv'],
                                ISHH_per_cap_demand * N_HH[i] * (parameter['theta_P1'] +
                                                                 parameter['lambda_RP']))

        if case_circulation == 1:
            # # minimum_CLI_prod is necessary to ensure minimum circulation
            # minimum_CLI_prod = size_CLI * ISHH_total_demand * circulation_efficiency / (parameter['lambda_RP'] +
            #                                                                            parameter['theta_P1'])
            min_CLI_finished_goods_demand = 0
            # min_CLI_finished_goods_demand = size_CLI * ISHH_total_demand * circulation_efficiency
            if i - processing_time_CLI >= 0:
                # Processing of mass HHCLI received earlier is complete
                # it is transferred out of processing compartment here
                CLI_processed[i] = max(HHCLI[i - processing_time_CLI] * circulation_efficiency, 0)
                # CLI_processed[i + 1] = max(HHCLI[i - processing_time_CLI] * circulation_efficiency, 0)

                min_CLI_finished_goods_demand = max(min(CLI_processed[i],
                                                        ISHH_total_supply[i - processing_time_CLI] *
                                                        size_CLI *
                                                        circulation_efficiency),
                                                    0)

                CLIIRP[i] = max(HHCLI[i - processing_time_CLI] * (1 - circulation_efficiency), 0)
                # CLIIRP[i + 1] = max(HHCLI[i - processing_time_CLI] * (1 - circulation_efficiency), 0)
            # First Flow into the IRP compartment
            IRP_in[i] = CLIIRP[i]    # in addition to uncirculated goods, it also includes process waste

            finished_goods_demand_CLI = max(ISHH_total_demand * willingness_to_pay_CLI, min_CLI_finished_goods_demand)
            finished_goods_demand_TI = ISHH_total_demand - finished_goods_demand_CLI

            # if case_reproduce_paper_1 == 1:
            #     finished_goods_demand_TI = ISHH_total_demand * (1 - size_CLI * circulation_efficiency)
            #     finished_goods_demand_CLI = ISHH_total_demand - finished_goods_demand_TI

            updated_per_cap_dem_TI = finished_goods_demand_TI/(N_HH[i] * (parameter['lambda_RP']
                                                                          + parameter['theta_P1']))

        finished_goods_demand_TI = ISHH_total_demand

        ISHH_fresh_total_demand = ISHH_total_demand
        ISHH_total_demand_data[i] = ISHH_total_demand
        # This is the quantity demanded

        # consumer goods production
        if case_circulation == 1:
            demand_term = updated_per_cap_dem_TI
            # demand_term = ISHH_per_cap_demand  # changed
        else:
            demand_term = ISHH_per_cap_demand

        TI_prod = max(parameter['a_prod_IS']
                      - parameter['b_prod_IS'] * wages
                      + parameter['c_prod_IS'] * (parameter['TI_bar'] -
                                                  (TI_def[i] + TI_inventory[i])) / (parameter['lambda_RP']
                                                                                    + parameter['theta_P1'])
                      - (parameter['d_prod_IS'] * demand_term),
                      parameter['msv'])

        if case_circulation == 1:
            if HH[i] == 0 or N_HH[i] < 2000:
                TI_prod = 0

        # raw material demand for IS production

        RPIS_dem = TI_prod * parameter['lambda_RP']

        P1IS_dem = TI_prod * parameter['theta_P1']

        # Energy compartment related flow (ESIRP), This is first computation.
        # This is approximation based on P1IS_dem, actual flow P1IS is updated in the mass balance process.
        # ESIRP would be altered based on updated value of P1IS later

        EETI_dem = min(parameter['gamma_EEIS'] * P1IS_dem /
                       parameter['theta_P1'],
                       parameter['gamma_EEIS'] * TI_prod)

        EEHH_dem = EEHH_per_capita_demand * N_HH[i]
        EECLI_dem = 0
        EE_dem_total = EEHH_dem + EETI_dem + EECLI_dem
        conve_energy = EE_dem_total
        ESIRP = conve_energy * Energy_Mass

        # Flows related to human-influenced ecological compartments

        if H1[i] == 0 or HH[i] == 0 or N_HH[i] < 2000:
            P1H1_dem = 0
            P2H1 = 0
        else:
            P1H1_dem = max(parameter['d_P1H1'] -
                           parameter['e_P1H1'] * wages -
                           parameter['g_P1H1'] * ((H1_def[i] + H1[i]) - parameter['H1_bar']) , P1H1_dem_min)
            P2H1 = parameter['k_hat']

        if P1[i] == 0 or H2[i] == 0:
            P1H2 = 0
        else:
            P1H2 = max(gRPP1 * P1[i] * RP[i] - parameter['m_P1'] * P1[i] - P1_prod,
                       parameter['msv'])

        if HH[i] == 0 or N_HH[i] < 2000:
            P1H2 = parameter['g_P1H2'] * P1[i] * H2[i]

        P1H2_dem = P1H2

        if H1[i] == 0 or C1[i] == 0:
            H1C1 = 0
        else:
            H1C1 = max(P1H1_dem + P2H1 - parameter['m_H1'] * H1[i] - H1_prod,
                       parameter['msv'])

        if HH[i] == 0 or N_HH[i] < 2000:
            H1C1 = parameter['g_H1C1'] * H1[i] * C1[i]

        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #

        #   Mass balance first: this is carried out to correct the flows of raw materials to TI
        #   repeat mass balance later

        # P1
        P1RP = max(P1[i] * parameter['m_P1'], parameter['msv'])
        RPP1 = max(RP[i] * P1[i] * gRPP1, parameter['msv'])  # Note that gRPP1 has been updated in line 535
        # based on parameter file value
        P1H1 = P1H1_dem
        P1HH = P1HH_per_cap_demand * N_HH[i]
        P1IS = P1IS_dem

        P1_stock = (P1[i] + RPP1 - P1RP - P1H1 - P1H2 - P1HH - P1IS)
        if P1_stock < 0:
            if (P1[i] + RPP1 - P1RP) < 0:
                P1RP = P1[i] + RPP1
                P1H1 = 0
                P1H2 = 0
                P1HH = 0
                P1IS = 0
                # P1CLI[i] = 0
            else:
                P1_demand_total = P1H1 + P1H2 + P1HH + P1IS
                P1_avail = P1[i] + RPP1 - P1RP
                P1H2 = P1_avail / P1_demand_total * P1H2
                P1H1 = P1_avail / P1_demand_total * P1H1
                P1HH = P1_avail / P1_demand_total * P1HH
                P1IS = P1_avail - (P1H2 + P1HH + P1H1)
        elif P1_def_total[i] < 0:
            P1_surplus = min((P1[i] + RPP1 - P1RP - P1H1 - P1H2 - P1HH - P1IS),
                             - P1_def_total[i])

            P1H1 += P1_surplus / P1_def_total[i] * P1H1_def[i]
            P1HH += P1_surplus / P1_def_total[i] * P1HH_def[i]
            P1IS += P1_surplus / P1_def_total[i] * P1TI_def[i]

        # P2 & P3

        P2H2 = parameter['g_P2H2'] * P2[i] * H2[i]
        P2H3 = parameter['g_P2H3'] * P2[i] * H3[i]
        P2RP = max(parameter['m_P2'] * P2[i], parameter['msv'])
        RPP2 = max(gRPP2 * RP[i] * P2[i],
                   parameter['msv'])  # Note that gRPP has been updated in line 535 based on parameter file value
        IRPP2 = max(rIRPP2 * P2[i] * IRP[i], parameter['msv'])
        P3RP = max(parameter['m_P3'] * P3[i], parameter['msv'])
        P3H3 = parameter['g_P3H3'] * P3[i] * H3[i]
        RPP3 = max(gRPP3 * RP[i] * P3[i],
                   parameter['msv'])  # Note that gRPP1 has been updated in line 535 based on parameter file value
        IRPP3 = max(rIRPP3 * P3[i] * IRP[i], parameter['msv'])

        if IRP[i] <= 0:
            IRPP2 = 0
            IRPP3 = 0
        elif IRP[i] - IRPP2 - IRPP3 - max(IRP[i] * parameter['m_IRPRP'], parameter['msv']) \
                + parameter['RPIRP'] + ESIRP + P1IRP < 0:
            if P2[i] != 0:
                IRPP2 = rIRPP2 * (IRP[i] - max(IRP[i] * parameter['m_IRPRP'], parameter['msv'])
                                  + parameter['RPIRP'] + ESIRP + P1IRP) / (rIRPP2 + rIRPP3)

            if P3[i] != 0:
                IRPP3 = rIRPP3 * (IRP[i] - max(IRP[i] * parameter['m_IRPRP'], parameter['msv'])
                                  + parameter['RPIRP'] + ESIRP + P1IRP) / (rIRPP2 + rIRPP3)

        P2_stock = P2[i] + IRPP2 + RPP2 - P2RP - P2H2 - P2H3 - P2H1
        if P2_stock < parameter['th_re']:
            if P2[i] + IRPP2 + RPP2 - P2RP < parameter['th_re']:
                P2RP = P2[i] + IRPP2 + RPP2
                P2H2 = 0
                P2H3 = 0
                P2H1 = 0
            else:
                totP2demand = P2H2 + P2H3 + P2H1
                P2avail = P2[i] + IRPP2 + RPP2 - P2RP
                P2H2 = P2H2 * P2avail / totP2demand
                P2H3 = P2H3 * P2avail / totP2demand
                P2H1 = P2avail - (P2H2 + P2H3)

        P3_stock = P3[i] + IRPP3 + RPP3 - P3RP - P3H3
        if P3_stock < parameter['th_re']:
            if P3[i] + IRPP3 + RPP3 - P3RP < parameter['th_re']:
                P3RP = P3[i] + IRPP3 + RPP3
                P3H3 = 0
            else:
                totP3demand = P3H3
                P3avail = P3[i] + IRPP3 + RPP3 - P3RP
                P3H3 = P3H3 * P3avail / totP3demand
        #
        # H1
        #
        H1RP = max(parameter['m_H1'] * H1[i], parameter['msv'])
        H1HH = H1HH_per_cap_demand * N_HH[i]

        H1_stock = H1[i] + P1H1 + P2H1 - H1RP - H1C1 - H1HH
        if H1_stock < 0:
            if H1[i] + P1H1 + P2H1 - H1RP < 0:
                H1RP = H1[i] + P1H1 + P2H1
                H1C1 = 0
                H1HH = 0
            else:
                totH1demand = H1C1 + H1HH
                H1avail = H1[i] + P1H1 + P2H1 - H1RP
                H1C1 = H1avail * H1C1 / totH1demand
                H1HH = H1avail - H1C1
        elif H1_def[i] < 0:
            H1HH = H1HH + min(H1[i] + P1H1 + P2H1 - H1RP - H1C1 - H1HH, - H1_def[i])

        #
        # H2
        #
        H2C1 = parameter['g_H2C1'] * C1[i] * H2[i]
        H2C2 = parameter['g_H2C2'] * H2[i] * C2[i]  # this may be shifted elsewhere as required
        H2RP = max(parameter['m_H2'] * H2[i], parameter['msv'])

        H2_stock = H2[i] + P1H2 + P2H2 - H2RP - H2C1 - H2C2
        if H2_stock < parameter['th_re']:
            if H2[i] + P1H2 + P2H2 - H2RP < parameter['th_re']:
                H2RP = H2[i] + P1H2 + P2H2
                H2C1 = 0
                H2C2 = 0
            else:
                totH2demand = H2C1 + H2C2
                H2avail = H2[i] + P1H2 + P2H2 - H2RP
                H2C1 = H2C1 * H2avail / totH2demand
                H2C2 = H2avail - H2C1
        #
        # H3
        #
        H3RP = max(parameter['m_H3'] * H3[i], parameter['msv'])
        H3C2 = parameter['g_H3C2'] * H3[i] * C2[i]

        H3_stock = H3[i] + P2H3 + P3H3 - H3RP - H3C2
        if H3_stock < parameter['th_re']:
            if H3[i] + P2H3 + P3H3 - H3RP < parameter['th_re']:
                H3RP = H3[i] + P2H3 + P3H3
                H3C2 = 0
            else:
                totH3demand = H3C2
                H3avail = H3[i] + P2H3 + P3H3 - H3RP
                H3C2 = H3C2 * H3avail / totH3demand
        #
        # C1
        #
        C1RP = max(parameter['m_C1'] * C1[i], parameter['msv'])
        if C1[i] + H1C1 + H2C1 - C1RP < parameter['th_re']:
            C1RP = C1[i] + H1C1 + H2C1

        #
        # C2
        #
        C2RP = max(parameter['m_C2'] * C2[i], parameter['msv'])
        if C2[i] + H2C2 + H3C2 - C2RP < parameter['th_re']:
            C2RP = C2[i] + H2C2 + H3C2

        #
        # HH
        #
        HHRP = (math.ceil(mHH * N_HH[i])
                + math.ceil(N_HH[i] * parameter['phi'] * (per_capita_mass[i] - pcm_ideal) ** 2)) * per_capita_mass[i]
        #
        # RP
        #
        IRPRP = max(IRP[i] * parameter['m_IRPRP'], parameter['msv'])
        RPIS = min(parameter['lambda_RP'] * P1IS / parameter['theta_P1'], RPIS_dem)

        stockRP = RP[i] + P1RP + P2RP + P3RP + H1RP + H2RP + H3RP + C1RP + C2RP + HHRP + IRPRP

        if stockRP < 0:
            stockRP = 0

        if stockRP - (RPP1 + RPP2 + RPP3) - parameter['RPIRP'] - RPIS <= 0:
            if parameter['RPIRP'] == 0:
                RPdemand = RPP1 + RPP2 + RPP3 + RPIS_dem
                RPP1 = RPP1 * stockRP / RPdemand
                RPP2 = RPP2 * stockRP / RPdemand
                RPP3 = RPP3 * stockRP / RPdemand
                if RPIS != 0:
                    RPIS = stockRP - (RPP1 + RPP2 + RPP3)
                else:
                    RPIS = 0

        P1IS = min(parameter['theta_P1'] * RPIS / parameter['lambda_RP'], P1IS)

        # Energy compartment related flow (ESIRP), This is second computation.
        # This is altered based on updated value of P1IS earlier
        EETI_dem = min(parameter['gamma_EEIS'] * P1IS /
                       parameter['theta_P1'],
                       parameter['gamma_EEIS'] * TI_prod)

        EEHH_dem = EEHH_per_capita_demand * N_HH[i]
        EECLI_dem = 0  # check later
        EE_dem_total = EEHH_dem + EETI_dem + EECLI_dem
        conve_energy = EE_dem_total
        updated_ESIRP = conve_energy * Energy_Mass

        # energy sector computations
        if ES[i] > 0:
            fuelcost = cEE / ES[i]
            wagecost = bEE * wages
            if ES[i] - updated_ESIRP < 0:
                updated_ESIRP = ES[i]
                ES[i + 1] = 0
            else:
                ES[i + 1] = ES[i] - updated_ESIRP
        else:
            ES[i + 1] = 0
            updated_ESIRP = 0
            EEHHmass = 0

        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #

        # Repeat mass balance: this is non-negatiivity check

        # P1
        P1_stock = (P1[i] + RPP1 - P1RP - P1H1 - P1H2 - P1HH - P1IS)
        if P1_stock < 0:
            if (P1[i] + RPP1 - P1RP) < 0:
                P1RP = P1[i] + RPP1
                P1H1 = 0
                P1H2 = 0
                P1HH = 0
                P1IS = 0
            else:
                P1_demand_total = P1H1 + P1H2 + P1HH + P1IS
                P1_avail = P1[i] + RPP1 - P1RP
                P1H2 = P1_avail / P1_demand_total * P1H2
                P1H1 = P1_avail / P1_demand_total * P1H1
                P1HH = P1_avail / P1_demand_total * P1HH
                P1IS = P1_avail - (P1H2 + P1HH + P1H1)
        elif P1_def_total[i] < 0:
            P1_surplus = min((P1[i] + RPP1 - P1RP - P1H1 - P1H2 - P1HH - P1IS),
                             - P1_def_total[i])
            P1H1 += P1_surplus / P1_def_total[i] * P1H1_def[i]
            P1HH += P1_surplus / P1_def_total[i] * P1HH_def[i]
            P1IS += P1_surplus / P1_def_total[i] * P1TI_def[i]

        # P2 & P3

        if IRP[i] <= 0:
            IRPP2 = 0
            IRPP3 = 0
        elif (IRP[i] - IRPP2 - IRPP3 - max(IRP[i] * parameter['m_IRPRP'], parameter['msv'])
              + parameter['RPIRP'] + updated_ESIRP + P1IRP) < 0:

            if P2[i] != 0:
                IRPP2 = (rIRPP2 * (IRP[i] - max(IRP[i] * parameter['m_IRPRP'], parameter['msv'])
                                   + parameter['RPIRP'] + updated_ESIRP + P1IRP) / (rIRPP2 + rIRPP3))

            if P3[i] != 0:
                IRPP3 = (rIRPP3 * (IRP[i] - max(IRP[i] * parameter['m_IRPRP'], parameter['msv'])
                                   + parameter['RPIRP'] + updated_ESIRP + P1IRP) / (rIRPP2 + rIRPP3))

        P2_stock = P2[i] + IRPP2 + RPP2 - P2RP - P2H2 - P2H3 - P2H1
        if P2_stock < parameter['th_re']:
            if P2[i] + IRPP2 + RPP2 - P2RP < parameter['th_re']:
                P2RP = P2[i] + IRPP2 + RPP2
                P2H2 = 0
                P2H3 = 0
                P2H1 = 0
            else:
                totP2demand = P2H2 + P2H3 + P2H1
                P2avail = P2[i] + IRPP2 + RPP2 - P2RP
                P2H2 = P2H2 * P2avail / totP2demand
                P2H3 = P2H3 * P2avail / totP2demand
                P2H1 = P2avail - (P2H2 + P2H3)

        # P3
        P3_stock = P3[i] + IRPP3 + RPP3 - P3RP - P3H3
        if P3_stock < parameter['th_re']:
            if P3[i] + IRPP3 + RPP3 - P3RP < parameter['th_re']:
                P3RP = P3[i] + IRPP3 + RPP3
                P3H3 = 0
            else:
                totP3demand = P3H3
                P3avail = P3[i] + IRPP3 + RPP3 - P3RP
                P3H3 = P3H3 * P3avail / totP3demand
        # H1
        H1_stock = H1[i] + P1H1 + P2H1 - H1RP - H1C1 - H1HH
        if H1_stock < 0:
            if H1[i] + P1H1 + P2H1 - H1RP < 0:
                H1RP = H1[i] + P1H1 + P2H1
                H1C1 = 0
                H1HH = 0
            else:
                totH1demand = H1C1 + H1HH
                H1avail = H1[i] + P1H1 + P2H1 - H1RP
                H1C1 = H1avail * H1C1 / totH1demand
                H1HH = H1avail - H1C1
        elif H1_def[i] < 0:
            H1HH = H1HH + min(H1[i] + P1H1 + P2H1 - H1RP - H1C1 - H1HH, - H1_def[i])

        # H2
        H2_stock = H2[i] + P1H2 + P2H2 - H2RP - H2C1 - H2C2
        if H2_stock < parameter['th_re']:
            if H2[i] + P1H2 + P2H2 - H2RP < parameter['th_re']:
                H2RP = H2[i] + P1H2 + P2H2
                H2C1 = 0
                H2C2 = 0
            else:
                totH2demand = H2C1 + H2C2
                H2avail = H2[i] + P1H2 + P2H2 - H2RP
                H2C1 = H2C1 * H2avail / totH2demand
                H2C2 = H2avail - H2C1

        # H3
        H3_stock = H3[i] + P2H3 + P3H3 - H3RP - H3C2
        if H3_stock < parameter['th_re']:
            if H3[i] + P2H3 + P3H3 - H3RP < parameter['th_re']:
                H3RP = H3[i] + P2H3 + P3H3
                H3C2 = 0
            else:
                totH3demand = H3C2
                H3avail = H3[i] + P2H3 + P3H3 - H3RP
                H3C2 = H3C2 * H3avail / totH3demand
        # C1

        if C1[i] + H1C1 + H2C1 - C1RP < parameter['th_re']:
            C1RP = C1[i] + H1C1 + H2C1

        # C2

        if C2[i] + H2C2 + H3C2 - C2RP < parameter['th_re']:
            C2RP = C2[i] + H2C2 + H3C2

        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #

        IS_in[i] = P1IS + RPIS
        finished_goods_available_for_use_TI = P1IS + RPIS + TI_inventory[i]  # this is used in updating the flows
        finished_goods_supply_TI_no_circulation, residual_stock_TI, \
        TI_def_local = flow_computation(demand=finished_goods_demand_TI - TI_def[i],
                                        available=finished_goods_available_for_use_TI)

        if case_circulation == 1:
            finished_goods_available_for_use_CLI = max(0, CLI_inventory[i] + CLI_processed[i] - CLI_bar)
            # finished_goods_available_for_use_CLI = max(0, CLI_inventory[i] + CLI_processed[i - 1] - CLI_bar)

            finished_goods_demand_TI = ISHH_total_demand * willingness_to_pay_TI
            finished_goods_demand_CLI = ISHH_total_demand - finished_goods_demand_TI

            # if case_reproduce_paper_1 == 1:
            #     finished_goods_demand_TI = ISHH_total_demand * (1 - size_CLI * circulation_efficiency)
            #     finished_goods_demand_CLI = ISHH_total_demand - finished_goods_demand_TI

            finished_goods_demand_CLI_data[i] = finished_goods_demand_CLI

            finished_goods_supply_TI = min(finished_goods_available_for_use_TI, finished_goods_demand_TI)
            finished_goods_supply_CLI = min(finished_goods_available_for_use_CLI, finished_goods_demand_CLI)

            residual_stock_TI = max(0, finished_goods_available_for_use_TI - finished_goods_supply_TI)
            residual_stock_CLI = max(0, finished_goods_available_for_use_CLI - finished_goods_supply_CLI) + CLI_bar

            finished_goods_def_TI = min(0, finished_goods_supply_TI - finished_goods_demand_TI)
            finished_goods_def_CLI = min(0, finished_goods_supply_CLI - finished_goods_demand_CLI)

            if finished_goods_def_TI < 0:
                additional_supply_CLI = min(1, TI_price / CLI_price) * min(-finished_goods_def_TI, max(residual_stock_CLI -
                                                                                               CLI_bar, 0))
                # assert additional_supply_CLI >= 0
                # assert residual_stock_CLI - additional_supply_CLI >= 0, '{}'.format([residual_stock_CLI,
                #                                                                      additional_supply_CLI,
                #                                                                      -finished_goods_def_TI,
                #                                                                      max(residual_stock_CLI -
                #                                                                          CLI_bar, 0), TI_price,
                #                                                                      CLI_price
                #                                                                     ])
                finished_goods_supply_CLI += additional_supply_CLI
                residual_stock_CLI -= additional_supply_CLI
                # finished_goods_def_TI = min(0,
                #                             finished_goods_supply_TI -
                #                             finished_goods_demand_TI +
                #                             additional_supply_CLI)

            if finished_goods_def_CLI < 0:
                additional_supply_TI = min(1, CLI_price/TI_price) * min(-finished_goods_def_CLI, residual_stock_TI)
                if case_additional_supply_price_independent == 1:  # delete this, only for testing
                    additional_supply_TI = min(-finished_goods_def_CLI, residual_stock_TI)  # delete this, only for testing
                assert additional_supply_TI >= 0
                finished_goods_supply_TI += additional_supply_TI
                residual_stock_TI -= additional_supply_TI
                # finished_goods_def_CLI = min(0,
                #                              finished_goods_supply_CLI -
                #                              finished_goods_demand_CLI +
                #                              additional_supply_TI)
            assert residual_stock_CLI >= 0, '{}'.format(i)
            assert residual_stock_TI >= 0, '{}'.format(i)

            ISHH_total_supply[i] = finished_goods_supply_TI + finished_goods_supply_CLI
            ISHH_total_supply_l1 = finished_goods_supply_TI + finished_goods_supply_CLI
            finished_goods_supply_TI_l1 = finished_goods_supply_TI
            finished_goods_supply_CLI_l1 = finished_goods_supply_CLI

            minimum_CLI_prod = size_CLI * ISHH_total_supply[i] * circulation_efficiency / (parameter['lambda_RP'] +
                                                                                           parameter['theta_P1'])

            # Constant prod cap
            # important urgent SOS please note
            # DO not comment the following line
            a_term_CLI_prod[i] = parameter['a_prod_IS'] * a_prod_CLI_scaling
            # increase in the production capacity is set wrt initial production capacity

            if case_6th_wave == 1:

                # 6th wave: increasing the production capacity

                upper_limit_production_growth_6th_wave = CLI_multiplier_prod_list[i]
                # calculating actual growth based on deficit
                if i > prod_cap_increase_period:
                    if CLI_def[i - prod_cap_increase_period] < 0:
                        base_prod_cap_to_be_increased_by = CLI_def[i -
                                                                   prod_cap_increase_period] / (parameter['lambda_RP'] +
                                                                                                parameter['theta_P1'])
                        absolute_CLI_prod_cap_multiplier = - base_prod_cap_to_be_increased_by / a_term_CLI_prod[
                            i - prod_cap_increase_period] + 1

                        actual_prod_cap_old = a_term_CLI_prod[i - 1]
                        actual_prod_cap_desired = parameter[
                                                      'a_prod_IS'] * a_prod_CLI_scaling * \
                                                  absolute_CLI_prod_cap_multiplier
                        upper_limit_industrial_growth_CLI = (kappa/100) * (1/52)
                        relative_increase_CLI_prod = 1 + actual_prod_cap_desired/actual_prod_cap_old * upper_limit_industrial_growth_CLI

                        assert absolute_CLI_prod_cap_multiplier >= 0
                        actual_growth_CLI_prod = min(1 + relative_increase_CLI_prod/100, upper_limit_production_growth_6th_wave)
                        actual_growth_CLI_prod = min(relative_increase_CLI_prod, upper_limit_production_growth_6th_wave)

                        percent_production_growth_CLI_data[i] = relative_increase_CLI_prod
                else:
                    actual_growth_CLI_prod = 1

            elif case_overall_limit_CWGR == 1:
                if i > prod_cap_increase_period:
                    if CLI_def[i - prod_cap_increase_period] < 0:
                        base_prod_cap_to_be_increased_by = CLI_def[i -
                                                                   prod_cap_increase_period] / (
                                                                       parameter['lambda_RP'] +
                                                                       parameter['theta_P1'])
                        absolute_CLI_prod_cap_multiplier = - base_prod_cap_to_be_increased_by / a_term_CLI_prod[
                            i - prod_cap_increase_period] + 1

                        actual_prod_cap_old = a_term_CLI_prod[i - 1]
                        actual_prod_cap_desired = parameter[
                                                      'a_prod_IS'] * a_prod_CLI_scaling * \
                                                  absolute_CLI_prod_cap_multiplier
                        upper_limit_industrial_growth_CLI = (kappa / 100) * (1 / 52)
                        relative_increase_CLI_prod = 1 + actual_prod_cap_desired / actual_prod_cap_old * upper_limit_industrial_growth_CLI

                        assert absolute_CLI_prod_cap_multiplier >= 0

                        actual_growth_CLI_prod = min(relative_increase_CLI_prod,
                                                     upper_limit_CWGR**i)

                        percent_production_growth_CLI_data[i] = relative_increase_CLI_prod
                else:
                    actual_growth_CLI_prod = 1

                # actual_growth_CLI_prod = 1  # use this to simulate the case No growth in production
                # actual_growth_CLI_prod = CLI_multiplier_prod_list[i]  # use this to simulate the case max growth
                assert actual_growth_CLI_prod >= 1

            # Included multiplier for c coefficient here
            # local_CLI_prod = parameter['a_prod_IS'] * a_prod_CLI_scaling * actual_growth_CLI_prod - \
            local_CLI_prod = a_term_CLI_prod[i - 1] * actual_growth_CLI_prod - \
                             parameter['b_prod_IS'] * b_prod_CLI_scaling * wages + \
                             parameter['c_prod_IS'] * c_prod_CLI_scaling * \
                             (CLI_bar - (CLI_def[i] + CLI_inventory[i])) / (parameter['lambda_RP'] +
                                                                            parameter['theta_P1'])
            # Following changes are incorporated to scale up other coefficients along with the production capacity i.e.
            # increase b, and c along with a ==> this leads to increased oscillations
            # local_CLI_prod = actual_growth_CLI_prod / a_prod_CLI_scaling * (
            #             a_prod_CLI_scaling * a_term_CLI_prod[i - 1] - parameter['b_prod_IS'] * b_prod_CLI_scaling * wages + parameter[
            #         'c_prod_IS'] * c_prod_CLI_scaling * (CLI_bar - (CLI_def[i] + CLI_inventory[i])) / (
            #                         parameter['lambda_RP'] + parameter['theta_P1']))
            #
            production_growth_CLI_data[i] = actual_growth_CLI_prod
            production_cap_CLI_data[i] = a_term_CLI_prod[i - 1] * actual_growth_CLI_prod

            CLI_prod = max(local_CLI_prod,
                           parameter['msv'],
                           minimum_CLI_prod
                           )
            # rationale: when flows reduce drastically it is wiser to compute production term based on minimum value
            # conditions to be satisfied
            # non-negative, good flows use local computation, low flow use minimum computation
            # good flow or low flow identified based on the ratio of terms computed
            # if minimum_CLI_prod > 0:
            #     if local_CLI_prod / minimum_CLI_prod > 2:  # sharp rise in production is avoided using this condition
            #         CLI_prod = min(local_CLI_prod, minimum_CLI_prod)
            #
            # CLI_prod = max(local_CLI_prod,
            #                parameter['msv'],
            #                minimum_CLI_prod
            #                )

            CLI_prod_min_data[i] = minimum_CLI_prod
            CLI_prod_local_data[i] = local_CLI_prod

            if HH[i] == 0 or N_HH[i] < 2000:
                CLI_prod = 0

            if CLI_prod > 0:
                if i > processing_time_CLI + years_CLI_development * 52:  # years_CLI_development * 52 to ensure CLI
                    # functions during the technological development phase of 30 years
                # if i > processing_time_CLI + 2:
                    # # if virgin_material_use_rate[i - 1] <= 0.5:
                    # if HHCLI[i - 1] <= HHCLI_threshold_flow:  # 2000  minimum number of humans,
                    #     # 10**-8 order of magnitude of per cap demand
                    # condition: if per head used goods transferred to CLI reduce, then stop producing
                    # Stop functioning when per capita used goods transferred to CLI fall below initial value.
                    if HHCLI[i - 1]/N_HH[i - 1] < HHCLI[1]/N_HH[1]:
                    # if (HHCLI[1]/N_HH[1])/(HHCLI[i - 1]/N_HH[i - 1]) > 0.05:
                        CLI_prod = 0


            CLI_price_data[i] = CLI_price
            CLI_prod_data[i] = CLI_prod

            # CLI_prod is in terms of number of units of consumer goods hence it is converted into mass
            raw_material_demand_CLI = (CLI_prod / circulation_efficiency) * (parameter['lambda_RP'] +
                                                                             parameter['theta_P1'])

            CLIHH[i] = finished_goods_supply_CLI
            TIHH[i] = finished_goods_supply_TI

            # First Flow into the IRP compartment
            IRP_in[i] = CLIIRP[i]    # in addition to uncirculated goods, it also includes process waste

            CLI_def[i + 1] = CLI_def[i] + finished_goods_def_CLI

            # added 20191112T1100 the minimum condition
            # a_prod_CLI_scaling_local[i + 1] = a_prod_CLI_scaling_local[i] * (1 + min(local_CLI_prod, 0))
            # if minimum_CLI_prod > 0:
            #     a_prod_CLI_scaling_local[i + 1] = a_prod_CLI_scaling_local[i] * (1 + max((minimum_CLI_prod -
            #                                                                               local_CLI_prod)/
            #                                                                              minimum_CLI_prod,
            #                                                                              0))
            # else:
            #     a_prod_CLI_scaling_local[i + 1] = a_prod_CLI_scaling_local[i]

            ISHH_total_supply[i] = finished_goods_supply_TI + finished_goods_supply_CLI
            ISHH_total_supply_l2 = finished_goods_supply_TI + finished_goods_supply_CLI
            finished_goods_supply_TI_l2 = finished_goods_supply_TI
            finished_goods_supply_CLI_l2 = finished_goods_supply_CLI
            total_def_IS = min(0, ISHH_total_supply[i] - ISHH_total_demand)

            finished_goods_supply_CLI_data[i] = finished_goods_supply_CLI_l2
            finished_goods_supply_TI_data[i] = finished_goods_supply_TI_l2

            IS_def[i + 1] = IS_def[i] + total_def_IS

            # Computing inflows to CLI_processing
            # Used goods available for circulation are limited by the maximum level of circulation
            raw_material_available_CLI = max(ISHH_total_supply[i] * max_circulation, 0)
            HHCLI_processing_available_data[i] = raw_material_available_CLI
            HHCLI_processing_demand_data[i] = raw_material_demand_CLI

            # used goods which cannot be sent to CLI
            uncirculated_used_goods = (ISHH_total_supply[i] * (1 - max_circulation))
            uncirculated_used_goods_data[i] = uncirculated_used_goods
            # transferring residual mass to IRP
            ISIRP[i] += uncirculated_used_goods  # this is the portion of the used goods not circulated,
            # also check raw material surplus
            IRP_in[i] += uncirculated_used_goods  # in addition to uncirculated goods, it also includes process waste

            # Only the quantity demanded by the CLI is supplied
            raw_material_supply_CLI = min(raw_material_available_CLI, raw_material_demand_CLI)
            # raw materials for CLI: demand supply gap, compute surplus
            raw_material_CLI_surplus = max(raw_material_available_CLI - raw_material_demand_CLI, 0)
            raw_material_CLI_def = raw_material_demand_CLI - raw_material_supply_CLI
            raw_material_CLI_def_data[i] = raw_material_CLI_def
            # print(*[raw_material_demand_CLI, raw_material_supply_CLI, raw_material_CLI_def], sep='\t')
            uncirculated_used_goods_data[i] = uncirculated_used_goods + raw_material_CLI_surplus

            if raw_material_CLI_surplus >= 0:
                # used consumer goods not sent for circulation, raw material surplus, are discarded
                ISIRP[i] += raw_material_CLI_surplus  # this is the portion of the used goods not circulated,
                # also check raw material surplus
                IRP_in[i] += raw_material_CLI_surplus

            HHCLI[i] = raw_material_supply_CLI
            # Next time step computation
            CLI_processing[i + 1] = CLI_processing[i] + HHCLI[i] - (CLI_processed[i] + CLIIRP[i])
            CLI_inventory[i + 1] = residual_stock_CLI
            # CLI_inventory[i + 1] = residual_stock_CLI + max(CLI_processed[i] - CLI_processed[i - 1], 0)

            if ISHH_total_supply[i] > 0:
                circulation_fraction[i] = raw_material_supply_CLI/ISHH_total_supply[i]
                circular_material_use_rate[i] = CLIHH[i]/ISHH_total_supply[i]
                virgin_material_use_rate[i] = IS_in[i]/ISHH_total_supply[i]
                raw_material_use_ratio[i] = IS_in[i]/HHCLI[i]
                if IS_in[i] > 0:
                    waste_per_unit_virgin_raw_material[i] = ISIRP[i]/IS_in[i]
                else:
                    waste_per_unit_virgin_raw_material[i] = 0
            else:
                circulation_fraction[i] = 0
        else:
            if TI_inventory[i] + P1IS + RPIS - ISHH_total_supply[i] <= 0:
                ISHH_total_supply[i] = TI_inventory[i] + P1IS + RPIS  # this is the actual flow
            elif TI_def[i] < 0 and N_HH[i] >= 2000:
                ISHH_total_supply[i] += min(TI_inventory[i] + P1IS + RPIS - ISHH_total_supply[i], -TI_def[i])

        # CLI pricing is incorporated in the IS price itself - weighted price. Hence, separate summation not required
        sum_of_flows = P1HH + H1HH + ISHH_total_supply[i] + EEHHmass
        flow_price_multiplication = P1_price * P1HH + H1_price * H1HH \
                                    + IS_price * ISHH_total_supply[i] + EE_price * EEHHmass

        if sum_of_flows <= 0:
            weightedprice = 0
            percapbirths = 0
        elif flow_price_multiplication <= 0:
            weightedprice = 0
            percapbirths = 0
        else:
            weightedprice = flow_price_multiplication / sum_of_flows
            percapbirths = max(eta_a - eta_b * math.sqrt(wages / weightedprice), parameter['msv'])

        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #

        # Storing data
        # Auxillary data
        stockRP_data[i] = stockRP
        mass_differential_data[i] = P1HH + H1HH
        EETI_dem_data[i] = EETI_dem
        EEHH_dem_data[i] = EEHH_dem
        ESIRP_data[i] = updated_ESIRP
        inflows_P2[i] = IRPP2 + RPP2
        outflows_P2[i] = -(- P2RP - P2H2 - P2H3 - P2H1)
        inflows_P3[i] = IRPP3 + RPP3
        outflows_P3[i] = -(- P3RP - P3H3)
        mass_ecological_RP[i] = P1[i] + P2[i] + P3[i] + H1[i] + H2[i] + H3[i] + C1[i] + C2[i] + RP[i] + HH[i]
        EECLI_data[i] = EECLI_dem
        EEHH_data[i] = EEHHmass
        EE_price_data[i] = EE_price
        wages_data[i] = wages

        # Flows: economic dimension
        P1HH_per_cap_demand_data[i] = P1HH_per_cap_demand
        H1HH_per_cap_demand_data[i] = H1HH_per_cap_demand
        ISHH_per_cap_demand_data[i] = ISHH_per_cap_demand
        EEHH_per_capita_demand_data[i] = EEHH_per_capita_demand
        P1HH_data[i] = P1HH
        H1HH_data[i] = H1HH
        EEHH_data[i] = EEHHmass
        P1H1_data[i] = P1H1
        P1IS_data[i] = P1IS
        P1IS_dem_data[i] = P1IS_dem
        RPIS_dem_data[i] = RPIS_dem
        IS_price_data[i] = IS_price

        # Flows: Intercompartmental
        P1H2_data[i] = P1H2
        P2H1_data[i] = P2H1
        P2H2_data[i] = P2H2
        P2H3_data[i] = P2H3
        P3H3_data[i] = P3H3
        H1C1_data[i] = H1C1
        H2C1_data[i] = H2C1
        H2C2_data[i] = H2C2
        H3C2_data[i] = H3C2

        # Flows: mortality
        P1RP_data[i] = P1RP
        P2RP_data[i] = P2RP
        P3RP_data[i] = P3RP
        H1RP_data[i] = H1RP
        H2RP_data[i] = H2RP
        H3RP_data[i] = H3RP
        C1RP_data[i] = C1RP
        C2RP_data[i] = C2RP
        HHRP_data[i] = HHRP

        # Flows: Recycling and RP flows
        IRPP2_data[i] = IRPP2
        IRPP3_data[i] = IRPP3
        IRPRP_data[i] = IRPRP
        RPP1_data[i] = RPP1
        RPP2_data[i] = RPP2
        RPP3_data[i] = RPP3
        RPIS_data[i] = RPIS

        # Economic variables: wages, and weighted price
        wages_data[i] = wages
        weightedprice_data[i] = weightedprice

        # Economic variables: prices
        P1_price_data[i] = P1_price
        H1_price_data[i] = H1_price
        TI_price_data[i] = TI_price
        EE_price_data[i] = EE_price

        # Economic variables: production
        P1_prod_data[i] = P1_prod
        H1_prod_data[i] = H1_prod
        TI_prod_data[i] = TI_prod

        # Human population related variables
        percapbirths_data[i] = percapbirths
        IS_inventory[i] = TI_inventory[i]
        if case_circulation == 1:
            IS_inventory[i] = TI_inventory[i] + CLI_inventory[i]
            if finished_goods_supply_CLI > 0:
                relative_circulation[i] = finished_goods_supply_TI/finished_goods_supply_CLI

        P1_demand_total_data[i] = P1IS_dem + P1HH_per_cap_demand * N_HH[i] + P1H1_dem + P1H2_dem
        P1_supply_total_data[i] = P1IS + P1HH + P1H1 + P1H2
        P1_demand_humans_data[i] = P1IS_dem + P1HH_per_cap_demand * N_HH[i] + P1H1_dem
        P1_supply_humans_data[i] = P1IS + P1HH + P1H1
        P1_supply_stocks[i] = max(P1_supply_total_data[i] - RPP1, 0)
        P1_created[i] = RPP1
        total_income[i] = wages
        total_expenditure[i] = flow_price_multiplication

            # if working_CLI == 0:
            #     circulation_fraction[i] = 0

        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #
        #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #    #

        # Computation of the comapartment masses and other variables for the next timestep
        if i < simulation_time:
            P1[i + 1] = P1[i] + RPP1 - P1RP - P1H2 - P1H1 - P1HH - P1IS

            if P1[i + 1] < P1[i] and i > 10000:
                RPP1 = RPP1

            if P1[i] == 0:
                P1H1_dem = 0
                P1IS_dem = 0
                P1HH_per_cap_demand = 0

            P1H1_def[i + 1] = P1H1_def[i] + P1H1 - P1H1_dem

            P1TI_def[i + 1] = P1TI_def[i] + P1IS - P1IS_dem

            P1HH_def[i + 1] = P1HH_def[i] + P1HH - P1HH_per_cap_demand * N_HH[i]

            P1_def_total[i + 1] = P1H1_def[i + 1] + P1TI_def[i + 1] + P1HH_def[i + 1]

            P2[i + 1] = P2[i] + IRPP2 + RPP2 - P2RP - P2H2 - P2H3 - P2H1

            P3[i + 1] = P3[i] + IRPP3 + RPP3 - P3RP - P3H3

            H1[i + 1] = H1[i] + P1H1 + P2H1 - H1RP - H1C1 - H1HH

            if H1[i] == 0:
                H1HH_per_cap_demand = 0

            H1_def[i + 1] = H1_def[i] + H1HH - H1HH_per_cap_demand * N_HH[i]

            H2[i + 1] = H2[i] + P1H2 + P2H2 - H2RP - H2C1 - H2C2

            H3[i + 1] = H3[i] + P2H3 + P3H3 - H3RP - H3C2

            C1[i + 1] = C1[i] + H1C1 + H2C1 - C1RP

            C2[i + 1] = C2[i] + H2C2 + H3C2 - C2RP

            HH[i + 1] = HH[i] + P1HH + H1HH - HHRP

            RP[i + 1] = stockRP - (RPP1 + RPP2 + RPP3) - parameter['RPIRP'] - RPIS

            TI_inventory[i + 1] = TI_inventory[i] + IS_in[i] - ISHH_total_supply[i]

            if TI_inventory[i + 1] < 0:
                TI_inventory[i + 1] = 0

            demand_term = ISHH_fresh_total_demand
            TI_def[i + 1] = TI_def[i] + ISHH_total_supply[i] - demand_term
            if case_circulation == 1:
                TI_inventory[i + 1] = residual_stock_TI
                IRP_in[i] += parameter['RPIRP'] + updated_ESIRP
                TI_def[i + 1] = TI_def[i] + finished_goods_def_TI
            else:
                IRP_in[i] = ISHH_total_supply[i] + parameter['RPIRP'] + updated_ESIRP

            IRP_out[i] = IRPP2 + IRPP3 + IRPRP
            IRP[i + 1] = IRP[i] - IRP_out[i] + IRP_in[i]

            N_HH[i + 1] = max(N_HH[i]
                              + math.ceil(percapbirths * N_HH[i])
                              - math.ceil(mHH * N_HH[i])
                              - math.ceil(N_HH[i] * parameter['phi'] *
                                          (per_capita_mass[i] - pcm_ideal) ** 2),
                              1.0)

            per_capita_mass[i + 1] = HH[i + 1] / N_HH[i + 1]

            if case_limit_pcm == 1:
                if per_capita_mass[i + 1] > per_capita_mass[0] * limit_pcm_multiplier_upper:
                    per_capita_mass[i + 1] = per_capita_mass[0] * limit_pcm_multiplier_upper
                    HH[i + 1] = min(per_capita_mass[i + 1] * N_HH[i + 1],
                                    HH[i] + (P1HH + H1HH - HHRP))
                    food_waste = max(0, HH[i] + (P1HH + H1HH - HHRP) - HH[i + 1])
                    RP[i + 1] += food_waste
                    food_waste_data[i + 1] = food_waste

                if per_capita_mass[i + 1] < per_capita_mass[0] * limit_pcm_multiplier_lower:
                    per_capita_mass[i + 1] = per_capita_mass[0] * limit_pcm_multiplier_lower
                    if N_HH[i + 1] > 0:
                        if per_capita_mass[i + 1] > 0:
                            N_HH[i + 1] = HH[i + 1] / per_capita_mass[i + 1]

            food_consumed = HH[i + 1] - HH[i]
            food_consumed_data[i] = food_consumed

            sys_mass[i] = P1[i] + P2[i] + P3[i] + H1[i] + H2[i] + H3[i] + C1[i] + C2[i] + RP[i] + IRP[i] + HH[i] + \
                          ES[i] + TI_inventory[i]

            if case_circulation == 1:
                sys_mass[i] += CLI_inventory[i] + CLI_processing[i]

            #  Checking extinction of ecological compartments

            if TI_inventory[i] == 0:
                switch_variable_IS += 1
                if switch_variable_IS == 1:
                    extinction_timestep_IS = i
                    print('TI became zero at {} timestep = {} years'.format(i, int(i / 52)))
            if P1[i] == 0:
                switch_variable_P1 += 1
                if switch_variable_P1 == 1:
                    extinction_timestep_P1 = i
                    print('P1 became zero at {} timestep = {} years'.format(i, int(i / 52)))
            if P2[i] == 0:
                switch_variable_P2 += 1
                if switch_variable_P2 == 1:
                    extinction_timestep_P2 = i
                    print('P2 became zero at {} timestep = {} years'.format(i, int(i / 52)))
            if P3[i] == 0:
                switch_variable_P3 += 1
                if switch_variable_P3 == 1:
                    extinction_timestep_P3 = i
                    print('P3 became zero at {} timestep'.format(i))
            if H1[i] == 0:
                switch_variable_H1 += 1
                if switch_variable_H1 == 1:
                    extinction_timestep_H1 = i
                    print('H1 became zero at {} timestep'.format(i))
            if H2[i] == 0:
                switch_variable_H2 += 1
                if switch_variable_H2 == 1:
                    extinction_timestep_H2 = i
                    print('H2 became zero at {} timestep'.format(i))
            if H3[i] == 0:
                switch_variable_H3 += 1
                if switch_variable_H3 == 1:
                    extinction_timestep_H3 = i
                    print('H3 became zero at {} timestep'.format(i))
            if C1[i] == 0:
                switch_variable_C1 += 1
                if switch_variable_C1 == 1:
                    extinction_timestep_C1 = i
                    print('C1 became zero at {} timestep'.format(i))
            if C2[i] == 0:
                switch_variable_C2 += 1
                if switch_variable_C2 == 1:
                    extinction_timestep_C2 = i
                    print('C2 became zero at {} timestep'.format(i))
            if ES[i] == 0:
                switch_variable_ES += 1
                if switch_variable_ES == 1:
                    extinction_timestep_ES = i
                    print('ES became zero at {} timestep'.format(i))

            if i % (simulation_time - 2) == 0 and i > 0:
            # if i % 2000 == 0 and i > 0:

                local_color = 'k'
                local_alpha = 0.4
                local_linestyle = '-'

                # Data processing, willingness to pay in percentage
                willingness_to_pay_CLI_data = [x * 100 for x in willingness_to_pay_CLI_data]

                # paper 2 Figure 3
                plotting_list = [
                    [r'$P1$', P1],
                    [r'$P2$', P2],
                    [r'$wages$', wages_data],
                    [r'$Circulation\ Fraction$', circulation_fraction],
                    [r'$X$', willingness_to_pay_CLI_data],
                    [r'$TI\ inventory$', TI_inventory],
                    [r'$Delta C$', production_growth_CLI_data],
                ]
                # paper 2 Figure 6
                plotting_list = [
                    [r'$P1$', P1],
                    [r'$P2$', P2],
                    [r'$wages$', wages_data],
                ]

    # storing the variable data to be plotted
    if save_data == 1:
        #     Creating necessary directories
        def f_create_necessary_dir(parent_directory, clear_everything=False):
            if clear_everything == True:
                if os.path.exists(parent_directory) and os.path.isdir(parent_directory):
                    shutil.rmtree(parent_directory)
            # this is the path of the parent directory
            if not path.isdir(parent_directory):
                mkdir('{}'.format(parent_directory))
            return

        output_directory = join(expanduser('~'), os.getcwd(), 'output')
        f_create_necessary_dir(output_directory)

        if case_circulation == 0:
            directory_name = 'No circulation'
        directory_name = 'Consumption  = {}  max premium = {} min discount = {}'.format(consumption_level_list[0], A, B)

        parent_directory = join(expanduser('~'), os.getcwd(), 'output', directory_name)
        f_create_necessary_dir(parent_directory)

        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M')
        file_name = join(expanduser('~'), parent_directory, timestamp+'.csv')
        # converting data to pandas dataframe for easy handling
        plotting_list_dict = {variable_data[0]: variable_data[1] for variable_data in plotting_list}
        # df = pd.DataFrame.from_dict(plotting_list_dict)
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in plotting_list_dict.items()]))

        with open(file_name, 'w') as f:
            df.to_csv(f, sep='\t')
            f.close()

        print('Data saved in {}'.format(directory_name))
