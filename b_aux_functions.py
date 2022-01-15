import random, math
import matplotlib.pyplot as plt
import os
import pandas as pd
# from agent import *
# from birth import *
from numpy.random import multinomial
import numpy as np
from os.path import join, expanduser
from os import path, getcwd, listdir, mkdir
import pathlib


def prod_multiplier_calculation_a_CLI(x, width, center, prod_cap_local):
    # sigmoid function
    t = x
    M = center
    B = 1 / width * 0.7

    C = 1
    Q = 1
    denominator_exponent = 1

    A = 0
    K = prod_cap_local
    local_a_CLI_prod_multiplier = np.array(
        A + (K - A) / (C + Q * np.exp(-B * (t - M))) ** (1 / denominator_exponent)).tolist()

    return local_a_CLI_prod_multiplier


def wtp_literature_segments(consumer_perception):
    x = consumer_perception
    x_premium = [x[0] for x in x]
    x_wtp = [x[1] for x in x]
    # print(x_premium)
    # print(x_wtp)
    lp1 = []
    lw1 = []
    for j, x in enumerate(x_premium):
        if j < len(x_premium) - 1:
            lp1.extend(np.linspace(x, x_premium[j + 1], 10000).tolist())
    for j, x in enumerate(x_wtp):
        if j < len(x_wtp) - 1:
            lw1.extend(np.linspace(x/100, x_wtp[j + 1]/100, 10000).tolist())

    wtp_dict = {}
    for k, v in zip(lp1, lw1):
        wtp_dict[k] = v
    return wtp_dict


def wtp_literature_linear(consumer_perception):

    nobody_wants_CLI = consumer_perception[1]
    everybody_wants_CLI = consumer_perception[0]
    premium = np.linspace(everybody_wants_CLI, nobody_wants_CLI, 1000).tolist()
    coefficients = np.polyfit(consumer_perception, [1, 0], 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    wtp = [slope * x + intercept for x in premium]
    # plt.plot(premium, wtp)
    # plt.show()
    # exit()

    wtp_dict = {}
    for k, v in zip(premium, wtp):
        wtp_dict[k] = v

    return wtp_dict


def wtp_double_logistics(size_CLI, circulation_efficiency, vertical_shift_wtp_crossover,
                         horizontal_shift):
    exp_term_A = size_CLI * circulation_efficiency
    exp_term_K = 1
    exp_term_C = 1
    exp_term_Q = 1
    exp_term_M = horizontal_shift
    exp_term_B = 120
    test_price_ratio_points = np.linspace(0.0, 2, 10000)
    plot_wtp_first = []
    for i, x in enumerate(test_price_ratio_points):
        # willingness_to_pay = math.log((1 - x)/x)
        willingness_to_pay = wtp_logistics(x=x,
                                           exp_term_A=exp_term_A,
                                           exp_term_B=exp_term_B,
                                           exp_term_C=exp_term_C,
                                           exp_term_K=exp_term_K,
                                           exp_term_M=exp_term_M,
                                           exp_term_Q=exp_term_Q)
        plot_wtp_first.append(willingness_to_pay)
    plot_wtp_second = []
    exp_term_B = horizontal_shift * 15
    exp_term_M = (2 - horizontal_shift) * 2
    for i, x in enumerate(test_price_ratio_points):
        # willingness_to_pay = math.log((1 - x)/x)
        willingness_to_pay = wtp_logistics(x=x,
                                           exp_term_A=0,
                                           exp_term_B=exp_term_B,
                                           exp_term_C=exp_term_C,
                                           exp_term_K=exp_term_K,
                                           exp_term_M=exp_term_M,
                                           exp_term_Q=exp_term_Q)
        plot_wtp_second.append(willingness_to_pay)
    #     transform y from 1 to 0.5
    plot_wtp_first = [x * vertical_shift_wtp_crossover for x in plot_wtp_first]
    #     transform y from 1 to 2
    plot_wtp_second = [x * (1 - vertical_shift_wtp_crossover) + vertical_shift_wtp_crossover for x in plot_wtp_second]
    plot_wtp_total = plot_wtp_first + plot_wtp_second
    # #     offset y to 0.5
    # offset_y = min(plot_wtp_second) - 0.5
    # plot_wtp_second = [x - offset_y for x in plot_wtp_second]

    #     transform x from 2 to 1
    test_price_ratio_points_1 = [x/2 for x in test_price_ratio_points]
    test_price_ratio_points_2 = [x/2 + 1 for x in test_price_ratio_points]
    test_price_ratio_points = test_price_ratio_points_1 + test_price_ratio_points_2

    wtp_logistics_dict = {}
    for k, v in zip(test_price_ratio_points, plot_wtp_total):
        wtp_logistics_dict[k] = v

    # plt.plot(plot_wtp_total)
    # plt.xlim([0.75, 1.5])
    # plt.ylim([0., 1.1])
    # plt.show()

    return wtp_logistics_dict


def wtp_logit(x, exp_term_A, exp_term_K, exp_term_C, exp_term_Q, exp_term_B, exp_term_M, exp_term_nu):
    # wtp_computation function
    # x = x/12
    # wtp_list = (5 - 1 / 2 * math.log(1/x - 1))

    scale_x = 1
    scale_y = 1
    x = x * scale_x
    term_ratio = ((exp_term_K - exp_term_A)/(x - exp_term_A)) ** exp_term_nu
    wtp_list = (- 1 / exp_term_B * math.log((term_ratio - exp_term_C)/exp_term_Q) + exp_term_M)/scale_y

    return wtp_list


def linear_WTP(low, high, price_ratio, width):
    return low + (high - low)/2 * price_ratio


def wtp_logistics(x, exp_term_A, exp_term_K, exp_term_C, exp_term_Q, exp_term_B, exp_term_M):
    # wtp_computation function
    term_1 = math.log(exp_term_C + exp_term_Q)
    term_2 = math.log((exp_term_K - exp_term_A)/(0.5 - exp_term_A))
    local_exponent = term_1/term_2
    fraction_diff = exp_term_K - exp_term_A
    exponent_part = (exp_term_C + np.exp(-exp_term_B * (x - exp_term_M)))**(1/local_exponent)

    wtp_list = np.array(exp_term_A + fraction_diff / exponent_part).tolist()

    return wtp_list


def flow_computation(available, demand):
    residual = available - demand
    if residual >= 0:
        flow_local = demand
        inventory_bar = residual
        deficit = 0
    else:
        flow_local = available
        inventory_bar = 0
        deficit = residual

    return [flow_local, inventory_bar, deficit]


def transfer_from_CLI(CLI_available_local, TI_available_local, total_demand, i):

    local_demand = total_demand
    local_supply = 0
    dem_CLI = local_demand
    # satisfy the demand using low price product first
    supply_CLI = min(CLI_available_local, local_demand)
    CLI_inventory_local = CLI_available_local - supply_CLI
    local_supply += supply_CLI  # update the supply
    local_demand -= supply_CLI  # update the demand
    dem_TI = local_demand
    supply_TI = 0
    total_deficit = 0

    if local_demand > 0:  # demand not satisfied by the low price product
        # satisfy the demand using high price product
        supply_TI = min(TI_available_local, local_demand)
    TI_inventory_local = TI_available_local - supply_TI
    local_supply += supply_TI  # update the supply
    local_demand -= supply_TI  # update the demand
    # assert supply_TI + supply_CLI <= total_demand
    if (supply_CLI + supply_TI) - total_demand > 10**-12:
        print(*['{}'.format(i), '{:.3E}'.format(supply_CLI), '{:.3E}'.format(supply_TI), '{:.3E}'.format(total_demand)], sep='\t')

    if local_demand > 0:
        total_deficit = (-local_demand)
    # print(CLI_inventory_local - supply_CLI + dem_CLI + TI_inventory_local - supply_TI + dem_TI + total_deficit)

    # assert CLI_inventory_local + supply_CLI + dem_CLI + TI_inventory_local + supply_TI + dem_TI + total_deficit == 0

    return_term = [CLI_inventory_local, supply_CLI, dem_CLI,
                   TI_inventory_local, supply_TI, dem_TI,
                   total_deficit]

    return return_term


def transfer_from_TI(CLI_available_local, TI_available_local, total_demand, i):

    local_demand = total_demand
    local_supply = 0
    dem_TI = local_demand
    # satisfy the demand using low price product first
    supply_TI = min(TI_available_local, local_demand)
    TI_inventory_local = TI_available_local - supply_TI
    local_supply += supply_TI  # update the supply
    local_demand -= supply_TI  # update the demand
    dem_CLI = local_demand
    supply_CLI = 0
    total_deficit = 0

    if local_demand > 0:  # demand not satisfied by the low price product
        # satisfy the demand using high price product
        supply_CLI = min(CLI_available_local, local_demand)
    CLI_inventory_local = CLI_available_local - supply_CLI
    local_supply += supply_CLI  # update the supply
    local_demand -= supply_CLI  # update the demand
    if local_demand > 0:
        total_deficit = (-local_demand)
    # assert CLI_inventory_local + supply_CLI + dem_CLI + TI_inventory_local + supply_TI + dem_TI + total_deficit == 0
    # assert [CLI_inventory_local, supply_CLI, dem_CLI, TI_inventory_local, supply_TI, dem_TI, total_deficit]
    # print(CLI_inventory_local - supply_CLI + dem_CLI + TI_inventory_local - supply_TI + dem_TI + total_deficit)
    if (supply_CLI + supply_TI) - total_demand > 10**-12:
        print(*['{}'.format(i), '{:.3E}'.format(supply_CLI), '{:.3E}'.format(supply_TI), '{:.3E}'.format(total_demand)], sep='\t')

    return_term = [CLI_inventory_local, supply_CLI, dem_CLI,
                   TI_inventory_local, supply_TI, dem_TI,
                   total_deficit]

    return return_term


def circulation(CLI_inventory_local_series, TI_inventory_local_series,
                CLI_processed_local, CLI_processing_local,
                CLI_price_local, TI_price_local,
                CLIIRP_local, HHCLI_local, IRP_in_local, ISHH_total_demand_local,
                RPIS_local, P1IS_local, ISHH_total_supply_local,
                CLIHH_array, TIHH_array, CLI_prod_local,
                lambda_RP_local, theta_P1_local, circulation_efficiency_local, 
                processing_time_CLI_local, price_tolerance_local, i):

    if i - processing_time_CLI_local >= 0:
        CLI_processed_local[i] = HHCLI_local[i - processing_time_CLI_local] * circulation_efficiency_local
        CLIIRP_local[i] = HHCLI_local[i - processing_time_CLI_local] - CLI_processed_local[i]
        CLI_processing_local[i] -= (CLI_processed_local[i] + CLIIRP_local[i])  # updated earlier as well

    IRP_in_local[i] = CLIIRP_local[i]

    CLI_available_for_HH_local = CLI_inventory_local_series[i] + CLI_processed_local[i]
    TI_available_for_HH_local = TI_inventory_local_series[i] + RPIS_local + P1IS_local

    # price_tolerance < 1 less environmental concern: need discount to buy CLI
    # price_tolerance > 1 more environmental concern: need penalty to buy TI

    if CLI_price_local <= price_tolerance_local * TI_price_local:  # CLI price is low, consume CLI first
        # deficit term as supply - demand, same format as used in the next timestep computation
        CLI_inventory_local_single_value, CLIHH_local, def_CLI_local, \
        TI_inventory_local_single_value, TIHH_local, def_TI_local, \
        IS_deficit_negative = demand_satisfaction_consumer_goods(CLI_available_for_HH_local,
                                                                 TI_available_for_HH_local,
                                                                 ISHH_total_demand_local)

    else:
        TI_inventory_local_single_value, TIHH_local, def_TI_local, \
        CLI_inventory_local_single_value, CLIHH_local, def_CLI_local, \
        IS_deficit_negative = demand_satisfaction_consumer_goods(TI_available_for_HH_local,
                                                                 CLI_available_for_HH_local,
                                                                 ISHH_total_demand_local)

    ISHH_total_supply_local[i] = TIHH_local + CLIHH_local
    CLIHH_array[i] = CLIHH_local
    TIHH_array[i] = TIHH_local

    # Computing inflows to CLI_processing from processing block
    HHCLI_processing_available = ISHH_total_supply_local[i]
    HHCLI_processing_demand = CLI_prod_local / circulation_efficiency_local * (lambda_RP_local
                                                                               + theta_P1_local)
    # CLI_prod is in terms of number of units of consumer goods hence it is converted into mass
    HHCLI_processing_supply = min(HHCLI_processing_available, HHCLI_processing_demand)
    if HHCLI_processing_available > HHCLI_processing_demand:  # used consumer goods not sent for circulation
        # are discarded
        IRP_in_local[i] += (HHCLI_processing_available - HHCLI_processing_demand)

    HHCLI_local[i] = HHCLI_processing_supply
    # Next time step computation
    CLI_processing_local[i + 1] = CLI_processing_local[i] + HHCLI_processing_supply
    CLI_inventory_local_series[i + 1] = CLI_inventory_local_single_value
    CLI_inventory_local_series[i] = CLI_inventory_local_single_value

    return_term = []
    return return_term


def demand_satisfaction_consumer_goods(priority_1_inventory, priority_2_inventory, total_demand):

    local_demand = total_demand
    local_supply = 0

    # satisfy the demand using low price product first
    supply_1 = min(priority_1_inventory, total_demand)
    priority_1_inventory = priority_1_inventory - supply_1
    # priority_1_inventory -= supply_1
    local_supply += supply_1
    supply_2 = 0
    total_deficit = 0
    deficit_1 = 0
    deficit_2 = 0

    if local_demand > supply_1:  # demand not satisfied by the low price product
        local_demand -= supply_1  # update the demand
        deficit_1 = (-local_demand)
        # satisfy the demand using high price product
        supply_2 = min(priority_2_inventory, local_demand)
        priority_2_inventory -= supply_2
        local_supply += supply_2  # update the supply
        local_demand -= supply_2  # update the demand
        if local_demand != 0:
            total_deficit = (-local_demand)
            deficit_2 = total_deficit

    return_term = [priority_1_inventory, supply_1, deficit_1,
                   priority_2_inventory, supply_2, deficit_2,
                   total_deficit]

    return return_term


def create_readme_file_for_simulation(case_directory):
    # creating the readme file with case details
    case_information_readme = []
    files_to_be_read = [join(expanduser('~'), os.getcwd(), 'main.py'),
                        join(expanduser('~'), os.getcwd(), 'initialization_and_series.py')]

    for file in files_to_be_read:
        case_information_readme.append('\n\n')
        with open(file, 'r') as f:
            local_data_storage = f.readlines()

            reading_condition_readme = False
            start_reading_flag_readme = "# START copying readme data here"
            stop_reading_flag_readme = "# STOP copying readme data here"

            for line_index, line in enumerate(local_data_storage):
                if stop_reading_flag_readme in line:
                    reading_condition_readme = False
                    continue
                if reading_condition_readme:
                    if "#" not in line:
                        case_information_readme.append(line)
                    # case_information_readme.append('\n'+line)
                if start_reading_flag_readme in line:
                    reading_condition_readme = True
            f.close()

    # extracting variable names
    all_variable_names_local = []
    with open(join(expanduser('~'), os.getcwd(), 'main.py'), 'r') as f:
        local_data_storage = f.readlines()
        reading_condition_variable_names = False
        start_reading_flag_variable_names = "# START copying variable_names data here"
        stop_reading_flag_variable_names = "# STOP copying variable_names data here"

        for line_index, line in enumerate(local_data_storage):
            if stop_reading_flag_variable_names in line:
                reading_condition_variable_names = False
                continue
            if reading_condition_variable_names:
                all_variable_names_local.append(line)
                # case_information_variable_names.append('\n'+line)
            if start_reading_flag_variable_names in line:
                reading_condition_variable_names = True
        f.close()
    # Formatting the list of variable names
    b = [x.replace(' ', '') for x in all_variable_names_local]
    c = [x.replace(',', '') for x in b]
    d = [x.replace('data', '') for x in c]
    e = [x.replace('_', ' ') for x in d]
    f = [x.replace('\n', '') for x in e]
    g = [x.replace('values', '') for x in f]
    a = [x for x in g if x != '']
    all_variable_names_local = a

    # Writing the requisite information into the readme file
    readme_file_path = join(expanduser('~'), case_directory, 'readme.txt')
    if not path.exists(readme_file_path):
        with open(readme_file_path, 'a') as f:
            f.write('\n'.join(all_variable_names_local))
            f.writelines(case_information_readme)
            f.close()
    return all_variable_names_local


def population_pyramid(population_groups):
    for i in range(len(population_groups)):
        if i == 0:
            lst = [random.randrange(0, 208, 1)
                   for x in range(int(population_groups[0] * 10000 / 100))]
        else:
            lst += [random.randrange(208 + 260 * (i - 1), 208 + i * 260, 1)
                    for x in range(int(population_groups[i] * 10000 / 100))]
    return lst


def integer_list(list_size, list_sum):
    """
    Inputs:
    list_size = the size of the list to return
    list_sum = The sum of list values
    Uniform distribution
    Output:
    A list of random integers of length 'list_size' whose sum is 'list_sum'.
    """
    prob_distribution = [1/list_size for i in range(list_size)]
    generated_numbers = multinomial(list_sum, prob_distribution, size=1)
    output = generated_numbers[0]
    return output


def plot_mass_main(humans, i, plot_directory_mass):
    human_mass_list_under_10k = []
    human_mass_list_beyond_10k = []
    j, k, l = 0, 0, 0
    
    # Please note: since all previously existing humans expire well before 6k timesteps, only trends of agents with uid
    # greater than 10k are plotted
    
    if i < 6000:
        existing_humans_under_10k, existing_humans_over_10k, all_existing_humans = 0, 0, 0
        
        for agent in humans:
            if agent.unique_id <= 10000:
                if agent.mass != 0:
                    human_mass_list_under_10k.append(agent.mass)
                    existing_humans_under_10k += 1
            elif agent.mass != 0:
                human_mass_list_beyond_10k.append(agent.mass)
                existing_humans_over_10k += 1
        
        all_existing_humans = existing_humans_under_10k + existing_humans_over_10k
        human_mass_list_all = human_mass_list_under_10k + human_mass_list_beyond_10k
        fig_num = str(i).zfill(5)
        plot_title = r'Mass distribution - upto ' \
                     r'uid 10k' + '\n' + 'at {}th time step'.format(i)
        # + '\n' + 'Number of agents = {}'.format(existing_humans_under_10k)
        fig_name = '{}/less_than_10k__{}'.format(plot_directory_mass, fig_num) + '.png'
        plot_mass_called(human_mass_list_under_10k, plot_title, fig_name)
        print("mass distribution plotted for {} upto uid 10k".format(i))

        plot_title = r'Mass distribution - ' \
                     r'uid more than 10k' + '\n' + 'at {}th time step'.format(i)
        # + '\n' + 'Number of agents = {}'.format(existing_humans_over_10k)
        fig_name = '{}/more_than_10k__{}'.format(plot_directory_mass, fig_num) + '.png'
        plot_mass_called(human_mass_list_beyond_10k, plot_title, fig_name)
        print("mass distribution plotted for {} uid more than 10k".format(i))

        plot_title = r'Mass distribution - ' \
                     r'all existing agents' + '\n' + 'at {}th time step'.format(i)
        # + '\n' + 'Number of agents = {}'.format(all_existing_humans)
        fig_name = '{}/all__{}'.format(plot_directory_mass, fig_num) + '.png'
        plot_mass_called(human_mass_list_all, plot_title, fig_name)
        print("mass distribution plotted for {} for all existing agents".format(i))

    else:
        human_mass_list_all = []
        all_existing_humans = 0
        k = 0
        for agent in humans:
            # if agent.unique_id > 10000:
            if agent.mass != 0:
                human_mass_list_all.append(agent.mass)
                k += 1
        all_existing_humans = k

        # human_mass_list_all = k
        #
        # plot_title = r'Mass distribution - ' \
        #              r'uid more than 10k' + '\n' + 'Number of agents = {}'.format(len(human_mass_list_beyond_10k))
        # fig_name = '{}/more_than_10k__{}'.format(plot_directory_mass, i) + '.png'
        # plot_mass_called(human_mass_list_beyond_10k, plot_title, fig_name)
        # print("mass distribution plotted for {} uid more than 10k".format(i))
        #
    
        plot_title = r'Mass distribution - ' \
                     r'all existing agents' + '\n' + 'at {}th time step'.format(i)
        # + '\n' + 'Number of agents = {}'.format(all_existing_humans)
        
        fig_num = str(i).zfill(5)
        fig_name = '{}/all__{}'.format(plot_directory_mass, fig_num) + '.png'
        plot_mass_called(human_mass_list_all, plot_title, fig_name)
        print("mass distribution plotted for {} for all existing agents".format(i))

    
    
    
    
    
    
    
    
    
    
    # Old code functioning
    # human_mass_list = []
    # j = 0
    # if i < 6000:
    #     for agent in humans:
    #         if agent.unique_id <= 10000:
    #             if agent.mass != 0:
    #                 human_mass_list.append(agent.mass)
    #                 j += 1
    #     plot_title = r'Mass distribution - upto uid 10k' + '\n' + 'Number of agents = {}'.format(j)
    #     fig_name = '{}/less_than_10k__{}'.format(plot_directory_mass, i) + '.png'
    #     plot_mass_called(human_mass_list, plot_title, fig_name)
    #     print("mass distribution plotted for {} upto uid 10k".format(i))
    # else:
    #     pass
    #
    # human_mass_list = []
    # j = 0
    # for agent in humans:
    #     if agent.unique_id > 10000:
    #         if agent.mass != 0:
    #             human_mass_list.append(agent.mass)
    #             j += 1
    # plot_title = r'Mass distribution - uid more than 10k' + '\n' + 'Number of agents = {}'.format(j)
    # fig_name = '{}/more_than_10k__{}'.format(plot_directory_mass, i) + '.png'
    # plot_mass_called(human_mass_list, plot_title, fig_name)
    # print("mass distribution plotted for {} uid more than 10k".format(i))
    #
    # human_mass_list = []
    # j = 0
    # for agent in humans:
    #     if agent.mass != 0:
    #         human_mass_list.append(agent.mass)
    #         j += 1
    # plot_title = r'Mass distribution - all existing agents' + '\n' + 'Number of agents = {}'.format(j)
    # fig_name = '{}/all__{}'.format(plot_directory_mass, i) + '.png'
    # plot_mass_called(human_mass_list, plot_title, fig_name)
    # print("mass distribution plotted for {} for all existing agents".format(i))



def plot_mass_called(human_mass_list, plot_title, fig_name):
    binwidth = 2.5e-6
    plt.hist(human_mass_list, bins=np.arange(0, 0.00025, binwidth),
             hold=True, facecolor='green', alpha=0.5)
    
    plt.xlabel('mass')
    plt.ylabel('freq')
    axes = plt.gca()
    axes.set_xlim([0, 0.00025])
    axes.set_ylim([0, 2400])
    plt.title(plot_title)
    plt.grid(True)
    plt.savefig(fig_name)
    plt.close()


def update_mass(humans, correction):
    # This function applies mass correction.
    for agent in humans:
        if agent.mass != 0:
            agent.mass += correction


def per_capita_mass_calculation(humans):
    # This function computes per capita mass for humans.
    total_mass = 0
    number = 0
    for agent in humans:
        total_mass += agent.mass  # add masses of all the agents
        if agent.mass != 0:
            number += 1  # only agents with non-zero mass are considered
    
    pcm = total_mass / number  # pcm = per capita mass
    
    return pcm


def human_compartment_mass_calculation(humans):
    total_mass = 0
    i = 0

    for agent in humans:
        # print((agent.mass), i)
        total_mass += agent.mass  # add masses of all the agents
        i += 1
    return total_mass


def mass_distribution(num_agents, total_mass):
    # This function takes number of agents to be created, and total mass from which agents are to be created
    # as the input, and returns normally distributed masses whose sum is equal to total mass provided
    
    import random
    mu = total_mass / num_agents
    
    if num_agents > 100:
        bins = 50
    else:
        bins = num_agents
    sigma = mu * 0.13746774193548386
    
    data = [random.gauss(mu, sigma) for c in range(num_agents)]
    
    # plot_mass_distribution(mu, sigma, bins, num_agents, data)
    
    return data


def plot_mass_distribution(mu, sigma, bins, num_agents, data):
    # This function plots histogram of the masses of the agents created.
    # This is generally called from mass_distribution,
    # but if required can be called from the main files as well. Then all the required information should be provided.
    
    import matplotlib.pyplot as plt
    plt.hist(data, bins, facecolor='green', alpha=0.5)
    plt.xlabel('mass')
    plt.ylabel('freq')
    plt.title(r'Mass distribution')
    plt.grid(True)
    
    print("Sample mean = %g; Stddev = %g; max = %g; min = %g for %i values"
          % (mu, sigma, max(data), min(data), num_agents))
    
    print(sum(data))
    plt.show()


def plot3(a1, a2, case, species, axis_label, j, show_plots, save_plots, plot_directory, additional_name_tags,
          label_1, label_2, range_from, range_to, linestyle_cwd, linestyle_sd, color_cwd, color_sd):

    length_flag = len(axis_label)
    
    plt.subplot(length_flag, 1, 1)
    l1 = plt.plot(a1.ix[range_from:range_to, j + 0], linestyle=linestyle_sd, color=color_sd, label=label_1)
    l2 = plt.plot(a2.ix[range_from:range_to, j + 0], linestyle=linestyle_cwd, color=color_cwd, label=label_2)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, min(0, 0.9 * min(a1.ix[range_from:range_to, j + 0]), 0.9 * min(a2.ix[range_from:range_to, j + 0])),
              max(max(1.1 * a1.ix[range_from:range_to, j + 0]), 1.1 * max(a2.ix[range_from:range_to, j + 0]))))
    plt.legend(handles=l1 + l2)
    plt.title(case + additional_name_tags + '\n' + species)
    plt.ylabel(axis_label[0])
    
    plt.subplot(length_flag, 1, 2)
    plt.plot(a1.ix[range_from:range_to, j + 1], linestyle=linestyle_sd, color=color_sd)
    plt.plot(a2.ix[range_from:range_to, j + 1], color=color_cwd, linestyle=linestyle_cwd)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, min(0, 0.9 * min(a1.ix[range_from:range_to, j + 1]), 0.9 * min(a2.ix[range_from:range_to, j + 1])),
              max(max(1.1 * a1.ix[range_from:range_to, j + 1]), 1.1 * max(a2.ix[range_from:range_to, j + 1]))))
    # if required remove zero from min limit of y
    
    plt.ylabel(axis_label[1])
    
    plt.subplot(length_flag, 1, 3)
    plt.plot(a1.ix[range_from:range_to, j + 2], linestyle=linestyle_sd, color=color_sd)
    plt.plot(a1.ix[range_from:range_to, j + 2], color=color_cwd, linestyle=linestyle_cwd)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, min(0, 0.9 * min(a1.ix[range_from:range_to, j + 2]), 0.9 * min(a2.ix[range_from:range_to, j + 2])),
              max(max(1.1 * a1.ix[range_from:range_to, j + 2]), 1.1 * max(a2.ix[range_from:range_to, j + 2]))))
    # if required remove zero from min limit of y
    plt.ylabel(axis_label[2])
    fig = plt.gcf()
    fig.set_size_inches(12, 6)

    if show_plots == 1:
        plt.show()
    if save_plots == 1:
        plt.savefig('{}/{}__'.format(plot_directory, species) + "{}".format(case) + '.png')
        plt.close()


def plot2(a1, a2, case, species, axis_label, j, show_plots, save_plots, plot_directory, additional_name_tags,
          label_1, label_2, range_from, range_to, linestyle_cwd, linestyle_sd, color_cwd, color_sd):

    length_flag = len(axis_label)
    
    plt.subplot(length_flag, 1, 1)
    l1 = plt.plot(a1.ix[range_from:range_to, j + 0], linestyle=linestyle_sd, color=color_sd, label=label_1)
    l2 = plt.plot(a2.ix[range_from:range_to, j + 0], color_cwd, linestyle=linestyle_cwd, label=label_2)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, min(0, 0.9 * min(a1.ix[range_from:range_to, j + 0]), 0.9 * min(a2.ix[range_from:range_to, j + 0])),
              max(max(1.1 * a1.ix[range_from:range_to, j + 0]), 1.1 * max(a2.ix[range_from:range_to, j + 0]))))
    # if required remove zero from min limit of y
    plt.legend(handles=l1 + l2)
    plt.title(case + additional_name_tags + '\n' + species)
    plt.ylabel(axis_label[0])
    
    plt.subplot(length_flag, 1, 2)
    plt.plot(a1.ix[range_from:range_to, j + 1], linestyle=linestyle_sd, color=color_sd)
    plt.plot(a2.ix[range_from:range_to, j + 1], color_cwd, linestyle=linestyle_cwd)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, min(0, 0.9 * min(a1.ix[range_from:range_to, j + 1]), 0.9 * min(a2.ix[range_from:range_to, j + 1])),
              max(max(1.1 * a1.ix[range_from:range_to, j + 1]), 1.1 * max(a2.ix[range_from:range_to, j + 1]))))
    # if required remove zero from min limit of y
    plt.ylabel(axis_label[1])
    fig = plt.gcf()
    fig.set_size_inches(12, 6)

    
    if show_plots == 1:
        plt.show()
    if save_plots == 1:
        plt.savefig('{}/{}__'.format(plot_directory, species) + "{}".format(case) + '.png')
        plt.close()




def plot_results(case, data_cwd, show_plots, save_plots, plot_directory, label_1, label_2, additional_name_tags):
    # Get SD data
    cwd = os.getcwd()
    # print(cwd)
    cwd += '/SD_Data'
    sd_data = pd.read_csv(cwd + '/{}.csv'.format(case), sep='\t', header=None)
    # sd_data = pd.read_csv(cwd + '/{}.txt'.format(case), sep=' ', header=None)

    data1 = sd_data
    
    data2 = data_cwd
    

    # Get cwd data
    species = ["Plants", "Herbivores", "Carnivores", "Resource Pools", "Human Compartment"]
    axis_label = [["P1", "P2", "P3"], ["H1", "H2", "H3"], ["C1", "C2"], ["RP", "IRP"], ["N_HH", "HH"]]
    j = 0

    # time range in which data needs to be plotted
    range_from = 0  # starting time-step
    range_to = len(data1[0])  # up to time-step

    # Plot formatting options
    # line style
    linestyle_sd = ":"  # SD data/ benchmark data
    linestyle_cwd = ":"  # Hybrid data/current data
    # line color
    color_sd = 'r'  # SD data/ benchmark data
    color_cwd = 'g'  # Hybrid data/current data

    # plt.plot(data2.iloc[:, -1])
    # plt.ylabel("system mass")
    # plt.savefig('{}/{}__'.format(plot_directory, "system mass") + "{}".format(case) + '.png')
    # plt.close()

    j = 0
    for i in range(len(species)):
        if len(axis_label[i]) == 3:
            plot3(data1.ix[:, j:j + 2], data2.ix[:, j:j + 2],
                  case, species[i], axis_label[i], j, show_plots, save_plots, plot_directory, additional_name_tags,
                  label_1, label_2, range_from, range_to, linestyle_cwd, linestyle_sd, color_cwd, color_sd)
            j += 3
        elif len(axis_label[i]) == 2:
            plot2(data1.ix[:, j:j + 1], data2.ix[:, j:j + 1],
                  case, species[i], axis_label[i], j, show_plots, save_plots, plot_directory, additional_name_tags,
                  label_1, label_2, range_from, range_to, linestyle_cwd, linestyle_sd, color_cwd, color_sd)
            j += 2


def worldview_trend_creation(x, width, center, exponent_multiplier, simulation_time, circulation_fraction_initial,
                             maximum_achievable_circulation):
    # sigmoid function
    sigmoid_mid_point = simulation_time * center
    C = 1
    Q = exponent_multiplier
    B = 1 / width
    M = sigmoid_mid_point
    t = x
    denomenator_exponent = 1
    phi_0 = 1 / (C + Q * np.exp(-B * (0 - M))) ** (1 / denomenator_exponent)
    phi_1 = 1 / (C + Q * np.exp(-B * (simulation_time - M))) ** (1 / denomenator_exponent)
    A = circulation_fraction_initial - (maximum_achievable_circulation - circulation_fraction_initial) / (
                phi_1 - phi_0) * phi_0
    K = A + (maximum_achievable_circulation - circulation_fraction_initial) / (phi_1 - phi_0)

    circulation_trend_local = np.array(
        A + (K - A) / (C + Q * np.exp(-B * (t - M))) ** (1 / denomenator_exponent)).tolist()

    return circulation_trend_local


def special_plots_worldviews():
    x_array = np.linspace(0, 10401, 10401)
    local_linewidth = 4

    growth_rate = 15
    maximum = 9
    delay = .3
    mac = .05 + maximum / 10
    circulation_fraction_initial = .17
    single_worldview = worldview_trend_creation(x=x_array, center=delay, width=52 * growth_rate,
                                                exponent_multiplier=1,
                                                maximum_achievable_circulation=mac,
                                                circulation_fraction_initial=circulation_fraction_initial,
                                                simulation_time=10400)

    plt.plot(np.linspace(0, 200, 10401), single_worldview, linewidth=local_linewidth)

    growth_rate = 5
    maximum = 9
    delay = .3
    mac = .05 + maximum / 10
    circulation_fraction_initial = .17
    single_worldview = worldview_trend_creation(x=x_array, center=delay, width=52 * growth_rate,
                                                exponent_multiplier=1,
                                                maximum_achievable_circulation=mac,
                                                circulation_fraction_initial=circulation_fraction_initial,
                                                simulation_time=10400)

    plt.plot(np.linspace(0, 200, 10401), single_worldview, linewidth=local_linewidth)

    growth_rate = 5
    maximum = 4
    delay = .7
    mac = .05 + maximum / 10
    circulation_fraction_initial = .17
    single_worldview = worldview_trend_creation(x=x_array, center=delay, width=52 * growth_rate,
                                                exponent_multiplier=1,
                                                maximum_achievable_circulation=mac,
                                                circulation_fraction_initial=circulation_fraction_initial,
                                                simulation_time=10400)

    plt.plot(np.linspace(0, 200, 10401), single_worldview, linewidth=local_linewidth)

    growth_rate = 15
    maximum = 4
    delay = .7
    mac = .05 + maximum / 10
    circulation_fraction_initial = .17
    single_worldview = worldview_trend_creation(x=x_array, center=delay, width=52 * growth_rate,
                                                exponent_multiplier=1,
                                                maximum_achievable_circulation=mac,
                                                circulation_fraction_initial=circulation_fraction_initial,
                                                simulation_time=10400)

    plt.plot(np.linspace(0, 200, 10401), single_worldview, linewidth=local_linewidth)

    growth_rate = 10
    maximum = 6
    delay = .5
    mac = .05 + maximum / 10
    circulation_fraction_initial = .17
    single_worldview = worldview_trend_creation(x=x_array, center=delay, width=52 * growth_rate,
                                                exponent_multiplier=1,
                                                maximum_achievable_circulation=mac,
                                                circulation_fraction_initial=circulation_fraction_initial,
                                                simulation_time=10400)

    plt.plot(np.linspace(0, 200, 10401), single_worldview, linewidth=local_linewidth)

    growth_rate = 10
    maximum = 7
    delay = .5
    mac = .05 + maximum / 10
    circulation_fraction_initial = .17
    single_worldview = worldview_trend_creation(x=x_array, center=delay, width=52 * growth_rate,
                                                exponent_multiplier=1,
                                                maximum_achievable_circulation=mac,
                                                circulation_fraction_initial=circulation_fraction_initial,
                                                simulation_time=10400)

    plt.plot(np.linspace(0, 200, 10401), single_worldview, linewidth=local_linewidth)
