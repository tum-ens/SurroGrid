from urbs.identify import *
import copy
import numpy as np
from datetime import date
import random
import re

def remove_battery(data):
    # de-flexiblise electricity demand by disallowing battery investment.
    # Thinning electrification can leave a grid with no battery rows at all,
    # in which case there is nothing to disallow and .loc raises on the key.
    if 'battery_private' not in data['storage'].index.get_level_values(2):
        return data
    data['storage'].loc[pd.IndexSlice[:, :, 'battery_private', :], ['inst-cap-p']] = 0
    data['storage'].loc[pd.IndexSlice[:, :, 'battery_private', :], ['inst-cap-c']] = 0
    data['storage'].loc[pd.IndexSlice[:, :, 'battery_private', :], ['cap-up-p']] = 0
    data['storage'].loc[pd.IndexSlice[:, :, 'battery_private', :], ['cap-up-c']] = 0
    return data

def remove_heat_storage(data):
    # de-flexiblise heat demand by disallowing heat-storage investment; as above,
    # a grid can end up with no heat-storage rows once electrification is thinned
    if 'heat_storage' not in data['storage'].index.get_level_values(2):
        return data
    data['storage'].loc[pd.IndexSlice[:, :, 'heat_storage', :], ['cap-up-p']] = 0
    data['storage'].loc[pd.IndexSlice[:, :, 'heat_storage', :], ['cap-up-c']] = 0
    data['storage'].loc[pd.IndexSlice[:, :, 'heat_storage', :], ['inst-cap-p']] = 0
    data['storage'].loc[pd.IndexSlice[:, :, 'heat_storage', :], ['inst-cap-c']] = 0
    return data

def remove_mob_flexibility(data):
    # remove mobility flexibility
    for (stf, sit, sto, com) in data['storage'].index:
        if sto.startswith('mobility'):
            data['storage'].loc[(stf, sit, sto, com), 'inst-cap-p'] = 0
            data['storage'].loc[(stf, sit, sto, com), 'inst-cap-c'] = 0
            data['storage'].loc[(stf, sit, sto, com), 'cap-up-p'] = 0
            data['storage'].loc[(stf, sit, sto, com), 'cap-up-c'] = 0

    return data

def create_xls_file_labels(uncoordinated, flexible, mode, data):
    coordination_text = '_uncoordinated' if uncoordinated else '_coordinated'
    flexible_text = '_flexible' if flexible else '_inflexible'

    if mode['14a']:
        regulation_text = '_14a'
    else:
        regulation_text = ''

    if data['site']['power_price_kw'].dropna().iloc[0] > 0.01:
        capacity_price_text = '_capacity_price'
    else:
        capacity_price_text = ''

    return coordination_text, flexible_text, regulation_text, capacity_price_text

def create_h5_file_labels(input_files, electrification):
    grid_text = input_files.split('_')[2]
    paradigm_text = input_files.split('_')[3][:-4]
    if electrification == 1:
        electrification_text = 'full'
    elif electrification == 0.5:
        electrification_text = 'half'
    elif electrification == 0.25:
        electrification_text = 'quarter'
    else:
        electrification_text = 'unknown'
    return grid_text, paradigm_text, electrification_text

def add_import_hp_bev_process(data):
    year = date.today().year

    #add import_hp as a process
    data['process'].loc[year, data['trafo_node'], 'import_hp'] = \
        data['process'].loc[year, data['trafo_node'], 'import'].copy(deep=True)
    data['process_commodity'].loc[year, 'import_hp', 'electricity_hp_import', 'In'] = (1, np.nan)
    data['process_commodity'].loc[year, 'import_hp', 'electricity_hp', 'Out'] = (1, np.nan)

    data['buy_sell_price']['electricity_hp_import'] = \
        data['buy_sell_price']['electricity_import'] * \
        data['global_prop'].loc[pd.IndexSlice[:, 'heatpump_elec_mult'], 'value'].iloc[0]
    data['commodity'].loc[year, data['trafo_node'], 'electricity_hp_import', 'Buy'] = \
        data['commodity'].loc[year, data['trafo_node'], 'electricity_import', 'Buy'].copy(deep=True)

    #add import_bev as a process
    data['process'].loc[year, data['trafo_node'], 'import_bev'] = \
        data['process'].loc[year, data['trafo_node'], 'import'].copy(deep=True)
    data['process_commodity'].loc[year, 'import_bev', 'electricity_bev_import', 'In'] = (1, np.nan)
    data['process_commodity'].loc[year, 'import_bev', 'electricity_bev', 'Out'] = (1, np.nan)
    data['buy_sell_price']['electricity_bev_import'] = \
        data['buy_sell_price']['electricity_import']
    data['commodity'].loc[year, data['trafo_node'], 'electricity_bev_import', 'Buy'] = \
        data['commodity'].loc[year, data['trafo_node'], 'electricity_import', 'Buy'].copy(deep=True)

    return data


def add_electricity_hp_bev_commodity(data, comtype='Stock'):
    # add commodity "electricity_hp" to all sites
    year = date.today().year
    demand_nodes = set([sit for (sit, demand) in data['demand'].columns])
    try:
        demand_nodes.remove(data['trafo_node'])
    except KeyError:
        pass
    try:
        demand_nodes.remove(data['mainbusbar_node'])
    except KeyError:
        pass

    if (comtype == ['Stock', 'Demand']):
        for site in data['site'].index.get_level_values(1):
            if site in demand_nodes:
                data['commodity'].loc[year, site, 'electricity_hp', 'Demand'] = (0, 0, 0)
                data['commodity'].loc[year, site, 'electricity_bev', 'Demand'] = (0, 0, 0)
            else:
                data['commodity'].loc[year, site, 'electricity_hp', 'Stock'] = (0, 0, 0)
                data['commodity'].loc[year, site, 'electricity_bev', 'Stock'] = (0, 0, 0)
    else:
        for site in data['site'].index.get_level_values(1):
            data['commodity'].loc[year, site, 'electricity_hp', comtype] = (0, 0, 0)
            data['commodity'].loc[year, site, 'electricity_bev', comtype] = (0, 0, 0)

    return data

def modify_hp_bev_processes(data):
    year = date.today().year
    demand_nodes = set([sit for (sit, demand) in data['demand'].columns])
    try:
        demand_nodes.remove(data['trafo_node'])
    except KeyError:
        pass
    try:
        demand_nodes.remove(data['mainbusbar_node'])
    except KeyError:
        pass

    # modify the input commodities of charging stations to usable commodities
    # data['process_commodity'][data['process_commodity'].index.get_level_values(1) == 'heatpump_air'].rename(
    #    index={'electricity': 'electricity_hp_usable'}, level=2, inplace=True)
    heat_pump_pro_com = data['process_commodity'][
        data['process_commodity'].index.get_level_values(1) == 'heatpump_air']
    heat_pump_pro_com.rename(index={'electricity': 'electricity_hp'}, level=2, inplace=True)
    data['process_commodity'].drop((year, 'heatpump_air', 'electricity', 'In'), inplace=True, axis=0)
    data['process_commodity'].drop((year, 'heatpump_air', 'common_heat', 'Out'), inplace=True, axis=0)
    data['process_commodity'].drop((year, 'heatpump_air', 'electricity-reactive', 'In'), inplace=True, axis=0)
    data['process_commodity'] = pd.concat([data['process_commodity'], heat_pump_pro_com])

    try:
        booster_pro_com = data['process_commodity'][
            data['process_commodity'].index.get_level_values(1) == 'heatpump_booster']
        booster_pro_com.rename(index={'electricity': 'electricity_hp'}, level=2, inplace=True)
        data['process_commodity'].drop((year, 'heatpump_booster', 'electricity', 'In'), inplace=True, axis=0)
        data['process_commodity'].drop((year, 'heatpump_booster', 'common_heat', 'Out'), inplace=True, axis=0)
        data['process_commodity'] = pd.concat([data['process_commodity'], booster_pro_com])
    except:
        pass

    charging_stations_pro_com = data['process_commodity'][
        data['process_commodity'].index.get_level_values(1).str.startswith('charging')]
    charging_stations_pro_com = charging_stations_pro_com.rename(index={'electricity': 'electricity_bev'},
                                                                 level=2)
    # drop existing charging stations
    unique_charging_stations = [ucs for (stf, ucs, com, direction) in data['process_commodity'].index \
                                if ucs.startswith('charging_station') and direction == 'In']
    for ucs in unique_charging_stations:
        data['process_commodity'].drop((year, ucs, 'electricity', 'In'), inplace=True, axis=0)
        data['process_commodity'].drop((year, ucs, 'mobility' + ucs[16:], 'Out'), inplace=True, axis=0)
    data['process_commodity'] = pd.concat([data['process_commodity'], charging_stations_pro_com])

    return data

def add_elec_to_hp_bev_process(data):
    year = date.today().year

    # add process elec_hp_to_usable to all buildings (same properties as curtailment)
    elec_to_hp_usable_pro = data['process'][
        data['process'].index.get_level_values(2) == 'curtailment']
    elec_to_hp_usable_pro = elec_to_hp_usable_pro.copy(deep=True)
    elec_to_hp_usable_pro.rename(index={'curtailment': 'elec_to_elec_hp'}, level=2, inplace=True)
    data['process'] = pd.concat([data['process'], elec_to_hp_usable_pro])
    data['process'].loc[pd.IndexSlice[year, :, 'elec_to_elec_hp'], ('var-cost')] = 0.01
    # one cent offset, to motivate always
    # buying the electricity_hp and electricity_bev instead of  buying electricity indirectly

    data['process_commodity'].loc[year, 'elec_to_elec_hp', 'electricity', 'In'] = (1, np.nan)
    data['process_commodity'].loc[year, 'elec_to_elec_hp', 'electricity_hp', 'Out'] = (1, np.nan)

    # same procedure for elec_bev
    elec_to_bev_usable_pro = data['process'][
        data['process'].index.get_level_values(2) == 'curtailment']
    elec_to_bev_usable_pro = elec_to_bev_usable_pro.copy(deep=True)
    elec_to_bev_usable_pro.rename(index={'curtailment': 'elec_to_elec_bev'}, level=2, inplace=True)
    data['process'] = pd.concat([data['process'], elec_to_bev_usable_pro])
    data['process'].loc[pd.IndexSlice[year, :, 'elec_to_elec_bev'], ('var-cost')] = 0.01

    # one cent offset, to motivate always
    # buying the electricity_hp and electricity_bev instead of buying electricity and converting indirectly

    data['process_commodity'].loc[year, 'elec_to_elec_bev', 'electricity', 'In'] = (1, np.nan)
    data['process_commodity'].loc[year, 'elec_to_elec_bev', 'electricity_bev', 'Out'] = (1, np.nan)

    return data


def add_curtailment_heat(data):
    year = date.today().year

    # add process curtailment_heat to all buildings (same properties as curtailment)
    curtailment_heat_pro = data['process'][data['process'].index.get_level_values(2) == 'curtailment']
    curtailment_heat_pro = curtailment_heat_pro.copy(deep=True)
    curtailment_heat_pro.rename(index={'curtailment': 'curtailment_heat'}, level=2, inplace=True)
    data['process'] = pd.concat([data['process'], curtailment_heat_pro])
    data['process_commodity'].loc[year, 'curtailment_heat', 'space_heat', 'In'] = (1, np.nan)

    return data


def add_hp_bev_flows(data):
    ### copy transmission lines with resistance to model transmission lines for reactive power flows

    heatpump_flow_lines = data['transmission'][data['transmission'].index.get_level_values(4) == 'electricity']
    heatpump_flow_lines = heatpump_flow_lines.copy(deep=True)
    heatpump_flow_lines.rename(index={'electricity': 'electricity_hp'}, level=4, inplace=True)
    ### set costs to zero as lines are not really built -

    heatpump_flow_lines.loc[:, ['inv-cost', 'fix-cost', 'var-cost', 'decom-saving']] *= 0
    ### concat new line data
    data['transmission'] = pd.concat([data['transmission'], heatpump_flow_lines])

    bev_flow_lines = data['transmission'][data['transmission'].index.get_level_values(4) == 'electricity']
    bev_flow_lines = bev_flow_lines.copy(deep=True)
    bev_flow_lines.rename(index={'electricity': 'electricity_bev'}, level=4, inplace=True)

    ### set costs to zero as lines are not really built -
    bev_flow_lines.loc[:, ['inv-cost', 'fix-cost', 'var-cost', 'decom-saving']] *= 0
    ### concat new line data
    data['transmission'] = pd.concat([data['transmission'], bev_flow_lines])
    return data


def distributed_building_specific_import_processes(data):
    # add all building-specific import options for uncoordinated planning
    year = date.today().year
    trafo_node = data['process'].query("Process == 'import'").index.get_level_values(1)[0]
    mainbusbar_node = data['process'].query("Process == 'Q_feeder_central'").index.get_level_values(1)[0]

    demand_nodes = list(data["demand"].columns.get_level_values(0).unique())
    demand_nodes.append(mainbusbar_node)

    for building in demand_nodes:
        # Set commodity import and feed-in for every building with first prize found in commodity sheet
        data['commodity'].loc[year, building, 'electricity_feed_in', 'Sell'] = (data['commodity']['price'].loc[:, :, 'electricity_feed_in',:].iloc[0], np.inf)
        data['commodity'].loc[year, building, 'electricity_import', 'Buy'] = (data['commodity']['price'].loc[:, :, 'electricity_import',:].iloc[0], np.inf)

        # Set process for feed-in and import as found for transformer with max capacities at 999
        data['process'].loc[year, building, 'feed_in'] = data['process'].loc[year, trafo_node, 'feed_in']
        data['process'].loc[year, building, 'import'] = data['process'].loc[year, trafo_node, 'import']
        data['process'].loc[year, building, 'feed_in']['cap-up'] = 999
        data['process'].loc[year, building, 'feed_in']['inst-cap'] = 999
        data['process'].loc[year, building, 'import']['cap-up'] = 999
        data['process'].loc[year, building, 'import']['inst-cap'] = 999

        # Add reactive power import/export to each node
        data['process'].loc[year, building, 'Q_feeder_central'] = data['process'].loc[year, mainbusbar_node, 'Q_feeder_central']
        data['process'].loc[year, building, 'Q_feeder_central']['cap-up'] = 999
        data['process'].loc[year, building, 'Q_feeder_central']['inst-cap'] = 999

    return data

def remove_heat_in_random(data, electrification):
    demand_nodes = list(data["site"].index.get_level_values("Name").unique())
    random.seed(43)
    disable_nodes = random.sample(demand_nodes, round(len(demand_nodes) * (100 - electrification)/100))

    data['process']=data["process"][~(data['process'].index.get_level_values(1).isin(disable_nodes) &
                                      (data['process'].index.get_level_values(2)==('heatpump_air')))]
    data['process']=data["process"][~(data['process'].index.get_level_values(1).isin(disable_nodes) &
                                      (data['process'].index.get_level_values(2)==('heatpump_grd')))]
    data['process']=data["process"][~(data['process'].index.get_level_values(1).isin(disable_nodes) &
                                      (data['process'].index.get_level_values(2)==('heatpump_booster')))]
    data['process']=data["process"][~(data['process'].index.get_level_values(1).isin(disable_nodes) &
                                      (data['process'].index.get_level_values(2)==('Heat_dummy_space')))]
    data['process']=data["process"][~(data['process'].index.get_level_values(1).isin(disable_nodes) &
                                      (data['process'].index.get_level_values(2)==('Heat_dummy_water')))]
    
    data['commodity']=data["commodity"][~(data['commodity'].index.get_level_values(1).isin(disable_nodes) &
                                          (data['commodity'].index.get_level_values(2)=='common_heat'))]
    data['commodity']=data["commodity"][~(data['commodity'].index.get_level_values(1).isin(disable_nodes) &
                                          (data['commodity'].index.get_level_values(2)=='space_heat'))]
    data['commodity']=data["commodity"][~(data['commodity'].index.get_level_values(1).isin(disable_nodes) &
                                          (data['commodity'].index.get_level_values(2)=='water_heat'))]
    
    data['storage']=data["storage"][~(data['storage'].index.get_level_values(1).isin(disable_nodes) &
                                     (data['storage'].index.get_level_values(2)=="heat_storage"))]


    data["demand"] = data["demand"].loc[:, ~((data["demand"].columns.get_level_values(0).isin(disable_nodes)) &
                                           (data["demand"].columns.get_level_values(1) == "water_heat"))]
    data["demand"] = data["demand"].loc[:, ~((data["demand"].columns.get_level_values(0).isin(disable_nodes)) &
                                           (data["demand"].columns.get_level_values(1) == "space_heat"))]
    data["eff_factor"] = data["eff_factor"].loc[:, ~((data["eff_factor"].columns.get_level_values(0).isin(disable_nodes)) &
                                           (data["eff_factor"].columns.get_level_values(1) == "heatpump_air"))]
    data["eff_factor"] = data["eff_factor"].loc[:, ~((data["eff_factor"].columns.get_level_values(0).isin(disable_nodes)) &
                                           (data["eff_factor"].columns.get_level_values(1) == "heatpump_grd"))]

    # year = date.today().year
    # demand_nodes = [sit for (sit, demand) in data['demand'].columns if demand == 'space_heat']
    # random.seed(42)
    # unelectrified_heat_nodes = random.sample(demand_nodes, int(len(demand_nodes) * (1 - electrification)))
    # data['process'].loc[(year, unelectrified_heat_nodes, 'heatpump_air'), 'cap-up'] = 0

    # instead of HP, provide heat by another means, e.g. gas boiler
    # data['commodity'].loc[(year, unelectrified_heat_nodes, 'common_heat'), 'price'] = 0.12  # assumed gas price
    # data['commodity'].loc[(year, unelectrified_heat_nodes, 'common_heat'), 'max'] = np.inf # allow gas supply
    # data['commodity'].loc[(year, unelectrified_heat_nodes, 'common_heat'), 'maxperhour'] = np.inf # allow gas supply
    return data

def remove_pv_in_random(data, electrification):
    demand_nodes = list(data["site"].index.get_level_values("Name").unique())
    random.seed(43)
    disable_nodes = random.sample(demand_nodes, round(len(demand_nodes) * (100 - electrification)/100))

    data['process']=data["process"][~(data['process'].index.get_level_values(1).isin(disable_nodes) &
                                      data['process'].index.get_level_values(2).str.startswith('Rooftop PV'))]
    data['commodity']=data["commodity"][~(data['commodity'].index.get_level_values(1).isin(disable_nodes) &
                                          data['commodity'].index.get_level_values(2).str.startswith('solar'))]
    data["supim"] = data["supim"].loc[:, ~((data["supim"].columns.get_level_values(0).isin(disable_nodes)) &
                                           (data["supim"].columns.get_level_values(1).str.startswith('solar')))]
    data['storage']=data["storage"][~(data['storage'].index.get_level_values(1).isin(disable_nodes) &
                                     (data['storage'].index.get_level_values(2)=="battery_private"))]

    # # Set PV and associated battery to 0 upper cap to prevent from expansion
    # data['process'].loc[((data['process'].index.get_level_values(1).isin(disable_pv_nodes)) &
    #                         (data['process'].index.get_level_values(2).str.startswith('Rooftop PV'))), 'cap-up'] = 0
    # data['storage'].loc[((data['storage'].index.get_level_values(1).isin(disable_pv_nodes)) &
    #                     (data['storage'].index.get_level_values(2).str.startswith('battery_private'))), ['cap-up-c', "cap-up-p"]]=0

    return data

def remove_mobility_in_random(data, electrification):
    random.seed(44)
    all_cars = [col for col in data['demand'].columns if col[1].startswith('mobility')]
    unelectrified_cars = random.sample(all_cars, round(len(all_cars) * (100 - electrification)/100))

    for (site, car) in unelectrified_cars:
        match = re.search(r'\d+$', car)
        car_idx = match.group() if match else None

        data['process']=data["process"][~((data['process'].index.get_level_values(1)==site) &
                                        (data['process'].index.get_level_values(2)=="charging_station"+car_idx))]
        data['commodity']=data["commodity"][~((data['commodity'].index.get_level_values(1)==site) &
                                        (data['commodity'].index.get_level_values(2)=="mobility"+car_idx))]   
        data['storage']=data["storage"][~((data['storage'].index.get_level_values(1)==site) &
                                        (data['storage'].index.get_level_values(2)=="mobility_storage"+car_idx))]
        data["demand"] = data["demand"].loc[:, ~((data["demand"].columns.get_level_values(0)==site) &
                                            (data["demand"].columns.get_level_values(1) == "mobility"+car_idx))]
        data["eff_factor"] = data["eff_factor"].loc[:, ~((data["eff_factor"].columns.get_level_values(0)==site) &
                                            (data["eff_factor"].columns.get_level_values(1) == "charging_station"+car_idx))]


    # year = date.today().year
    # demand_nodes = [sit for (sit, demand) in data['demand'].columns if demand == 'space_heat']
    # random.seed(44)
    # all_cars = [col for col in data['demand'].columns if col[1].startswith('mobility')]
    # unelectrified_cars = random.sample(all_cars, int(len(all_cars) * (100 - electrification)/100))

    # for (site, car) in unelectrified_cars:
    #     car_idx = car[-1]
    #     data['process'].loc[(year, site, 'charging_station' + car_idx), 'inst-cap'] = 0
    #     data['process'].loc[
    #         (year, site, 'charging_station' + car_idx), 'cap-up'] = 0  # set charging_station capacity to zero
    #     data['commodity'].loc[(year, site, 'mobility' + car_idx, 'Demand'), 'price'] = 0.6  # 60 cent for public charging
    #     data['commodity'].loc[(year, site, 'mobility' + car_idx, 'Demand'), 'max'] = np.inf
    #     data['commodity'].loc[(year, site, 'mobility' + car_idx,
    #                            'Demand'), 'maxperhour'] = np.inf  # add stock availabiltiy for mobility commodity
        
    return data

def adopt_variable_tariffs(data, vartariff_nodes):
    year = date.today().year
    demand_nodes = [sit for (sit, demand) in data['demand'].columns if demand == 'space_heat']

    # rename import to import_var process for those nodes
    for building in vartariff_nodes:
        import_pro = data['process'][
            data['process'].index.get_level_values(2) == 'import']
        import_pro = import_pro[import_pro.index.get_level_values(1) == building]
        import_pro = import_pro.copy(deep=True)
        import_pro.rename(index={'import': 'import_var'}, level=2, inplace=True)
        data['process'].drop((year, building, 'import'), inplace=True, axis=0)
        data['process'] = pd.concat([data['process'], import_pro])

    # rename commodities "electricity_import" to "electricity_import_var" for those nodes
    for building in vartariff_nodes:
        electricity_import_com = data['commodity'][
            data['commodity'].index.get_level_values(2) == 'electricity_import']
        electricity_import_com = electricity_import_com[electricity_import_com.index.get_level_values(1) == building]
        electricity_import_com = electricity_import_com.copy(deep=True)
        electricity_import_com.rename(index={'electricity_import': 'electricity_import_var'}, level=2, inplace=True)
        data['commodity'].drop((year, building, 'electricity_import', 'Buy'), inplace=True, axis=0)
        data['commodity'] = pd.concat([data['commodity'], electricity_import_com])

    # add process-commodity for import_var (same as import, just different import commodity with variable prices)
    import_var_pro_com = data['process_commodity'][
        data['process_commodity'].index.get_level_values(1) == 'import']
    import_var_pro_com = import_var_pro_com.copy(deep=True)
    import_var_pro_com.rename(index={'electricity_import': 'electricity_import_var'}, level=2, inplace=True)
    import_var_pro_com.rename(index={'import': 'import_var'}, level=1, inplace=True)
    data['process_commodity'] = pd.concat([data['process_commodity'], import_var_pro_com])
    return data

def adopt_variable_tariffs_all(data):
    """
    Modify the 'buy_sell_price' DataFrame within the provided dataset by adopting variable tariffs.

    This function takes the dataset, modifies the 'buy_sell_price' DataFrame within
    by dropping the 'electricity_import' column and renaming the 'electricity_import_var' column to
    'electricity_import'. This leads to all buildings (100%) having variable tariffs.
    The modified dataset is then returned.

    Args:
        data: A dictionary containing the input data, where 'buy_sell_price' is expected
                     to be a DataFrame with columns 'electricity_import' and 'electricity_import_var'.

    Returns:
        data: The updated dataset with modifications applied to the 'buy_sell_price' DataFrame.
    """
    data['buy_sell_price'] = data['buy_sell_price'].drop(['electricity_import'], axis=1)
    data['buy_sell_price'] = data['buy_sell_price'].rename(columns={'electricity_import_var': 'electricity_import'})
    return data

def set_curtailment_limits(data_grid_plan):
    # set limits to DSO-side grid curtailment using the "availability" functionality
    # (it cannot be greater than the feed-in (minus demand) at each hour!)
    year = date.today().year
    demand_nodes = [sit for (sit, demand) in data_grid_plan['demand'].columns
                    if demand == 'electricity']
    try:
        demand_nodes.remove(data_grid_plan['trafo_node'])
    except ValueError:
        pass
    try:
        demand_nodes.remove(data_grid_plan['mainbusbar_node'])
    except ValueError:
        pass

    for building in demand_nodes:
        data_grid_plan['process'].loc[(year, building, 'curtailment'),'inst-cap'] = 1
        data_grid_plan['process'].loc[(year, building, 'curtailment'),'cap-up'] = 1

        data_grid_plan['availability'][(building, 'curtailment')] = - (data_grid_plan['demand'][
                                                                                        (building,
                                                                                         'electricity')])
        data_grid_plan['availability'][(building, 'curtailment')][ \
            data_grid_plan['availability'][(building, 'curtailment')] < 0] = 0
    return data_grid_plan


### START - HOODS-Grid routines

def delete_solar_supim_timeseries(data_grid_plan, building):
    bui_solar_comms = [solcom for (bui, solcom) in data_grid_plan['supim'].columns
                       if solcom.startswith('solar') and bui == building]
    for scs in bui_solar_comms:
        data_grid_plan['supim'].drop((building, scs), inplace=True, axis=1)
    return data_grid_plan

def delete_charging_station_eff_factors(data_grid_plan, building):
    year = date.today().year

    bui_charging_stations = [cs for (stf, bui, cs) in data_grid_plan['process'].index
                             if cs.startswith('charging_station') and bui == building
                             and stf == year]
    for bcs in bui_charging_stations:
        data_grid_plan['eff_factor'].drop((building, bcs), inplace=True, axis=1)
    return data_grid_plan

def delete_heatpump_eff_factors(data_grid_plan, building):
    data_grid_plan['eff_factor'].drop((building, 'heatpump_air'), inplace=True, axis=1)
    return data_grid_plan

def delete_non_electric_demand(data_grid_plan, building):
    #delete heat demands
    data_grid_plan['demand'].drop((building, 'space_heat'), inplace=True, axis=1)
    data_grid_plan['demand'].drop((building, 'water_heat'), inplace=True, axis=1)

    #delete all mobility demands
    bui_mob_demands = [dem for (bui, dem) in data_grid_plan['demand'].columns
                       if dem.startswith('mobility') and bui == building]
    for mob_demand in bui_mob_demands:
        data_grid_plan['demand'].drop((building, mob_demand), inplace=True, axis=1)
    return data_grid_plan

def shift_demand_to_elec(data_grid_plan, building, prob_cluster, mode, vartariff, vartariff_nodes=None):
    year = date.today().year

    if mode['14a']:
        # 14a mode: define electricity, electricity_hp, and electricity_bev demands separately.
        if vartariff > 0:
            # vartariff on: include import AND import_var for the participating buildings (otherwise only import)
            if building in vartariff_nodes: # buildings participating in vartariff
                data_grid_plan['demand'].loc[year, :][(building, 'electricity')].iloc[1:] = (
                        prob_cluster._result['tau_pro'].loc[:, year, building, 'import_var'].iloc[1:]
                        - prob_cluster._result['tau_pro'].loc[:, year, building, 'feed_in'].iloc[1:])
            else: # buildings NOT participating in vartariff
                data_grid_plan['demand'].loc[year, :][(building, 'electricity')].iloc[1:] = (
                        prob_cluster._result['tau_pro'].loc[:, year, building, 'import'].iloc[1:]
                        - prob_cluster._result['tau_pro'].loc[:, year, building, 'feed_in'].iloc[1:])

        else: # vartariff off: only include import, not import_var
            data_grid_plan['demand'].loc[year, :][(building, 'electricity')].iloc[1:] = (
                    prob_cluster._result['tau_pro'].loc[:, year, building, 'import'].iloc[1:]
                    - prob_cluster._result['tau_pro'].loc[:, year, building, 'feed_in'].iloc[1:])

        # demand for electricity_hp, initialize
        data_grid_plan['demand'][(building, 'electricity_hp')] = copy.deepcopy(
            data_grid_plan['demand'][(building, 'electricity')])
        # demand for electricity_hp, set
        data_grid_plan['demand'].loc[year, :][(building, 'electricity_hp')].iloc[1:] = \
            prob_cluster._result['tau_pro'].loc[:, year, building, 'import_hp'].iloc[1:]

        # demand for electricity_bev, initialize
        data_grid_plan['demand'][(building, 'electricity_bev')] = copy.deepcopy(
            data_grid_plan['demand'][(building, 'electricity')])
        # demand for electricity_bev, set
        data_grid_plan['demand'].loc[year, :][(building, 'electricity_bev')].iloc[1:] = \
            prob_cluster._result['tau_pro'].loc[:, year, building, 'import_bev'].iloc[1:]
    else:
        # no 14a: define only electricity demand (not the 14a commodities such as electricity_hp or electricity_bev).
        if vartariff > 0:
            if building in vartariff_nodes:
                data_grid_plan['demand'].loc[year, :][(building, 'electricity')].iloc[1:] = (
                        prob_cluster._result['tau_pro'].loc[:, year, building, 'import_var'].iloc[1:]
                        - prob_cluster._result['tau_pro'].loc[:, year, building, 'feed_in'].iloc[1:])
            else:
                data_grid_plan['demand'].loc[year, :][(building, 'electricity')].iloc[1:] = (
                        prob_cluster._result['tau_pro'].loc[:, year, building, 'import'].iloc[1:]
                        - prob_cluster._result['tau_pro'].loc[:, year, building, 'feed_in'].iloc[1:])
        else:
            data_grid_plan['demand'].loc[year, :][(building, 'electricity')].iloc[1:] = (
                    prob_cluster._result['tau_pro'].loc[:, year, building, 'import'].iloc[1:]
                    - prob_cluster._result['tau_pro'].loc[:, year, building, 'feed_in'].iloc[1:])

    # define reactive power demand as Q_feeder_central values from HOODS-Bui
    data_grid_plan['demand'].loc[year, :][(building, 'electricity-reactive')].iloc[1:] = \
        prob_cluster._result['e_pro_out'] \
            .loc[:, year, building, 'Q_feeder_central', 'electricity-reactive']

    return data_grid_plan

def delete_processes_for_hoods_grid(data_grid_plan, building, grid_curtailment):
    ### delete all processes besides Q-compensation, import and feed-in
    year = date.today().year

    #delete all pv processes
    bui_rooftop_pvs = [pv for (stf, bui, pv) in data_grid_plan['process'].index
                       if pv.startswith('Rooftop PV') and bui == building
                       and stf == year]
    for pvs in bui_rooftop_pvs:
        data_grid_plan['process'].drop((year, building, pvs), inplace=True, axis=0)

    # delete heat-related processes
    data_grid_plan['process'].drop((year, building, 'Heat_dummy_space'), inplace=True, axis=0)
    data_grid_plan['process'].drop((year, building, 'Heat_dummy_water'), inplace=True, axis=0)
    data_grid_plan['process'].drop((year, building, 'heatpump_air'), inplace=True, axis=0)
    try:
        data_grid_plan['process'].drop((year, building, 'heatpump_booster'), inplace=True, axis=0)
    except:
        pass

    # delete charging stations
    bui_charging_stations = [cs for (stf, bui, cs) in data_grid_plan['process'].index
                             if cs.startswith('charging_station') and bui == building and stf == year]
    for bcs in bui_charging_stations:
        data_grid_plan['process'].drop((year, building, bcs), inplace=True, axis=0)

    # delete curtailment, if a DSO-side curtailment is not allowed
    if not grid_curtailment:
        data_grid_plan['process'].drop((year, building, 'curtailment'), inplace=True, axis=0)
    else:
        # if a DSO-side PV curtailment is allowed, set a variable cost to it (DSO pays FIT to building)
        data_grid_plan['process'].loc[(year, building, 'curtailment'), 'var-cost'] = \
            data_grid_plan['buy_sell_price']['electricity_feed_in'].values[-1]

    # delete heat slack
    data_grid_plan['process'].drop((year, building, 'Slack_heat'), inplace=True, axis=0)
    return data_grid_plan

def delete_commodities_for_hoods_grid(data_grid_plan, building):
    year = date.today().year

    # delete heat commodities (demand and stocK)
    data_grid_plan['commodity'].drop((year, building, 'space_heat', 'Demand'), inplace=True, axis=0)
    data_grid_plan['commodity'].drop((year, building, 'water_heat', 'Demand'), inplace=True, axis=0)
    data_grid_plan['commodity'].drop((year, building, 'common_heat', 'Stock'), inplace=True, axis=0)

    # delete solar supim commodities
    bui_solar_comms = [solcom for (bui, solcom) in data_grid_plan['supim'].columns
                       if solcom.startswith('solar') and bui == building]
    for scs in bui_solar_comms:
        data_grid_plan['commodity'].drop((year, building, scs, 'SupIm'), inplace=True, axis=0)

    # delete mobility demand commodities
    bui_mob_commodities = [com for (stf, sit, com, typ) in data_grid_plan['commodity'].index
                           if com.startswith('mobility') and sit == building]
    for com in bui_mob_commodities:
        data_grid_plan['commodity'].drop((year, building, com, 'Demand'), inplace=True, axis=0)
    return data_grid_plan


def delete_procoms_for_hoods_grid(data_grid_plan, grid_curtailment):
    year = date.today().year

    data_grid_plan['process_commodity'].drop((year, 'Heat_dummy_space', 'common_heat', 'In'), inplace=True, axis=0)
    data_grid_plan['process_commodity'].drop((year, 'Heat_dummy_space', 'space_heat', 'Out'), inplace=True, axis=0)
    data_grid_plan['process_commodity'].drop((year, 'Heat_dummy_water', 'common_heat', 'In'), inplace=True, axis=0)
    data_grid_plan['process_commodity'].drop((year, 'Heat_dummy_water', 'water_heat', 'Out'), inplace=True, axis=0)
    data_grid_plan['process_commodity'].drop((year, 'heatpump_air', 'electricity', 'In'), inplace=True, axis=0)
    data_grid_plan['process_commodity'].drop((year, 'heatpump_air', 'common_heat', 'Out'), inplace=True, axis=0)
    data_grid_plan['process_commodity'].drop((year, 'heatpump_air', 'electricity-reactive', 'In'), inplace=True, axis=0)
    try:
        data_grid_plan['process_commodity'].drop((year, 'heatpump_booster', 'electricity', 'In'), inplace=True, axis=0)
        data_grid_plan['process_commodity'].drop((year, 'heatpump_booster', 'common_heat', 'Out'), inplace=True, axis=0)
    except:
        pass

    unique_charging_stations = [ucs for (stf, ucs, com, direction) in data_grid_plan['process_commodity'].index
                                if ucs.startswith('charging_station') and direction == 'In']
    for ucs in unique_charging_stations:
        data_grid_plan['process_commodity'].drop((year, ucs, 'electricity', 'In'), inplace=True, axis=0)
        data_grid_plan['process_commodity'].drop((year, ucs, 'mobility' + ucs[16:], 'Out'), inplace=True, axis=0)

    if not grid_curtailment:
        data_grid_plan['process_commodity'].drop((year, 'curtailment', 'electricity', 'In'), inplace=True, axis=0)
    data_grid_plan['process_commodity'].drop((year, 'Slack_heat', 'common_heat', 'Out'), inplace=True, axis=0)
    return data_grid_plan




def add_hp_bev_regulation_process(data_grid_plan, data_bui, var_cost=0):
    year = date.today().year
    demand_nodes = [sit for (sit, demand) in data_grid_plan['demand'].columns
                    if demand == 'electricity']
    try:
        demand_nodes.remove(data_grid_plan['trafo_node'])
    except ValueError:
        pass
    try:
        demand_nodes.remove(data_grid_plan['mainbusbar_node'])
    except ValueError:
        pass

    # GENERAL EXPLANATION:
    # 1. the regulation processes consumer "regulate_com" commodity, which costs exactly the same as import electricity,
    # plus a very small process variable cost (var_cost; default: 0.0001)
    # 2. there are two regulation processes: "hp_14a_steuve_regulate" and "bev_14a_steuve_regulate", for HP and BEV.
    # 3. these processes are only available at the time steps, where the demand for 14a commodities are higher than 4.2 kW
    # per HP or BEV. this is satisfied by setting its capacity to 1, and the availability to the surplus above 4.2 kW.
    # (if in a time step the HP imports 6 kW, then the availability will be equal to 6 - 4.2 = 1.8 kW)
    # if this surplus is negative (demand < 4.2 kW), the availability will be set to zero (no regulation possible).
    # 4. these processes generate the corresponding 14a commodity, effectively reducing its import from the grid
    # (:= reduced grid load)

    for building in demand_nodes:
        # HP downregulation process
        data_grid_plan['commodity'].loc[year, building, 'regulate_com', 'Buy'] = (1, np.inf, np.inf)
        data_grid_plan['process'].loc[year, building, 'hp_14a_steuve_regulate'] = copy.deepcopy(
            data_grid_plan['process'].loc[
                year, data_grid_plan['trafo_node'], 'import'])
        data_grid_plan['process'].loc[year, building, 'hp_14a_steuve_regulate']['inst-cap'] = 1
        data_grid_plan['process'].loc[year, building, 'hp_14a_steuve_regulate']['cap-up'] = 1
        data_grid_plan['process'].loc[year, building, 'hp_14a_steuve_regulate']['var-cost'] = var_cost
        data_grid_plan['availability'][(building, 'hp_14a_steuve_regulate')] = (data_grid_plan['demand'][
                                                                                (building,
                                                                                 'electricity_hp')] - 4.2)
        data_grid_plan['availability'][(building, 'hp_14a_steuve_regulate')][ \
            data_grid_plan['availability'][(building, 'hp_14a_steuve_regulate')] < 0] = 0

        # BEV downregulation process
        data_grid_plan['process'].loc[year, building, 'bev_14a_steuve_regulate'] = copy.deepcopy(
            data_grid_plan['process'].loc[
                year, data_grid_plan['trafo_node'], 'import'])
        data_grid_plan['process'].loc[year, building, 'bev_14a_steuve_regulate']['inst-cap'] = 1
        data_grid_plan['process'].loc[year, building, 'bev_14a_steuve_regulate']['cap-up'] = 1
        data_grid_plan['process'].loc[year, building, 'bev_14a_steuve_regulate']['var-cost'] = var_cost
        bui_car_count = data_bui['process'].loc[year, building, :]
        bui_car_count = len(bui_car_count[bui_car_count.index.get_level_values(0).str.startswith('charging')])
        data_grid_plan['availability'][(building, 'bev_14a_steuve_regulate')] = (data_grid_plan['demand'][
                                                                                     (building,
                                                                                      'electricity_bev')] - 4.2 * bui_car_count)
        data_grid_plan['availability'][(building, 'bev_14a_steuve_regulate')][ \
            data_grid_plan['availability'][(building, 'bev_14a_steuve_regulate')] < 0] = 0

    data_grid_plan['buy_sell_price']['regulate_com'] = data_grid_plan['buy_sell_price'][
                                                           'electricity_import'] * 1.00
    data_grid_plan['process_commodity'].loc[year, 'hp_14a_steuve_regulate', 'regulate_com', 'In'] = (
        1, np.nan)
    data_grid_plan['process_commodity'].loc[year, 'bev_14a_steuve_regulate', 'regulate_com', 'In'] = (
        1, np.nan)
    data_grid_plan['process_commodity'].loc[year, 'hp_14a_steuve_regulate', 'electricity_hp', 'Out'] = \
        (1, np.nan)
    data_grid_plan['process_commodity'].loc[year, 'hp_14a_steuve_regulate', 'electricity-reactive', 'Out'] = \
        (np.tan(np.arccos(data_grid_plan['global_prop'].loc[year, 'PF_HP']['value'])), np.nan)
    data_grid_plan['process_commodity'].loc[year, 'bev_14a_steuve_regulate', 'electricity_bev', 'Out'] = \
        (1, np.nan)

    return data_grid_plan


### START - HOODS-Bui-React routines
def limit_hp_for_hoods_bui_react(data_hp_react, data, building, prob_cluster, prob_grid_plan):

    # replace zero capacities by 0.01 to avoid division by zero later
    cap_hp = (0.01 if prob_cluster._result['cap_pro'].loc[:, building, 'heatpump_air'].values[0] == 0
              else prob_cluster._result['cap_pro'].loc[:, building, 'heatpump_air'].values[0])
    cap_booster = (0.01 if prob_cluster._result['cap_pro'].loc[:, building, 'heatpump_booster'].values[0] == 0
                   else prob_cluster._result['cap_pro'].loc[:, building, 'heatpump_booster'].values[0])


    # See equations (2.89) and (2.91) in the dissertation of Soner Candas for the interpretation of the following
    # operations

    # the capacity limitations are imposed using the "availability" feature
    # initialize availabilities
    data_hp_react['availability'][(building, 'heatpump_air')] = data_hp_react['availability'][
        (data['trafo_node'], 'import')].values
    data_hp_react['availability'][(building, 'heatpump_booster')] = data_hp_react['availability'][
        (data['trafo_node'], 'import')].values

    if prob_cluster._result['cap_pro'].loc[:, building, 'heatpump_air'].values[0] <= 4.2: # Case 1
        # no restrictions if cap is smaller than 4.2
        data_hp_react['availability'][(building, 'heatpump_air')].iloc[1:] = 1 # Case 1, no restriction
        if (prob_cluster._result['cap_pro'].loc[:, building, 'heatpump_air'].values[0] + # Case 1.1, no restriction
            prob_cluster._result['cap_pro'].loc[:, building, 'heatpump_booster'].values[0]) <= 4.2:
            # total cap is smaller too, no restriction on peaker heater.
            data_hp_react['availability'][(building, 'heatpump_booster')].iloc[1:] = 1
        else:
            # total cap exceeds 4.2 (case 1.2), peaker may be curtailed:
            data_hp_react['availability'][(building, 'heatpump_booster')].iloc[1:] = \
                (prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'hp_14a_steuve_regulate',
                 'electricity_hp'] <= 0) * 1 + \
                (prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'hp_14a_steuve_regulate',
                 'electricity_hp'] > 0) * \
                (prob_cluster._result['e_pro_in'].loc[
                 :, :, building, 'heatpump_booster', 'electricity_hp'].values -
                 prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'hp_14a_steuve_regulate',
                 'electricity_hp'].values) / cap_booster

    else: # heat pump capacity larger than 4.2, regulation possible (Case 2)
        # restrict main HP cap (equation 2.90)
        data_hp_react['availability'][(building, 'heatpump_air')].iloc[1:] = \
            (prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'hp_14a_steuve_regulate',
             'electricity_hp'] <= prob_cluster._result['e_pro_in'].loc[
                                  :, :, building, 'heatpump_booster', 'electricity_hp'].values) * 1 + \
            (prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'hp_14a_steuve_regulate',
             'electricity_hp'] > prob_cluster._result['e_pro_in'].loc[
                                 :, :, building, 'heatpump_booster', 'electricity_hp'].values) * \
            (prob_cluster._result['e_pro_in'].loc[
             :, :, building, 'heatpump_air', 'electricity_hp'].values -
             prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'hp_14a_steuve_regulate',
             'electricity_hp'] +
             prob_cluster._result['e_pro_in'].loc[
             :, :, building, 'heatpump_booster', 'electricity_hp'].values) / cap_hp

        # restrict HP peaker cap (equation 2.91)
        data_hp_react['availability'][(building, 'heatpump_booster')].iloc[1:] = \
            (prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'hp_14a_steuve_regulate',
             'electricity_hp'] == 0) * 1 + \
            ((prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'hp_14a_steuve_regulate',
              'electricity_hp'] > 0) &
             (prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'hp_14a_steuve_regulate',
              'electricity_hp'] <= prob_cluster._result['e_pro_in'].loc[
                                   :, :, building, 'heatpump_booster', 'electricity_hp'].values)) * \
            (prob_cluster._result['e_pro_in'].loc[
             :, :, building, 'heatpump_booster', 'electricity_hp'].values -
             prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'hp_14a_steuve_regulate',
             'electricity_hp']) / cap_booster + \
            ((prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'hp_14a_steuve_regulate',
              'electricity_hp'] >
              prob_cluster._result['e_pro_in'].loc[:, :, building, 'heatpump_booster', 'electricity_hp'].values) * 0)
    return data_hp_react

def limit_bev_for_hoods_bui_react(data_hp_react, data, building, prob_cluster, prob_grid_plan, data_grid_plan):
    # See equation 2.92 in the dissertation of Soner Candas
    year = date.today().year
    bui_charging_stations = [cs for (stf, bui, cs) in data['process'].index
                             if cs.startswith('charging_station') and bui == building
                             and stf == year]
    for cs in bui_charging_stations:
        data_hp_react['availability'][(building, cs)] = data_hp_react['availability'][
            (data['trafo_node'], 'import')].values
        if prob_cluster._result['cap_pro'].loc[year, building, cs] == 0:
            data_hp_react['availability'][(building, cs)].iloc[1:] = 0
        else:
            data_hp_react['availability'][(building, cs)].iloc[1:] = \
                (prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'bev_14a_steuve_regulate',
                 'electricity_bev'] <= 0) * 1 + \
                (prob_grid_plan._result['e_pro_out'].loc[:, :, building, 'bev_14a_steuve_regulate',
                 'electricity_bev'] > 0) * \
                (data_grid_plan['demand'][(building, 'electricity_bev')].iloc[1:] -
                 prob_grid_plan._result[
                     'e_pro_out']. \
                 loc[:, :, building, 'bev_14a_steuve_regulate', 'electricity_bev'].values) / \
                prob_cluster._result['cap_pro'].loc[year, building, cs] / len(
                    bui_charging_stations)
    return data_hp_react

################################## END LVDS ##################################
def bev_uncontrolled_charging(data):
    """Inflexible case: each vehicle charges at full power as soon as it is plugged in, and this profile is added
    to the electricity demand. The mobility demand, mobility storage and charging station are then set to zero."""
    dem = data['demand']; eff = data['eff_factor']
    sto = data['storage']; pro = data['process']
    mob_cols = [c for c in dem.columns if str(c[1]).startswith('mobility')]
    n_conv = 0
    for (site, mname) in mob_cols:
        n = mname.replace('mobility', '')
        cs = f'charging_station{n}'
        # params
        try:
            cap = float(sto.loc[(slice(None), site, f'mobility_storage{n}', slice(None)), 'inst-cap-c'].iloc[0])
            pmax = float(pro.loc[(slice(None), site, cs), 'inst-cap'].iloc[0])
        except Exception:
            continue
        d = dem[(site, mname)].to_numpy(dtype=float)
        a = eff[(site, cs)].to_numpy(dtype=float) if (site, cs) in eff.columns else np.ones_like(d)
        T = len(d); ch = np.zeros(T); soc = cap
        for t in range(T):
            soc = max(0.0, soc - d[t])          # drive consumption first
            if a[t] > 0 and soc < cap:
                c = min(pmax, cap - soc)        # charge asap at full power
                ch[t] = c; soc += c
        dem[(site, 'electricity')] = dem[(site, 'electricity')].to_numpy(dtype=float) + ch
        dem[(site, mname)] = 0.0                # disable mobility demand
        sto.loc[(slice(None), site, f'mobility_storage{n}', slice(None)),
                ['inst-cap-c','cap-up-c','inst-cap-p','cap-up-p']] = 0
        pro.loc[(slice(None), site, cs), ['inst-cap','cap-up']] = 0
        n_conv += 1
    print(f"INFLEX-BEV: {n_conv} vehicles converted to uncontrolled charging", flush=True)
    return data


def transformer_rating(input_path):
    """Rated power of the transformer in kVA, from the pandapower grid stored in the input file (raw_data/net)."""
    import io
    import json
    import h5py
    with h5py.File(input_path, 'r') as f:
        raw = f['raw_data/net'][()]
    if isinstance(raw, np.ndarray):
        raw = raw.item() if raw.shape == () else raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode()
    trafo = json.loads(raw)['_object']['trafo']
    df = pd.read_json(io.StringIO(trafo['_object']), orient=trafo.get('orient', 'split'))
    return float(df['sn_mva'].sum() * 1000)


def capacities_from_result(ref_h5):
    """Heat pump, auxiliary heater and rooftop PV capacities from the result of the same run without control.
    Returns {site: (heat pump kW, auxiliary heater kW)} and {(site, PV process): kW}."""
    cap = pd.read_hdf(ref_h5, '/urbs_out/MILP/cap_pro')
    cap_table, pv_table = {}, {}
    for (stf, sit, pro), val in cap.items():
        hp, bo = cap_table.get(sit, (0.0, 0.0))
        if pro == 'heatpump_air':
            cap_table[sit] = (float(val), bo)
        elif pro == 'heatpump_booster':
            cap_table[sit] = (hp, float(val))
        elif str(pro).startswith('Rooftop'):
            pv_table[(sit, pro)] = float(val)
    return cap_table, pv_table


# diagnostics of the last call of inflex_14a / flex_14a (the ratios of flex_14a are written with --p14a-lam-out)
P14A_LAST_DIAG = {}


def _heat_to_el(h, cop, cap_hp):
    """Electricity a heat pump system draws to deliver heat h.

    The pump covers what it can at its coefficient of performance; anything
    beyond that falls to the resistive auxiliary heater, which has no eff_factor
    in the model and therefore runs at a coefficient of one. Ziffer 2.4.1.b
    counts that heater as part of the same system, so the connection power spans
    both and the conversion has to span both as well."""
    hp_heat = np.minimum(h, cap_hp * cop)
    return hp_heat / cop + np.maximum(h - hp_heat, 0.0)


def _el_to_heat(el, cop, cap_hp):
    """Heat the same system delivers when it draws el."""
    hp_el = np.minimum(el, cap_hp)
    return hp_el * cop + np.maximum(el - hp_el, 0.0)


def inflex_14a(data, rated_kva, cap_table=None, pv_table=None,
               p14a_kw=4.2, threshold=1.0, staged=True, pf_elc=0.959, pf_hp=0.959):
    """Section 14a EnWG control on the inflexible case, applied to the demand before the optimisation.

    Congested hours are those where the uncontrolled aggregate load exceeds the
    transformer rating. In those hours every controllable device is held at the
    minimum power the regulation guarantees it (BK6-22-300 Ziffer 4.5.1): 4.2 kW
    for a charge point, and for a heat pump either 4.2 kW or, above 11 kW of
    connection power, 0.4 times that power. Deferred charging carries over in the
    vehicle state of charge and deferred heat carries over as a backlog.

    Ziffer 4.3 additionally requires the return to normal operation to be staged
    so that it does not create a fresh overload. With staged=True the backlog is
    therefore worked off only within the headroom left below the rating once every
    device has been guaranteed what it would have drawn without any control at
    all, and that headroom is shared across devices in proportion to their
    backlog, which is the non-discriminatory allocation the same provision calls
    for. Because the headroom is measured against the uncontrolled profile it is
    non-negative in any hour that was not congested, so no device ends up worse
    off than without control and the aggregate stays within the rating.

    With staged=False each device works off its backlog at its own rate without
    regard to the aggregate. That reproduces the earlier behaviour and is kept
    for comparison; it can drive the peak above the uncontrolled one.

    cap_table and pv_table come from capacities_from_result (the same inflexible run without control). Without
    cap_table the connection power is estimated from the peak heat demand; without pv_table PV is not netted off.

    Assumes demand/process units of kW; call AFTER remove_battery and
    remove_heat_storage."""
    dem = data['demand']; eff = data['eff_factor']
    sto = data['storage']; pro = data['process']

    sites = sorted(set(c[0] for c in dem.columns if c[1] == 'electricity'))
    T = len(dem)
    base = np.zeros(T)
    for st in sites:
        base += dem[(st, 'electricity')].to_numpy(dtype=float)

    # ---- BEV pass 1: uncontrolled profiles per vehicle ----
    mob_cols = [c for c in dem.columns if str(c[1]).startswith('mobility')]
    veh = []
    ch1_total = np.zeros(T)
    for (site, mname) in mob_cols:
        n = mname.replace('mobility', '')
        cs = f'charging_station{n}'
        try:
            cap = float(sto.loc[(slice(None), site, f'mobility_storage{n}', slice(None)), 'inst-cap-c'].iloc[0])
            pmax = float(pro.loc[(slice(None), site, cs), 'inst-cap'].iloc[0])
        except Exception:
            continue
        d = dem[(site, mname)].to_numpy(dtype=float)
        a = eff[(site, cs)].to_numpy(dtype=float) if (site, cs) in eff.columns else np.ones(T)
        ch = np.zeros(T); soc = cap
        for t in range(T):
            soc = max(0.0, soc - d[t])
            if a[t] > 0 and soc < cap:
                c = min(pmax, cap - soc); ch[t] = c; soc += c
        ch1_total += ch
        veh.append(dict(site=site, mname=mname, cs=cs, n=n, cap=cap, pmax=pmax,
                        d=d, a=a, ch1=ch, ch2=np.zeros(T), soc=cap))

    # ---- HP electricity estimate: heat demand / COP where HP allowed ----
    # devices per site = number of primary heat pumps (heatpump_air) at the site.
    # 14a dimming is per controllable device; the urbs model represents each
    # building/bus with one heatpump_air process (heatpump_booster is an electric
    # backup heater, not separately dimmed). Read from the model, not assumed.
    dw = {}
    for st in sites:
        try:
            n = int((pro.loc[(slice(None), st, slice(None)), :].index
                     .get_level_values(2) == 'heatpump_air').sum())
        except Exception:
            n = 1
        dw[st] = max(1, n)
    hp_est = np.zeros(T); hp_sites = []
    # heat pumps that do not qualify as controllable devices keep their demand
    # unchanged, so their draw has to be counted when the headroom is measured
    hp_fixed = np.zeros(T)
    for st in sites:
        try:
            capup = float(pro.loc[(slice(None), st, 'heatpump_air'), 'cap-up'].iloc[0])
        except Exception:
            capup = 0.0
        if capup <= 0: continue
        cop = eff[(st, 'heatpump_air')].to_numpy(dtype=float) if (st, 'heatpump_air') in eff.columns else np.full(T, 3.0)
        cop = np.where(cop > 0.1, cop, 3.0)
        hcols = [c for c in dem.columns if c[0] == st and ('heat' in str(c[1]).lower() or 'water' in str(c[1]).lower())]
        h = np.zeros(T)
        for hc in hcols:
            h += dem[hc].to_numpy(dtype=float)
        # BNetzA BK6-22-300 Ziffer 4.5.1: minimum power is 4.2 kW per device,
        # except heat pumps whose grid-connection power exceeds 11 kW, where it
        # is 0.4 x connection power.
        # Ziffer 4.5.1 measures the minimum power against the grid connection
        # power, which is the rating of the installed equipment. That is what the
        # uncontrolled run actually builds, so it is read from cap_table rather
        # than inferred from demand: the heat pump plus its auxiliary heater,
        # which Ziffer 2.4.1.b counts as one system and 2.4.2 aggregates behind
        # one connection. Sizing from peak demand instead assumes the pump alone
        # covers the design hour with no store and no booster, which overstates
        # the connection power by roughly half.
        if cap_table is not None:
            cap_hp, cap_bo = cap_table.get(st, (0.0, 0.0))
            p_conn = float(cap_hp) + float(cap_bo)
        else:
            # without capacities: the electrical peak the heat pump must cover, max_t(Q_heat(t) / COP(t))
            p_conn = float(np.max(h / cop)) if h.max() > 0 else 0.0
            cap_hp = p_conn
        # with no pump capacity on record the split cannot be made, so the whole
        # demand is charged at the pump's coefficient as before
        cap_hp_eff = float(cap_hp) if cap_hp > 1e-9 else float('inf')
        el = _heat_to_el(h, cop, cap_hp_eff)
        hp_est += el
        # BK6-22-300 Ziffer 2.4.1: only devices with a grid-connection power
        # ABOVE 4.2 kW qualify as steuerbare Verbrauchseinrichtungen. Smaller
        # heat pumps fall outside the regime and must not be curtailed.
        if p_conn <= p14a_kw or not hcols:
            hp_fixed += el
            continue
        # Ziffer 2.4.2 aggregates the heat pumps behind one grid connection into
        # a single controllable device, and p_conn above is already that combined
        # figure, so the minimum power is one floor for the site, not one per unit
        p_min = 0.4 * p_conn if p_conn > 11.0 else p14a_kw
        # The setpoint is measured against the equipment the household has, so the
        # run must keep that equipment: curtailment thins the heat profile, and a
        # free capacity would let the optimisation rebuild a smaller pump than the
        # one the setpoint was written for.
        if cap_table is not None and cap_hp > 1e-9:
            for _c in ('inst-cap', 'cap-up'):
                if pro[_c].dtype.kind in 'iu':
                    pro[_c] = pro[_c].astype(float)
            pro.loc[(slice(None), st, 'heatpump_air'), ['inst-cap', 'cap-up']] = cap_hp
            try:
                pro.loc[(slice(None), st, 'heatpump_booster'), ['inst-cap', 'cap-up']] = cap_bo
            except Exception:
                pass          # the site may have no booster row at all
        hp_sites.append(dict(st=st, cop=cop, p_min=p_min, cap_hp=cap_hp_eff,
                             p_conn=p_conn, hcols=hcols, h=h, el=el, carry=0.0,
                             hmax=float(h.max()), h_new=np.zeros(T)))

    # ---- congestion mask, from the uncontrolled profile ----
    # The rating is an apparent power, so active power is converted with the same
    # power factor the load flow applies, 0.959 for every electricity withdrawal
    # including heat pumps and charging.
    # Ziffer 4.2 has the operator act on a state determination, and what it
    # measures at the transformer is the net withdrawal, after rooftop generation
    # has covered what it can. Photovoltaic output is therefore netted off here
    # as it already is on the flexible baseline, where the trigger reads the
    # import process. Capacities come from the uncontrolled run, since they are
    # decided by the optimisation and are not known from the demand data alone.
    pv_out = np.zeros(T)
    if pv_table is not None and 'supim' in data:
        sup = data['supim']
        # the supim column is keyed by the commodity a process consumes, and a
        # rooftop unit named "Rooftop PV_22_98" draws "solar_22_98", so the link
        # is read from process_commodity rather than assumed from the name
        pc = data['process_commodity']
        feed = {}
        for key in pc.index:
            prc_name, com_name, direction = key[-3], key[-2], key[-1]
            if direction == 'In' and str(prc_name).startswith('Rooftop'):
                feed[str(prc_name)] = com_name
        for (st, prc), cap in pv_table.items():
            col = (st, feed.get(str(prc), prc))
            if col in sup.columns:
                pv_out += float(cap) * sup[col].to_numpy(dtype=float)
        print(f"P14A: rooftop generation netted off, peak {pv_out.max():.1f} kW", flush=True)
    total = (base + ch1_total - pv_out) / pf_elc + hp_est / pf_hp
    total = np.maximum(total, 0.0)
    congested = total > (rated_kva * threshold)
    n_cong = int(congested.sum())
    limit = rated_kva * threshold

    # ---- joint time loop over both device classes ----
    n_blocked = 0
    n_insufficient = 0
    lam = np.zeros(T)
    for t in range(T):
        # 1) what each device is guaranteed this hour
        for v in veh:
            v['soc'] = max(0.0, v['soc'] - v['d'][t])
            want = min(v['pmax'], v['cap'] - v['soc']) if (v['a'][t] > 0 and v['soc'] < v['cap']) else 0.0
            v['want'] = want

        if congested[t]:
            # Ziffer 4.2 asks that the reduction be objectively necessary and
            # Ziffer 4.3 that all participating devices be drawn upon without
            # discrimination, every device counting as equally relieving. One
            # common ratio, set to the smallest value that clears the overload,
            # satisfies both. Because the ratio never exceeds one, the minimum
            # power of Ziffer 4.5.1 is preserved.
            # the overload is an apparent power, so the curtailable room is
            # converted with the same power factors before the ratio is formed
            # Ziffer 4.4.a: under direct control the operator receives a setpoint
            # and reduces the device to that value, so the ratio applies to the
            # device's connection power, not to what it happens to draw, and a
            # device already below its setpoint is untouched. The reduction that
            # a given ratio actually delivers is therefore piecewise linear in
            # the ratio: device i starts contributing once the setpoint falls to
            # its current draw, at ratio (cap - draw) / (cap - p_min), and from
            # there contributes (cap - p_min) per unit of ratio. Sorting those
            # breakpoints gives the smallest ratio that clears the overload,
            # which is what Ziffer 4.3 sentence 1 asks for.
            over = total[t] - limit
            seg = []                      # (breakpoint ratio, slope in kVA)
            for v in veh:
                cap_i, draw = v['pmax'], min(v['want'], v['ch1'][t])
                span = cap_i - p14a_kw
                if span <= 1e-9 or draw <= p14a_kw:
                    continue
                seg.append(((cap_i - draw) / span, span / pf_elc))
            for d_ in hp_sites:
                cap_i = d_['p_conn']; pmin = d_['p_min']
                draw = d_['el'][t]
                span = cap_i - pmin
                if span <= 1e-9 or draw <= pmin:
                    continue
                seg.append(((cap_i - draw) / span, span / pf_hp))
            seg.sort(key=lambda x: x[0])

            lam[t] = 1.0 if seg else 0.0
            slope_sum = offset = 0.0
            cleared = False
            for k, (bp, sl) in enumerate(seg):
                nxt = seg[k + 1][0] if k + 1 < len(seg) else 1.0
                slope_sum += sl
                offset += sl * bp
                if slope_sum * nxt - offset >= over - 1e-9:
                    lam[t] = min(1.0, max(bp, (over + offset) / slope_sum))
                    cleared = True
                    break
            if not cleared:
                n_insufficient += 1     # even at ratio one the overload remains

            for v in veh:
                sp = v['pmax'] - lam[t] * (v['pmax'] - p14a_kw)
                v['ch2'][t] = min(min(v['want'], v['ch1'][t]), max(p14a_kw, sp))
            for d_ in hp_sites:
                pmin = d_['p_min']
                sp = max(pmin, d_['p_conn'] - lam[t] * (d_['p_conn'] - pmin))
                # a setpoint above the current demand leaves the device untouched,
                # but it must not be used to work off the backlog while the control
                # event is running: Ziffer 4.3 sentence 3 puts recovery after it
                d_['h_new'][t] = min(d_['h'][t],
                                     _el_to_heat(sp, d_['cop'][t], d_['cap_hp']))
        else:
            for v in veh:
                # never more than the uncontrolled draw before the backlog is shared out
                v['ch2'][t] = min(v['want'], v['ch1'][t]) if staged else v['want']
            for d_ in hp_sites:
                d_['h_new'][t] = d_['h'][t]

        # 2) share out the remaining headroom among devices still behind
        if not congested[t]:
            if staged:
                # headroom is measured against an apparent-power rating, so the
                # guaranteed draw and the outstanding backlog are both converted
                # with the same power factors before the share is worked out
                guaranteed = ((base[t] + sum(v['ch2'][t] for v in veh)) / pf_elc
                              + (hp_fixed[t]
                                 + sum(_heat_to_el(d_['h_new'][t], d_['cop'][t], d_['cap_hp'])
                                       for d_ in hp_sites)) / pf_hp)
                head = limit - guaranteed
                need_v = [max(0.0, v['want'] - v['ch2'][t]) / pf_elc for v in veh]
                need_h = [(_heat_to_el(d_['h_new'][t] + d_['carry'], d_['cop'][t], d_['cap_hp'])
                           - _heat_to_el(d_['h_new'][t], d_['cop'][t], d_['cap_hp'])) / pf_hp
                          for d_ in hp_sites]
                tot = sum(need_v) + sum(need_h)
                if tot > 1e-9:
                    if head <= 1e-9:
                        n_blocked += 1
                    else:
                        f = min(1.0, head / tot)
                        for v, nd in zip(veh, need_v):
                            v['ch2'][t] += f * nd * pf_elc      # back to active power
                        for d_, nd in zip(hp_sites, need_h):
                            grant = f * nd * pf_hp              # back to active power
                            el_now = _heat_to_el(d_['h_new'][t], d_['cop'][t], d_['cap_hp'])
                            phys = max(0.0, d_['p_conn'] - el_now)
                            # The connection power was sized on the design hour, when
                            # the coefficient of performance is at its lowest. In a
                            # milder hour the same power yields more heat, which would
                            # let the recovery push thermal output past anything the
                            # installation was built to deliver, so it is also held to
                            # the largest heat demand the site ever has.
                            room = max(0.0, d_['hmax'] - d_['h_new'][t])
                            reachable = _el_to_heat(el_now + min(grant, phys),
                                                    d_['cop'][t], d_['cap_hp']) - d_['h_new'][t]
                            d_['h_new'][t] += min(d_['carry'], max(0.0, reachable), room)
            else:
                for d_ in hp_sites:
                    d_['h_new'][t] += min(d_['carry'], d_['p_min'] * d_['cop'][t])

        # 3) advance the state
        for v in veh:
            v['soc'] += v['ch2'][t]
        for d_ in hp_sites:
            d_['carry'] = max(0.0, d_['h'][t] + d_['carry'] - d_['h_new'][t])

    # ---- write the dimmed BEV profiles into electricity demand ----
    n_conv = 0
    for v in veh:
        dem[(v['site'], 'electricity')] = dem[(v['site'], 'electricity')].to_numpy(dtype=float) + v['ch2']
        dem[(v['site'], v['mname'])] = 0.0
        sto.loc[(slice(None), v['site'], f"mobility_storage{v['n']}", slice(None)),
                ['inst-cap-c','cap-up-c','inst-cap-p','cap-up-p']] = 0
        pro.loc[(slice(None), v['site'], v['cs']), ['inst-cap','cap-up']] = 0
        n_conv += 1

    # ---- write the dimmed heat profiles ----
    # The device is controlled as a whole, so the curtailment is decided on the
    # summed heat demand, but the served amount has to be split back over the
    # individual services in proportion to what each of them is actually owed,
    # its own demand this hour plus its own backlog. Scaling every column by the
    # same ratio instead moves energy between services, because congestion falls
    # in winter hours where space heat dominates while recovery falls in hours
    # where water heat does.
    shifted_kwh = 0.0
    for d_ in hp_sites:
        h, h_new = d_['h'], d_['h_new']
        shifted_kwh += float(np.maximum(h - h_new, 0).sum())
        cols = {hc: dem[hc].to_numpy(dtype=float).copy() for hc in d_['hcols']}
        owed = {hc: 0.0 for hc in d_['hcols']}
        out = {hc: np.zeros(T) for hc in d_['hcols']}
        for t in range(T):
            want = {hc: cols[hc][t] + owed[hc] for hc in d_['hcols']}
            tot = sum(want.values())
            if tot > 1e-9:
                f = min(1.0, h_new[t] / tot)
                for hc in d_['hcols']:
                    out[hc][t] = want[hc] * f
                    owed[hc] = want[hc] - out[hc][t]
        for hc in d_['hcols']:
            dem[hc] = out[hc]
        if d_['carry'] > 1e-6:
            print(f"P14A WARNING: site {d_['st']} unserved heat at year end: {d_['carry']:.1f} kWh", flush=True)

    n_large = sum(1 for d_ in hp_sites if d_['p_min'] > p14a_kw)
    bev_deferred = float(sum(np.maximum(v['ch1'] - v['ch2'], 0).sum() for v in veh))
    # exposed so the control adequacy limit can be swept without solving.
    # kept in a module attribute rather than in data, because everything in data
    # is written to HDF5 on save and a dict cannot be stored there
    global P14A_LAST_DIAG
    P14A_LAST_DIAG = dict(
        n_cong=n_cong, n_insufficient=n_insufficient, n_blocked=n_blocked,
        lam_mean=float(lam[congested].mean()) if n_cong else 0.0,
        n_hp=len(hp_sites), n_bev=len(veh), n_large=n_large,
        peak_uncontrolled=float(total.max()), limit=float(limit),
        shifted_kwh=float(shifted_kwh), bev_deferred=bev_deferred)
    print(f"P14A-14a4.5.1: {n_large}/{len(hp_sites)} HP sites above 11 kW "
          f"(floor 0.4 x P_conn instead of {p14a_kw} kW)", flush=True)
    print(f"P14A-14a4.3: staged release {'ON' if staged else 'OFF'} | "
          f"hours with backlog but no headroom: {n_blocked}", flush=True)
    print(f"P14A-14a4.2: mean reduction ratio in congested hours "
          f"{(lam[congested].mean() if n_cong else 0):.3f} | "
          f"hours where full curtailment still leaves an overload: {n_insufficient}",
          flush=True)
    print(f"P14A-DIAG: base_max {base.max():.0f} | ch1_max {ch1_total.max():.0f} | hp_est_max {hp_est.max():.0f} | hp_sites {len(hp_sites)} | total_max {total.max():.0f}", flush=True)
    print(f"P14A: {n_conv} BEV dimmed-uncontrolled | congested {n_cong}/{T} h "
          f"(rated {rated_kva:.0f} kVA x {threshold}) | HP heat shifted {shifted_kwh:.0f} kWh "
          f"| BEV energy deferred {bev_deferred:.0f} kWh", flush=True)
    return data


DEFICIT_PRICE = 1000.0   # EUR/kWh, far above any real heat source


def flex_14a(data, rated_kva, ref_h5, prev_lam=None,
             p14a_kw=4.2, threshold=1.0, pf_elc=0.959):
    """Section 14a EnWG control on the flexible case, through the availability feature.

    On the inflexible baseline the controlled load is known before the run, so
    the curtailment can be written into the demand series. On the flexible
    baseline it is an outcome of the optimisation, so a first pass is needed:
    ref_h5 is the result file of the same grid solved without control, giving
    the transformer load, the capacities actually built, and what each device
    drew hour by hour.

    Ziffer 4.4.a describes direct control as a setpoint on the maximum grid
    effective power of the device, which is what the availability constraint
    expresses, tau <= cap * availability. One common ratio is applied to every
    participating device, measured against its own curtailable range, and set to
    the smallest value that clears the overload:

        setpoint_i(t)    = cap_i - lambda(t) * (cap_i - p_min_i)
        availability_i(t) = setpoint_i(t) / cap_i

    Minimum power follows Ziffer 4.5.1: 4.2 kW per device, or 0.4 times the
    connection power for a heat pump above 11 kW. Devices at or below 4.2 kW are
    not controllable devices under Ziffer 2.4.1 and are left alone. Electricity
    storage is included while charging, per Ziffer 2.4.1.d; heat storage is not,
    since it does not draw from the grid.

    Because the second pass may shift load into hours the first pass left clear,
    the caller can pass the ratios from the previous pass in prev_lam; the new
    ratios are taken as the element-wise maximum, so the sequence is monotone and
    terminates. That mirrors Ziffer 4.2, where the operator acts on a state
    determination and adjusts. prev_lam is read with --p14a-lam-in, the new ratios are written with --p14a-lam-out.

    Heat-pump capacity is pinned to its first-pass value. The constraint limits
    throughput to a share of capacity, so leaving capacity free would let the
    optimisation enlarge the unit to undo the limit, and pinning it also matches
    the assumption made on the inflexible baseline, that the household sized its
    heat pump without anticipating control."""

    pro, sto = data['process'], data['storage']
    T = len(data['demand'])
    stf = int(pro.index.get_level_values(0)[0])
    limit = rated_kva * threshold
    ts = np.arange(T)

    with pd.HDFStore(ref_h5, 'r') as store:
        cap_pro = store['/urbs_out/MILP/cap_pro']
        e_in = store['/urbs_out/MILP/e_pro_in'].reset_index()
        cap_sto = store['/urbs_out/MILP/cap_sto_p'] if '/urbs_out/MILP/cap_sto_p' in store else None
        e_sto = store['/urbs_out/MILP/e_sto_in'].reset_index() if '/urbs_out/MILP/e_sto_in' in store else None
    e_in.columns = ['t', 'stf', 'sit', 'pro', 'com', 'val']

    def series(df, mask):
        return df[mask].groupby('t').val.sum().reindex(ts, fill_value=0.0).to_numpy(dtype=float)

    load = series(e_in, e_in.pro == 'import') / pf_elc
    congested = load > limit
    n_cong = int(congested.sum())

    # m.sit is built from the commodity table, so a site absent from it never
    # enters the model and a setpoint written for it would be rejected
    model_sites = set(data['commodity'].index.get_level_values(1))

    dev = []                      # controllable devices with their first-pass draw
    deficit_sites = set()
    mob_sites = set()
    for (y, sit, prc) in pro.index:
        if sit not in model_sites:
            continue
        try:
            cap = float(cap_pro.loc[(slice(None), sit, prc)].iloc[0])
        except Exception:
            continue
        if prc == 'heatpump_booster':
            continue                      # carried with the heat pump below
        if prc == 'heatpump_air':
            # Ziffer 2.4.1.b counts an auxiliary or emergency heater as part of
            # the heat pump system and Ziffer 2.4.2 aggregates what sits behind
            # one connection, so the pair carries a single setpoint. Applying the
            # same share to both keeps their sum at that setpoint. Without it the
            # optimisation simply moves heat onto the resistive booster, which
            # draws several times the power for the same heat.
            units = [u for u in ('heatpump_air', 'heatpump_booster')
                     if (y, sit, u) in pro.index]
            caps = {}
            for u in units:
                try:
                    caps[u] = float(cap_pro.loc[(slice(None), sit, u)].iloc[0])
                except Exception:
                    caps[u] = 0.0
            cap = sum(caps.values())
            if cap <= p14a_kw:
                continue
            p_min = 0.4 * cap if cap > 11.0 else p14a_kw
            for _c in ('inst-cap', 'cap-up'):
                if pro[_c].dtype.kind in 'iu':
                    pro[_c] = pro[_c].astype(float)
            # Both units are pinned. The availability limit is a share of cap_pro,
            # so a device left free simply grows and takes its allowance with it:
            # on one grid the booster was rebuilt twelve times larger to escape.
            # The setpoint has to rest on the capacity the household actually has.
            for u in units:
                pro.loc[(y, sit, u), ['inst-cap', 'cap-up']] = caps[u]
            deficit_sites.add(sit)
            draw = series(e_in, (e_in.sit == sit) & (e_in.pro.isin(units))
                          & (e_in.com == 'electricity'))
            dev.append(dict(cols=[(sit, u) for u in units], kind='pro',
                            cap=cap, p_min=p_min, draw=draw))
            continue
        if str(prc).startswith('charging_station'):
            if cap <= p14a_kw:
                continue
            draw = series(e_in, (e_in.sit == sit) & (e_in.pro == prc) & (e_in.com == 'electricity'))
            dev.append(dict(cols=[(sit, prc)], kind='pro', cap=cap,
                            p_min=p14a_kw, draw=draw))
            mob_sites.add((sit, str(prc)[len('charging_station'):]))
        continue

    if cap_sto is not None and e_sto is not None:
        e_sto.columns = ['t', 'stf', 'sit', 'sto', 'com', 'val']
        for (y, sit, stg, com) in cap_sto.index:
            if stg != 'battery_private' or com != 'electricity':
                continue
            if sit not in model_sites:
                continue
            cap = float(cap_sto.loc[(y, sit, stg, com)])
            if cap <= p14a_kw:            # Ziffer 2.4.1.d threshold
                continue
            draw = series(e_sto, (e_sto.sit == sit) & (e_sto.sto == stg) & (e_sto.com == com))
            # availability is a share of the charging power, so a storage left free
            # would simply be built larger and take its allowance with it
            sto_tbl = data['storage']
            for _c in ('inst-cap-p', 'cap-up-p'):
                if _c in sto_tbl.columns and sto_tbl[_c].dtype.kind in 'iu':
                    sto_tbl[_c] = sto_tbl[_c].astype(float)
            try:
                sto_tbl.loc[(slice(None), sit, stg, com), ['inst-cap-p', 'cap-up-p']] = cap
            except Exception:
                pass
            dev.append(dict(cols=[(sit, stg)], kind='sto', cap=cap,
                            p_min=p14a_kw, draw=draw))

    lam = np.zeros(T)
    n_insufficient = 0
    for t in range(T):
        if not congested[t]:
            continue
        over = load[t] - limit
        seg = []
        for d_ in dev:
            span = d_['cap'] - d_['p_min']
            if span <= 1e-9 or d_['draw'][t] <= d_['p_min']:
                continue
            seg.append(((d_['cap'] - d_['draw'][t]) / span, span / pf_elc))
        seg.sort(key=lambda x: x[0])
        if not seg:
            n_insufficient += 1
            continue
        lam[t] = 1.0
        slope_sum = offset = 0.0
        cleared = False
        for k, (bp, sl) in enumerate(seg):
            nxt = seg[k + 1][0] if k + 1 < len(seg) else 1.0
            slope_sum += sl
            offset += sl * bp
            if slope_sum * nxt - offset >= over - 1e-9:
                lam[t] = min(1.0, max(bp, (over + offset) / slope_sum))
                cleared = True
                break
        if not cleared:
            n_insufficient += 1

    if prev_lam is not None:
        lam = np.maximum(lam, np.asarray(prev_lam, dtype=float))

    idx = (data['eff_factor'].index if not data['eff_factor'].empty
           else pd.MultiIndex.from_product([[stf], ts], names=['support_timeframe', 't']))
    # Ziffer 4.3 sentence 3 requires the return to be staged so that it does not
    # cause a fresh overload. Leaving the limit open outside congested hours lets
    # every deferred device come back at once, and on the inflexible baseline that
    # was enough to make control worse than doing nothing. Outside congested hours
    # each device is therefore allowed its uncontrolled draw plus a share of what
    # is still free below the rating, divided in proportion to the room each one
    # has left. The uncontrolled draw is always granted: withholding it would be
    # curtailment, not recovery.
    spare = np.maximum(0.0, (limit - load) * pf_elc)      # active power headroom
    room = np.zeros(T)
    for d_ in dev:
        room += np.maximum(0.0, d_['cap'] - d_['draw'])
    frac = np.divide(spare, room, out=np.zeros(T), where=room > 1e-9)
    frac = np.minimum(frac, 1.0)

    ava_pro, ava_sto = {}, {}
    for d_ in dev:
        setpoint = d_['cap'] - lam * (d_['cap'] - d_['p_min'])
        recovery = d_['draw'] + frac * np.maximum(0.0, d_['cap'] - d_['draw'])
        allowed = np.where(congested, setpoint, np.minimum(recovery, d_['cap']))
        share = np.clip(allowed / d_['cap'], 0.0, 1.0)
        tbl = ava_sto if d_['kind'] == 'sto' else ava_pro
        for col in d_['cols']:
            tbl[col] = share
    for name, tbl in (('availability', ava_pro), ('availability_sto', ava_sto)):
        if tbl:
            df = pd.DataFrame(tbl, index=idx)
            df.columns = pd.MultiIndex.from_tuples(df.columns)
            data[name] = df

    # With both units held at their reference capacity a site can be short of
    # heat in a cold hour, which the inflexible path reports as unserved heat but
    # which the flexible model can only express as infeasibility. A last-resort
    # source is therefore added, built like import: a Buy commodity feeding a
    # process into common_heat. Its price is set far above any real alternative,
    # so it is used only when nothing else can serve the demand; the quantity is
    # the unserved heat and the cost is excluded from reported figures.
    if deficit_sites:
        com, bsp = data['commodity'], data['buy_sell_price']
        prc_tbl, pc = data['process'], data['process_commodity']
        for st in sorted(deficit_sites):
            com.loc[(stf, st, 'heat_deficit', 'Buy'), 'price'] = 1.0
            row = prc_tbl.loc[(stf, st, 'import')].copy()
            row['inst-cap'] = 2000.0
            row['cap-up'] = 2000.0
            for c in ('inv-cost-fix', 'inv-cost', 'fix-cost', 'var-cost'):
                if c in row.index:
                    row[c] = 0.0
            prc_tbl.loc[(stf, st, 'heat_backup')] = row
        bsp['heat_deficit'] = DEFICIT_PRICE
        pc.loc[(stf, 'heat_backup', 'heat_deficit', 'In'), 'ratio'] = 1.0
        pc.loc[(stf, 'heat_backup', 'common_heat', 'Out'), 'ratio'] = 1.0
        data['commodity'] = com.sort_index()
        data['process'] = prc_tbl.sort_index()
        data['process_commodity'] = pc.sort_index()
        print(f"P14A-AVA: last-resort heat source added at {len(deficit_sites)} sites "
              f"(price {DEFICIT_PRICE:.0f}, quantity reported as unserved heat)", flush=True)

    # Driving demand is a Demand commodity and must be met, so holding a charge
    # point at its minimum power can make the model infeasible outright. The
    # inflexible path never hits this because charging there is written into the
    # demand series and shortfall simply shows up as deferred energy. A
    # last-resort charger is added for the same reason as the heat one, priced
    # far above any alternative, and its throughput is the unserved mobility.
    if mob_sites:
        com, bsp = data['commodity'], data['buy_sell_price']
        prc_tbl, pc = data['process'], data['process_commodity']
        for st, num in sorted(mob_sites):
            name = f'mobility_backup_{num}'
            com.loc[(stf, st, 'mobility_deficit', 'Buy'), 'price'] = 1.0
            row = prc_tbl.loc[(stf, st, 'import')].copy()
            row['inst-cap'] = 2000.0
            row['cap-up'] = 2000.0
            for c in ('inv-cost-fix', 'inv-cost', 'fix-cost', 'var-cost'):
                if c in row.index:
                    row[c] = 0.0
            prc_tbl.loc[(stf, st, name)] = row
            pc.loc[(stf, name, 'mobility_deficit', 'In'), 'ratio'] = 1.0
            pc.loc[(stf, name, f'mobility{num}', 'Out'), 'ratio'] = 1.0
        bsp['mobility_deficit'] = DEFICIT_PRICE
        data['commodity'] = com.sort_index()
        data['process'] = prc_tbl.sort_index()
        data['process_commodity'] = pc.sort_index()
        print(f"P14A-AVA: last-resort charger added for {len(mob_sites)} charge points",
              flush=True)

    global P14A_LAST_DIAG
    P14A_LAST_DIAG = dict(n_cong=n_cong, n_insufficient=n_insufficient,
                          n_deficit_sites=len(deficit_sites),
                          n_mob_backup=len(mob_sites),
                          lam_mean=float(lam[congested].mean()) if n_cong else 0.0,
                          n_hp=sum(1 for d_ in dev if d_['cols'][0][1] == 'heatpump_air'),
                          n_bev=sum(1 for d_ in dev if str(d_['cols'][0][1]).startswith('charging')),
                          n_sto=sum(1 for d_ in dev if d_['kind'] == 'sto'),
                          peak_uncontrolled=float(load.max()), limit=float(limit), lam=lam)
    print(f"P14A-AVA: {P14A_LAST_DIAG['n_hp']} heat pumps, "
          f"{P14A_LAST_DIAG['n_bev']} charge points, {P14A_LAST_DIAG['n_sto']} batteries | "
          f"congested {n_cong}/{T} h (rated {rated_kva:.0f} kVA x {threshold}) | "
          f"mean ratio {P14A_LAST_DIAG['lam_mean']:.3f} | "
          f"hours still over at ratio one: {n_insufficient}", flush=True)
    return data
