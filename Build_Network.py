import pandas as pd
import numpy as np
import pypsa
# 1. Specify your Excel file path
excel_file = "Data/Data_File_Year.xlsx"  # Replace with your actual file path

def Create_Var(excel_file):
    # 2. Read the entire Excel file (None loads all sheets as a dictionary)
    excel_dict = pd.read_excel(excel_file, sheet_name=None)

    # 3. Clean up sheet names and inject them into the global environment
    for sheet, dataframe in excel_dict.items():
        # Replace spaces and hyphens with underscores to make valid Python variable names
        var_name = sheet.replace(" ", "_").replace("-", "_").lower()
        # Dynamically create the variable
        globals()[var_name] = dataframe
        
        print(f"DataFrame: '{var_name}' with {dataframe.shape[0]} rows.")
    

def Network_Base(IEX,Cost_Scale):
    Create_Var(excel_file)
    n = pypsa.Network(name = "Eastern Interconnection")
    n.set_snapshots(snapshots.Snapshot)
    for index, row in buses.iterrows():
        n.add("Bus", row['name'], v_nom=row['v_nom'],x=row['x'],y=row['y'])

    for index, row in line_types.iterrows():
        n.add("LineType", row['name'],
            f_nom=row['f_nom'],
            r_per_length=row['r_per_length'],
            x_per_length=row['x_per_length'],
            c_per_length=row['c_per_length'],
            i_nom=row['i_nom'],
            mounting=row['mounting'],
            cross_section = row['cross_section'],
            references = row['description'])
        
    for index, row in lines.iterrows():
        n.add("Line", 
            row['name'], 
            bus0=row['bus0'], 
            bus1=row['bus1'],
            length = row['length'],
            type = row['type'],
            carrier =row['carrier'],
            num_parallel = row['num_parallel'],
            s_nom_extendable = True,)
    # Calculate s_nom for lines with types
    for line in n.lines.itertuples():
        if line.type:
            line_type = n.line_types.loc[line.type]
            v_nom = n.buses.loc[line.bus0, 'v_nom']
            num_parallel = line.num_parallel
            s_nom =num_parallel* np.sqrt(3) * line_type.i_nom * v_nom
            n.lines.at[line.Index, 's_nom'] = s_nom

    p_max_puv = p_max_pu.set_index('Snapshot') 
    for index, row in generators.iterrows():
    
        n.add("Generator", row['name'],
            bus=row['bus'],
            p_nom=row['p_nom'],
            #p = row['']
            marginal_cost=row['marginal_cost'],
            carrier = row['carrier'],
            lat=row['x'],
            lon =row['y'],
            #p_max_pu = p_max_puv.p_max_pu,
            p_set = p_max_puv['mean']*row['p_nom'],
            )


    for index, row in transformers.iterrows():
        n.add("Transformer", row['name'],
            bus0=row['bus0'],
            bus1=row['bus1'],
            x=row['x'],
            r=row['r'],
            s_nom_extendable = True,
            s_nom = row['s_nom'])
        
    print(loads_p_set)
    loads_p_setv = loads_p_set.set_index('Snapshot') 
    loads_p_setv = loads_p_setv.replace(0.0, np.nan)
    loads_p_setv  = loads_p_setv.fillna(loads_p_setv.mean()) # Average fill
    for index, row in loads.iterrows():
        n.add("Load", row['name'],
            bus=row['bus'],
            p_set=loads_p_setv[row['name']],
            carrier = 'Internal Load')
    n.add("Carrier", 'Internal Load')


    for index, row in carriers.iterrows():
        n.add("Carrier", row['name'])

    #pypsa.pf.apply_line_types(n)
    #pypsa.pf.apply_transformer_t_model(n)
    n.calculate_dependent_values()
    ### adding the Dudhkoshi Storage Project
    #n.add(
    #"StorageUnit",
    #"Dudhkoshi Storage",
    #bus="Dudhkoshi_400",
    #p_nom=700,  # Nominal power capacity in MW
    #max_hours=15*24,  # Maximum storage duration in hours
    #efficiency_store=0.9,  # Efficiency of storing energy
    #efficiency_dispatch=0.9,  # Efficiency of dispatching energy
    #inflow = 200,  # Inflow rate in MW
    #p_nom_extendable=False,  # Allow power capacity to be optimized
    #lifetime = 50,
    #discount_rate = 0.08,
    #overnight_cost = 2.2*100000000/(700*Cost_Scale),  # Cost per MW
    #lat = 27.134482087914275,
    #lon = 86.63541407814571
#)



    MC_India = IEX/Cost_Scale  # convert NPR/MWh to model units
    MC_Nepal = 0
    MC_High_Load_Shedding = 100000/Cost_Scale
    # ── Khimti interconnection ────────────────────────────────────
    if "Khimti Slack" not in n.generators.index:
        n.add("Generator", "Khimti Slack",
            bus          = "Khimti_400",
            p_nom        = 2000,
            p_max_pu     = 1,
            carrier = 'Khimti Interconnection',
            marginal_cost= MC_High_Load_Shedding)

    n.add("Carrier", 'Khimti Interconnection')

    # ── Dhalkabar interconnection ─────────────────────────────────
    if "Dhalkabar Slack" not in n.generators.index:
        n.add("Generator", "Dhalkabar Slack",
            bus          = "Dhalkabar_400",
            p_nom        = 1000,
            p_max_pu     = 1,
            carrier = 'Dhalkebar Interconnection',
            marginal_cost= MC_High_Load_Shedding)


    n.add("Carrier", 'Dhalkebar Interconnection')

    # ── India interconnection — Purnea ────────────────────────────
    if "India Slack Purnea" not in n.generators.index:
        n.add("Load", "India Demand Purnea",
            bus   = "Purnea_400",
            carrier = 'Purnea Export',
            p_set = 5000)
                          # ← adjust to realistic India demand [MW]
        n.add("Generator", "India Slack Purnea",
            bus          = "Purnea_400",
            p_nom        = 5000,
            marginal_cost= MC_India,
            carrier = "Purnea Grid",
            control = 'Slack')   # ← MC_India preserved, not overwritten
    n.add("Carrier", 'Purnea Export')
    n.add('Carrier', 'Purnea Grid')
    print('Network Created: ')
    print(n.component)

    return n


import numpy as np
import pandas as pd


def generate_ar1_scenarios(p_max_pu: pd.Series,
                            n_scenarios: int = 200,
                            seed: int = 42) -> np.ndarray:
    """
    Generate synthetic AR(1) p_max_pu scenarios for a single generator.

    Parameters
    ----------
    p_max_pu    : Series indexed by Snapshot (8760 hourly rows), values in [0, 1].
    n_scenarios : Number of synthetic annual scenarios.
    seed        : Random seed.

    Returns
    -------
    ndarray shape (n_scenarios, 8760)
    """
    rng = np.random.default_rng(seed)
    Q   = p_max_pu.values                      # extract numpy array from Series
    T   = len(Q)

    # Hourly climatology via ±12-h rolling window
    clim_mean = p_max_pu.rolling(25, center=True, min_periods=1).mean().values
    clim_std  = p_max_pu.rolling(25, center=True, min_periods=1).std(ddof=1).fillna(1e-4).values
    clim_std  = np.where(clim_std < 1e-6, 1e-6, clim_std)

    # Standardised residuals and lag-1 autocorrelation
    Z         = (Q - clim_mean) / clim_std
    rho       = float(np.corrcoef(Z[:-1], Z[1:])[0, 1])
    sigma_eps = np.sqrt(max(1 - rho**2, 0))

    # AR(1) simulation — vectorised over all scenarios at once
    arr       = np.zeros((n_scenarios, T))
    eps       = rng.normal(0, sigma_eps, (n_scenarios, T))
    arr[:, 0] = rng.choice(Z, n_scenarios)
    for t in range(1, T):
        arr[:, t] = rho * arr[:, t-1] + eps[:, t]

    return np.clip(clim_mean + arr * clim_std, 0.0, 1.0)


def get_scenario(n_scenarios):

    scenarios = generate_ar1_scenarios(p_max_pu.p_max_pu, n_scenarios)
    print(scenarios)
    return scenarios

 