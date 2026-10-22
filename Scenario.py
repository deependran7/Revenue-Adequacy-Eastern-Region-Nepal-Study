import pypsa
import numpy as np


def crf(lifetime, discount_rate):
    r = discount_rate
    n = lifetime
    return (r * (1 + r)**n) / ((1 + r)**n - 1)


def generator_capital_cost(overnight_cost_per_mw, lifetime=40, discount_rate=0.08):
    return overnight_cost_per_mw * crf(lifetime, discount_rate)


def Scenario_Analysis(network, Hydro_Cost_per_MW, line_transformer_cost,load_growth_factor,Cost_Scale):
    """
    line_transformer_cost format:
        {
            'Type':        {0: 'Cost'},
            132:           {0: 224010.178684714},   # Rs/km/MVA
            220:           {0: 130138.841583967},   # Rs/km/MVA
            400:           {0: 36267.5044832196},   # Rs/km/MVA
            'Transformer': {0: 8928571.42857143}    # Rs/MVA
        }
    """

    # ─────────────────────────────────────────────────────────────
    # 1. SNAPSHOT WEIGHTING
    # ─────────────────────────────────────────────────────────────
    snapshot_weight = 1
    network.snapshot_weightings["objective"]  = snapshot_weight
    network.snapshot_weightings["generators"] = snapshot_weight
    network.snapshot_weightings["stores"]     = snapshot_weight

    # ─────────────────────────────────────────────────────────────
    # 2. HYDRO GENERATOR COSTS AND LIMITS
    # ─────────────────────────────────────────────────────────────
    LIFETIME      = 40
    DISCOUNT_RATE = 0.085

    hydro_gens = network.generators[
        network.generators["carrier"] == "Hydro"
    ].index

    cap_cost = generator_capital_cost(
        overnight_cost_per_mw=Hydro_Cost_per_MW/Cost_Scale,
        lifetime=LIFETIME,
        discount_rate=DISCOUNT_RATE,
    )

    for gen in hydro_gens:
        cap_mw = network.generators.loc[gen, "p_nom"]
        network.generators.loc[gen, "capital_cost"]  = cap_cost
        network.generators.loc[gen, "marginal_cost"] = 0
        

    # ─────────────────────────────────────────────────────────────
    # 3. TRANSMISSION LINE COSTS AND EXPANSION LIMITS
    # ─────────────────────────────────────────────────────────────
    MAX_CIRCUITS   = 6
    LIFETIME      = 25
    DISCOUNT_RATE = 0.085
    annuity_factor = crf(LIFETIME, DISCOUNT_RATE)

    # Extract overnight costs per voltage level (Rs/km/MVA → Rs/km/MW)
    overnight_cost_per_km_mw = {
        132: line_transformer_cost[132][0]/Cost_Scale,
        220: line_transformer_cost[220][0]/Cost_Scale,
        400: line_transformer_cost[400][0]/Cost_Scale,
    }

    # Map voltage from bus0
    network.lines['v_nom'] = (
        network.lines['bus0']
        .map(network.buses['v_nom'])
        .round()
        .astype(int)
    )

    # Warn about unmatched voltages
    unmatched = ~network.lines['v_nom'].isin(overnight_cost_per_km_mw)
    if unmatched.any():
        print("WARNING — unmatched voltages:")
        print(network.lines.loc[unmatched, ['bus0', 'v_nom']])

    # Annualised capital cost (Rs/MW/yr) = overnight(Rs/km/MW) × length(km) × CRF
    network.lines['capital_cost'] = (
        network.lines['v_nom'].map(overnight_cost_per_km_mw)   # Rs/km/MW
        * network.lines['length']                               # → Rs/MW
        * annuity_factor                                        # → Rs/MW/yr
    )

    # Expansion bounds
    network.lines['s_nom_min']        = network.lines['s_nom']*0.0 #########################################check this
    network.lines['s_nom_max']        = network.lines['s_nom'] * MAX_CIRCUITS
    network.lines['s_nom_extendable'] = True

    # Clean up temporary column
    network.lines.drop(columns=['v_nom'], inplace=True)

    #print(network.lines[[
    #    'type', 'num_parallel', 's_nom', 's_nom_min',
    #    's_nom_max', 's_nom_extendable', 'capital_cost'
    #]].to_string())

    # ─────────────────────────────────────────────────────────────
    # 4. TRANSFORMER COSTS AND EXPANSION LIMITS
    # ─────────────────────────────────────────────────────────────

    # Overnight cost (Rs/MVA = Rs/MW), annualised with CRF
    overnight_transformer_per_mw = line_transformer_cost['Transformer'][0]  # Rs/MVA
    annual_transformer_cost      = overnight_transformer_per_mw * annuity_factor  # Rs/MW/yr

    network.transformers['s_nom_extendable'] = True
    network.transformers['s_nom_min']        = network.transformers['s_nom']*0.0####################################
    network.transformers['s_nom_max']        = 100 * network.transformers['s_nom']
    network.transformers['capital_cost']     = annual_transformer_cost/Cost_Scale

    #print(f"\nTransformer capital_cost = NPR {annual_transformer_cost:>12,.1f} /MVA/yr")

    # ─────────────────────────────────────────────────────────────
    # 5. LOAD GROWTH
    # ─────────────────────────────────────────────────────────────
    network.loads_t.p_set = load_growth_factor * network.loads_t.p_set

    # ─────────────────────────────────────────────────────────────
    # 6. OPTIMIZE
    # ─────────────────────────────────────────────────────────────
    #network.optimize()

    return network