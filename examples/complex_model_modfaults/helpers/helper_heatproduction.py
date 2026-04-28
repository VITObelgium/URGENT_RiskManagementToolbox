from iapws.iapws97 import _Region1


def cumulative_heat(td, PRD, INJ, a=0):
    """
    Computes cumulative heat production during the entire simulation for one specific doublet
    td: dataframe with well information
    PRD: name of producer well
    INJ: name of injector welL
    a: time index in td where production starts, a=0 is default assumes production
    """

    Tags = {
        "T": " : temperature (K)",
        "P": " : BHP (bar)",
        "Q": " : water rate (m3/day)",
    }  # ROOTS of the tags in td dataframe that are needed for accessing data
    # reading well vectors
    time = td["time"]
    tprod = td[PRD + Tags["T"]]
    tinje = td[INJ + Tags["T"]]
    Qprd = td[PRD + Tags["Q"]]  # VOLUMETRIC FLOW [m3/hr]
    BHP_PRD = td[PRD + Tags["P"]]
    BHP_INJ = td[INJ + Tags["P"]]

    CP = 4200  # j/kg*K specific heat capacity of water
    Heat_v = 0  # [MWyear]
    Energy_exp = 0  # [MWyear]
    a = 0  # Index that indicates where the production period starts. It is a=0 if the production starts right away
    for i in range(a, len(time[a:]) + a - 1):
        #'values' provide a list of 1 element then to get the item we use [0]
        Tinj = tinje.values[i + 1 : i + 2][
            0
        ]  # Injection temperature , values to extract
        dtime = (
            time.values[i + 1 : i + 2][0] - time.values[i : i + 1][0]
        ) / 365  # delta time is in years
        dTemp = tprod.values[i + 1 : i + 2][0] - Tinj
        BHP_i = BHP_PRD.values[i + 1 : i + 2][0]
        Temp_prod_i = tprod.values[i + 1 : i + 2][0]
        Qprd_i = -1 * (Qprd.values[i + 1 : i + 2][0])  # VOLUMETRIC FLOW [m3/hr]
        rho_water = (
            1 / (_Region1(Temp_prod_i, (BHP_i) * 0.1)["v"])
        )  # density [kg/m3], 'v' is specific volume

        # v stands for calculation using volumetric rate
        Qmass_v = Qprd_i / (24 * 3600) * rho_water  # [kg/sec]
        dtHeat_v = Qmass_v * CP * dTemp / 1e6 * dtime  # [MWyear]
        dtpower_V = Qmass_v * CP * dTemp / 1e6  # [MW]

        # expended energy
        pump_efficiency = 0.5  # dimensionless ESP efficiency
        DPinj_i = abs(
            (BHP_INJ.values[i + 1 : i + 2][0]) - (BHP_INJ.values[a : a + 1][0])
        )  #
        DPpro_i = abs(
            (BHP_PRD.values[a : a + 1][0]) - (BHP_PRD.values[i + 1 : i + 2][0])
        )  #
        dtEnergy_exp = (
            Qprd_i
            * 1000
            / (3600 * 24)
            / rho_water
            * (DPinj_i + DPpro_i)
            * 1e5
            / pump_efficiency
            / 1e6
            * dtime
        )  # [MWyear] expended energy

        Energy_exp = Energy_exp + dtEnergy_exp  # [MWy] cumulative expended energy
        Heat_v = Heat_v + dtHeat_v - dtEnergy_exp  # [MWy] cumulative net

    print("Heat_v[MWy]", Heat_v)
    return Heat_v


def injection_conditions(td, INJ):
    """
    provides the injection temperature and pressure for the injector well with name 'INJ'
    extracted fro td
    """
    Tags = {
        "T": " : temperature (K)",
        "P": " : BHP (bar)",
        "Q": " : water rate (m3/day)",
    }  # ROOTS ot the tags int td dataframe that are needed for accessing data
    Tinj = td[INJ + Tags["T"]].values[-1]  # [K] last value at the current call
    BHPinj = td[INJ + Tags["P"]].values[-1]  # [bar]
    return Tinj, BHPinj
