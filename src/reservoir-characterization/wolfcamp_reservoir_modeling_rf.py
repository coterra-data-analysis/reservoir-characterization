import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata, interp1d
from pyproj import Transformer
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def detect_grid_node_size(grid_xyz: pd.DataFrame, x_field="x", y_field="y") -> float:
    """Detects median distance to nearest neighbor in XYZ dataframe

    Args:
        grid_xyz (pd.DataFrame): Grid with x, y columns
        x_field (str): defaults to 'x'
        y_field (str): defaults to 'y'

    Returns:
        float: Median distance to neighbor
    """
    tree = cKDTree(np.c_[grid_xyz.x.values, grid_xyz.y.values])
    dist, _ = tree.query(np.c_[grid_xyz[x_field].values, grid_xyz[y_field].values], k=2)
    dist = dist.flatten()
    dist = dist[dist != 0]
    median_dist = np.median(dist)
    return median_dist


def mask_valid_data(
    grid_x: np.array, grid_y: np.array, grid_z: np.array, grid_xyz: pd.DataFrame
) -> np.array:
    """Removes interpolated data that is not within the normal bounds of input XYZ file

    Args:
        grid_x (np.array): Interpolated x values
        grid_y (np.array): Interpolated y values
        grid_z (np.array): Interpolated z values
        grid_xyz (pd.DataFrame): Original dataframe with columns x,y,z

    Returns:
        np.array: Masked grid_z with Nan where interpolated values are out-of-bounds
    """
    tree = cKDTree(np.c_[grid_xyz.x.values, grid_xyz.y.values])
    dist, _ = tree.query(np.c_[grid_x.ravel(), grid_y.ravel()], k=1)
    dist = dist.reshape(grid_x.shape)
    grid_xyz_node_size = detect_grid_node_size(grid_xyz)
    grid_z[dist > grid_xyz_node_size * 1.1] = np.nan
    return grid_z


def interpolate_xyz_grid(
    grid_xyz: pd.DataFrame, desired_node_size: float, interpolation_method="linear"
) -> tuple:
    """Interpolate XYZ dataframe to new node size using an interpolation method

    Args:
        grid_xyz (pd.DataFrame): Input with columns x,y,z
        desired_node_size (float): New grid node size
        interpolation_method (str): 'linear' or 'cubic'

    Returns:
        tuple: (x_grid:np.array, y_grid:np.array, z_grid:np.array)
    """
    if desired_node_size == 0:
        desired_node_size = detect_grid_node_size(grid_xyz)
    grid_interp_x, grid_interp_y = np.mgrid[
        grid_xyz.x.min() : grid_xyz.x.max() : desired_node_size,
        grid_xyz.y.min() : grid_xyz.y.max() : desired_node_size,
    ]
    grid_interp_z = griddata(
        grid_xyz[["x", "y"]].to_numpy(),
        grid_xyz.z,
        (grid_interp_x, grid_interp_y),
        method=interpolation_method,
    )
    grid_interp_z = mask_valid_data(
        grid_interp_x, grid_interp_y, grid_interp_z, grid_xyz
    )
    return grid_interp_x, grid_interp_y, grid_interp_z


def interpolate_grid(
    interpolation_method: str,
    grid: pd.DataFrame,
    target_x: np.array,
    target_y: np.array,
):
    interp_z = griddata(
        grid[["x", "y"]].to_numpy(),
        grid[grid.columns[2]].to_numpy(),
        (target_x, target_y),
        method=interpolation_method,
    )
    interp_z = mask_valid_data(target_x, target_y, interp_z, grid)
    grid = pd.DataFrame(
        np.c_[target_x.ravel(), target_y.ravel(), interp_z.ravel()],
        columns=["x", "y", grid.columns[2]],
    )
    return grid


def condition_pvt_lookup_table(
    lookup_table_pvt: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.array]:
    lookup_table_pvt = lookup_table_pvt.fillna(0)
    lookup_table_pvt = lookup_table_pvt.sort_values(
        by=["ogri", "temp", "pi"], ignore_index=True
    )
    lookup_table_pvt_nodes = np.array(
        [
            lookup_table_pvt["ogri"].values,
            lookup_table_pvt["temp"].values,
            lookup_table_pvt["pi"].values,
        ]
    ).T
    return lookup_table_pvt, lookup_table_pvt_nodes


def sample_res_grids_to_common_nodes(
    xyz_structure: str,
    xyz_bvhce: str,
    xyz_ogri: str,
    xyz_wgrp: str,
    desired_node_size: float = 0,
    interpolation_method: str = "linear",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Takes list of XYZ grids and samples them to common nodes. Returns dataframe of X,Y nodes with Z columns corresponding to grid values

    Args:
        grid_file_list (List): Grid filepaths
        desired_node_size (float): Distance between each grid point in resampled grid. Defaults to 0 (lowest detected node size among inputs)
        interpolation_method (str, optional): 'linear' or 'cubic'. Defaults to 'linear'.

    Returns:
        pd.Series: tvd
        pd.Series: bvhce
        pd.Series: ogri
        pd.Series: wgrp
    """

    # Read input grids
    structure = pd.read_csv(xyz_structure, names=["x", "y", "tvd"])
    bvhce = pd.read_csv(xyz_bvhce, names=["x", "y", "bvhce"])
    ogri = pd.read_csv(xyz_ogri, names=["x", "y", "ogri"])
    wgrp = pd.read_csv(xyz_wgrp, names=["x", "y", "wgrp"])

    structure = structure.sort_values(by=["x", "y"]).reset_index(drop=True)
    bvhce = bvhce.sort_values(by=["x", "y"]).reset_index(drop=True)
    ogri = ogri.sort_values(by=["x", "y"]).reset_index(drop=True)
    wgrp = wgrp.sort_values(by=["x", "y"]).reset_index(drop=True)

    # Reproject grids if necessary
    for grid in [structure, bvhce, ogri, wgrp]:
        if np.max(abs(grid.x)) <= 180:
            transformer = Transformer.from_crs(
                "epsg:4267", "epsg:32039", always_xy=True
            )
            grid.x, grid.y = transformer.transform(grid.x, grid.y)

    # Detect smallest grid node size
    if desired_node_size == 0:
        ns = detect_grid_node_size(structure)
        for grid in [bvhce, ogri, wgrp]:
            ns_grid = detect_grid_node_size(grid)
            if ns_grid < ns:
                ns = ns_grid
        desired_node_size = ns

    # Detect x and y ranges in original data
    min_x = structure.x.min()
    max_x = structure.x.max()
    min_y = structure.y.min()
    max_y = structure.y.max()
    for grid in [bvhce, ogri, wgrp]:
        if grid.x.min() < min_x:
            min_x = grid.x.min()
        if grid.x.max() > max_x:
            max_x = grid.x.max()
        if grid.y.min() < min_y:
            min_y = grid.y.min()
        if grid.y.max() > max_y:
            max_y = grid.y.max()

    # Determine new nodeset and project values to it
    target_x, target_y = np.mgrid[
        min_x:max_x:desired_node_size, min_y:max_y:desired_node_size
    ]

    ogri_2 = interpolate_grid(interpolation_method, ogri, target_x, target_y)
    bvhce_2 = interpolate_grid(interpolation_method, bvhce, target_x, target_y)
    wgrp_2 = interpolate_grid(interpolation_method, wgrp, target_x, target_y)
    structure_2 = interpolate_grid(interpolation_method, structure, target_x, target_y)

    # Merge into one dataframe
    res_df = pd.merge(structure_2, bvhce_2, on=["x", "y"], how="left")
    res_df = pd.merge(res_df, ogri_2, on=["x", "y"], how="left")
    res_df = pd.merge(res_df, wgrp_2, on=["x", "y"], how="left")
    res_df = res_df.dropna(how="any")

    res_df = res_df.set_index(["x", "y"])
    return res_df["tvd"], res_df["bvhce"], res_df["ogri"], res_df["wgrp"]


def pi__delaware(tvd: pd.Series) -> pd.Series:
    # Based on MHolland fit between TVD and SJereij dfit interpretation
    # Piecewise function should be defined as pandas sets to be applied to tvd series
    # "Apply stepwise vector / dictionary"
    def dfit_pi(tvd: float) -> float:
        if tvd < 7000:
            return tvd * 0.45
        elif (tvd >= 7000) & (tvd < 10000):
            return (tvd * (tvd - 1600)) / 12000
        elif tvd >= 10000:
            return (tvd * (tvd + 7500)) / 25000

    pi = pd.Series(data=tvd.apply(dfit_pi), name="pi",dtype='float64')
    return pi


def ti__delaware(tvd: pd.Series) -> pd.Series:
    ti = pd.Series(((tvd / 100) * 1.1) + 68, name="ti",dtype='float64')
    return ti


def psat__delaware(
    ogri: pd.Series, ti: pd.Series, psat_lookup_table: pd.DataFrame
) -> pd.Series:
    ogri.loc[ogri<25] = 25
    psat_lookup_table = psat_lookup_table.sort_values(
        by=["ogri", "temp"], ignore_index=True
    )
    psat_nodes = np.array(
        [psat_lookup_table["ogri"].values, psat_lookup_table["temp"].values]
    ).T
    psat_values = np.array(psat_lookup_table["psat"].values)
    psat = pd.Series(
        griddata(psat_nodes, psat_values, np.c_[ogri, ti], method="linear"),
        index=ogri.index,
        name="psat",
        dtype='float64'
    )

    return psat


def fluid_type__delaware(
    ogri: pd.Series, ti: pd.Series, lookup_table_critical_temperature: pd.DataFrame
) -> pd.Series:

    lookup_table_critical_temperature = lookup_table_critical_temperature.sort_values(
        by="ogri", ignore_index=True
    )
    lookup_table_nodes_ogri = np.array(
        lookup_table_critical_temperature["ogri"].values
    ).T
    lookup_table_nodes_critical_temperature = np.array(
        lookup_table_critical_temperature["temp"].values
    )
    critical_temperature = pd.Series(
        griddata(
            lookup_table_nodes_ogri,
            lookup_table_nodes_critical_temperature,
            ogri,
            method="linear",
        ),
        name="crit_point",
        index=ogri.index,
        dtype='float64'
    ).fillna(0)
    mask_fluid_type = pd.Series(ti.gt(critical_temperature))
    fluid_type = pd.Series("Gas", name="fluid_type", index=ti.index).where(
        mask_fluid_type, other="Oil"
    ).astype('category')
    return fluid_type


def reservoir_type__delaware(pi: pd.Series, psat: pd.Series) -> pd.Series:

    mask_reservoir_type = pd.Series(pi.ge(psat))
    reservoir_type = pd.Series(
        "Undersaturated", name="reservoir_type", index=pi.index
    ).where(mask_reservoir_type, other="Saturated").astype('category')
    return reservoir_type


def ratio_rsi_rvi__delaware(
    reservoir_type: pd.Series, ogri: pd.Series, pi: pd.Series, psat: pd.Series
) -> pd.Series:
    ratio_rsi_rvi = pd.Series(np.NaN,index=reservoir_type.index,name='ratio_rsi_rvi',dtype='float64')
    ratio_rsi_rvi.loc[reservoir_type == "Saturated"] = ogri + 40.61 - (0.11 * (pi - psat))
    ratio_rsi_rvi.loc[reservoir_type != "Saturated"] = ogri
    return ratio_rsi_rvi


def swi__delaware(template: pd.Series) -> float:
    swi = pd.Series(0.5, index=template.index, name="swi",dtype='float64')
    return swi


def cf__delaware(template: pd.Series) -> float:
    cf = pd.Series(0.00000564, index=template.index, name="cf",dtype='float64')
    return cf


def pbar_ab__delaware(template: pd.Series) -> float:
    pbar_ab = pd.Series(500, index=template.index, name="pbar_ab",dtype='float64')
    return pbar_ab


def pvt_rf__delaware(
    lookup_table_pvt: pd.DataFrame,
    lookup_table_pvt_nodes: np.array,
    lookup_variable: str,
    return_variable: str,
    ogri_or_yield: pd.Series,
    temperature: pd.Series,
    pressure: pd.Series,
) -> pd.Series:
    lookup_table_values = np.array(lookup_table_pvt[lookup_variable].values)
    pvt_return = pd.Series(
        griddata(
            lookup_table_pvt_nodes,
            lookup_table_values,
            np.c_[ogri_or_yield, temperature, pressure],
            method="linear",
        ),
        name=return_variable,
        index=ogri_or_yield.index,
        dtype='float64',
    )
    return pvt_return


def sg_so_saturations__delaware(
    reservoir_type: pd.Series,
    fluid_type: pd.Series,
    swi: pd.Series,
    bo_i: pd.Series,
    rv_i: pd.Series,
    rs_i: pd.Series,
    ratio_rsi_rvi: pd.Series,
    bgd_i: pd.Series,
    ogri: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Calculates sg and so saturations
    Returns:
        Tuple[pd.Series, pd.Series]: sg,so
    """
    mask_undersaturated_gas = (reservoir_type == "Undersaturated") & (
        fluid_type == "Gas"
    )
    mask_undersaturated_oil = (reservoir_type == "Undersaturated") & (
        fluid_type == "Oil"
    )
    mask_saturated = reservoir_type == "Saturated"

    ratio = (bo_i * (1 - rv_i * 1000 / ratio_rsi_rvi)) / (bgd_i * (1000 / ogri - rs_i))
    sgrel = 1 / (1 + ratio)
    sorel = 1 - sgrel

    sg = pd.Series(np.NaN, index=reservoir_type.index, name="sg",dtype='float64')
    so = pd.Series(np.NaN, index=reservoir_type.index, name="so",dtype='float64')

    sg.loc[mask_undersaturated_gas] = 1 - swi
    sg.loc[mask_undersaturated_oil] = 0
    sg.loc[mask_saturated] = sgrel * (1 - swi)

    so.loc[mask_undersaturated_gas] = 0
    so.loc[mask_undersaturated_oil] = 1 - swi
    so.loc[mask_saturated] = sorel * (1 - swi)
    return sg, so


def salinity__delaware(template: pd.Series) -> float:
    salinity = pd.Series(0, index=template.index, name="salinity",dtype='float64')
    return salinity


def pvt_term__delaware(
    reservoir_type: pd.Series,
    fluid_type: pd.Series,
    rv_i: pd.Series,
    bgd_i: pd.Series,
    cg_i: pd.Series,
    mug_i: pd.Series,
    bo_i: pd.Series,
    co_i: pd.Series,
    muo_i: pd.Series,
) -> pd.Series:
    pvt_term = pd.Series(np.NaN, index=reservoir_type.index, name="pvt_term",dtype='float64')
    mask_saturated_or_gas = (reservoir_type == "Saturated") | (fluid_type == "Gas")
    pvt_term.loc[mask_saturated_or_gas] = (rv_i / bgd_i * np.sqrt(cg_i / mug_i)) * 1000
    pvt_term.loc[~mask_saturated_or_gas] = (1 / bo_i * np.sqrt(co_i / muo_i)) * 1000
    return pvt_term


def bti__delaware(
    swi: pd.Series,
    so: pd.Series,
    bo_i: pd.Series,
    sg: pd.Series,
    rv_i: pd.Series,
    bgd_i: pd.Series,
) -> pd.Series:
    bti = (1 - swi) / (so / bo_i + sg * rv_i / bgd_i)
    bti = pd.Series(bti, index=swi.index, name="bti",dtype='float64')
    return bti


def ooip__delaware(bvhce: pd.Series, bti: pd.Series) -> pd.Series:
    ooip = (7758 * bvhce * (1 / bti)) / 1000
    ooip = pd.Series(ooip, index=bvhce.index, name="ooip",dtype='float64')
    return ooip


def ogip__delaware(ooip: pd.Series, ogri: pd.Series) -> pd.Series:
    ogip = ooip * 1000 / ogri
    ogip = pd.Series(ogip, index=ooip.index, name="ogip",dtype='float64')
    return ogip


def yield_decline__delaware(ogri: pd.Series, pi: pd.Series) -> pd.Series:
    yield_decline = 0.90733797 + 0.00059782 * ogri - 0.00009292 * pi
    yield_decline = pd.Series(yield_decline, index=ogri.index, name="yield_decline",dtype='float64')
    return yield_decline


def ogrp__delaware(ogri: pd.Series, yield_decline: pd.Series) -> pd.Series:
    ogrp = ogri - ogri * yield_decline
    ogrp = pd.Series(ogrp, index=ogri.index, name="ogrp",dtype='float64')
    return ogrp


def bw_gas_free_water(ti: pd.Series, pbar_ab: pd.Series) -> pd.Series:
    A1 = 0.9947
    A2 = 0.0000058
    A3 = 0.00000102
    B1 = -0.000004228
    B2 = 0.000000018376
    B3 = -0.0000000000677
    C1 = 0.00000000013
    C2 = -1.3855 * 10 ** (-12)
    C3 = 4.285 * 10 ** (-15)
    A = A1 + A2 * ti + A3 * ti ** 2
    B = B1 + B2 * ti + B3 * ti ** 2
    C = C1 + C2 * ti + C3 * ti ** 2
    bw = pd.Series(A + B * pbar_ab + C * pbar_ab ** 2, index=ti.index, name="bw",dtype='float64')
    return bw


def cw_meehan(pi: pd.Series, ti: pd.Series, salinity: pd.Series) -> pd.Series:
    A = 3.8546 - 0.000134 * pi
    B = -0.01052 + 4.77 * 10 ** (-7) * pi
    C = 3.9267 * 10 ** (-5) - 8.8 * 10 ** (-10) * pi
    CZ = (
        1
        + (-0.052 + 0.00027 * ti - 0.00000114 * ti ** 2 + 0.000000001121 * ti ** 3)
        * (salinity / 10000) ** 0.7
    )
    cw = pd.Series((10 ** (-6)) * (A + B * ti + C * ti ** 2), index=pi.index, name="cw",dtype='float64')
    cw.loc[(salinity == 0) | (salinity.isna())] = cw
    cw.loc[(salinity != 0) & (~salinity.isna())] = CZ * cw
    return cw


def m__delaware(sg: pd.Series, so: pd.Series) -> pd.Series:
    m = pd.Series(np.NaN, index=so.index, name="m",dtype='float64')
    m.loc[so == 0] = 10 ** 25
    m.loc[so != 0] = sg.loc[so != 0] / so.loc[so != 0]
    return m


def rp__delaware(ogrp: pd.Series) -> pd.Series:
    rp = pd.Series(np.NaN, index=ogrp.index, name="rp",dtype='float64')
    rp.loc[ogrp == 0] = 10 ** 25
    rp.loc[ogrp != 0] = 1 / ogrp * 1000
    return rp


def wor__delaware(wgrp: pd.Series, rp: pd.Series) -> pd.Series:
    wor = wgrp / (1 / rp * 1000)
    wor = pd.Series(wor, index=wgrp.index, name="wor",dtype='float64')
    return wor


def rf_oil__delaware(
    bo_ab: pd.Series,
    rs_i: pd.Series,
    rv_ab: pd.Series,
    bgd_ab: pd.Series,
    rs_ab: pd.Series,
    bo_i: pd.Series,
    pi: pd.Series,
    pbar_ab: pd.Series,
    cf: pd.Series,
    cw: pd.Series,
    swi: pd.Series,
    m: pd.Series,
    rp: pd.Series,
    rv_i: pd.Series,
    bgd_i: pd.Series,
    wor: pd.Series,
    bw: pd.Series,
) -> pd.Series:

    Eowf = (
        (bo_ab * (1 - rs_i * rv_ab) + bgd_ab * (rs_i - rs_ab)) / (1 - rs_ab * rv_ab)
        - bo_i
        + bo_i * (pi - pbar_ab) * ((cf + cw * swi) / (1 - swi))
    )
    Egwf = (
        (bgd_ab * (1 - rs_ab * rv_i) + bo_ab * (rv_i - rv_ab)) / (1 - rs_ab * rv_ab)
        - bgd_i
        + bgd_i * (pi - pbar_ab) * ((cf + cw * swi) / (1 - swi))
    )
    rf_oil = (Eowf + m * bo_i * Egwf / bgd_i) / (
        (1 + m * bo_i * rv_i / bgd_i)
        * (
            (bo_ab * (1 - rv_ab * rp) + bgd_ab * (rp - rs_ab)) / (1 - rv_ab * rs_ab)
            + wor * bw
        )
    )
    rf_oil = pd.Series(rf_oil, index=bo_ab.index, name="rf_oil",dtype='float64')
    return rf_oil


def rf_gas__delaware(
    rp: pd.Series,
    rf_oil: pd.Series,
    bgd_i: pd.Series,
    m: pd.Series,
    bo_i: pd.Series,
    rv_i: pd.Series,
    rs_i: pd.Series,
) -> pd.Series:
    rf_gas = rp * rf_oil * (bgd_i + m * bo_i * rv_i) / (m * bo_i + bgd_i * rs_i)
    rf_gas = pd.Series(rf_gas, index=rf_oil.index, name="rf_gas",dtype='float64')
    return rf_gas


def roip__delaware(ooip: pd.Series, rf_oil: pd.Series) -> pd.Series:
    roip = ooip * rf_oil
    roip = pd.Series(roip, index=ooip.index, name="roip",dtype='float64')
    return roip


def rgip__delaware(ogip: pd.Series, rf_gas: pd.Series) -> pd.Series:
    rgip = ogip * rf_gas
    rgip = pd.Series(rgip, index=ogip.index, name="rgip",dtype='float64')
    return rgip


def plot_output(output:pd.DataFrame)->None:
    for col in set(output.columns)-set(['x','y']):
        fig, ax = plt.subplots(1, 1, figsize=[20, 20])
        if col not in [
            "fluid_type_numeric",
            "reservoir_type_numeric",
            "so_isNegative",
            "co_i",
            "cg_i",
            "bti",
            "pvt_term",
            "fluid_type_boundary_proximity",
            "fluid_type_special_cols",
        ]:
            output.plot.scatter("x", "y", c=col, colormap="rainbow", ax=ax)
        elif col in [
            "co_i",
            "cg_i",
            "bti",
            "pvt_term",
            "fluid_type_boundary_proximity",
        ]:
            if col in ["co_i", "cg_i"]:
                output.plot.scatter(
                    "x",
                    "y",
                    c=col,
                    colormap="rainbow",
                    ax=ax,
                    norm=LogNorm(vmin=0.00001, vmax=0.0001),
                )
            if col == "bti":
                output.plot.scatter(
                    "x",
                    "y",
                    c=col,
                    colormap="rainbow",
                    ax=ax,
                    norm=LogNorm(vmin=1, vmax=10),
                )
            if col == "pvt_term":
                output.plot.scatter(
                    "x",
                    "y",
                    c=col,
                    colormap="rainbow",
                    ax=ax,
                    norm=LogNorm(vmin=2, vmax=10),
                )
            if col == "fluid_type_boundary_proximity":
                output.plot.scatter(
                    "x", "y", c=col, colormap=output, ax=ax, vmin=-200, vmax=200
                )
        else:
            output.plot.scatter("x", "y", c=col, ax=ax)
        ax.set_xlim([output["x"].min() - 5280, output["x"].max() + 5280])
        ax.set_ylim([output["y"].min() - 5280, output["y"].max() + 5280])
        ax.set_title(col)
        ax.set_aspect("equal", "box")
        plt.savefig(f"refactor_{col}.png", bbox_inches="tight")
        # plt.show()
        plt.close()


def wolfcamp_petrophysical_modeling():
    PSAT_LOOKUP_TABLE = r"C:\Users\kdavison\OneDrive - Cimarex Energy Company\Documents\Python Scripts\wolfcamp reservoir modeling\psat_lookup_table.csv"
    CRIT_POINT_OGRI_TEMP_TABLE = r"C:\Users\kdavison\OneDrive - Cimarex Energy Company\Documents\Python Scripts\wolfcamp reservoir modeling\critical_point_ogri_temp_lookup_table.csv"
    PVT_LOOKUP_TABLE = r"C:\Users\kdavison\OneDrive - Cimarex Energy Company\Documents\Python Scripts\wolfcamp reservoir modeling\pvt_lookup.csv"

    GRID_TVD = r"P:\projects\PB DELAWARE BASIN\GRIDS\XECG\Petra Grids_Alt\WFMP_RES_CHAR\WFMP_A1_Upper_TVD_Structure.XYZ"
    GRID_OGRI = r"P:\projects\PB DELAWARE BASIN\GRIDS\XECG\Petra Grids_Alt\WFMP_RES_CHAR\OGRi.XYZ"
    GRID_BVHCE = r"P:\projects\PB DELAWARE BASIN\GRIDS\XECG\Petra Grids_Alt\WFMP_RES_CHAR\BVHCe.XYZ"
    GRID_WGRP = r"P:\projects\PB DELAWARE BASIN\GRIDS\XECG\Petra Grids_Alt\WFMP_RES_CHAR\WGRp.XYZ"

    lookup_table_psat = pd.read_csv(PSAT_LOOKUP_TABLE)
    lookup_table_crit_point_ogri_temp = pd.read_csv(CRIT_POINT_OGRI_TEMP_TABLE)
    lookup_table_pvt = pd.read_csv(PVT_LOOKUP_TABLE)
    lookup_table_pvt, lookup_table_pvt_nodes = condition_pvt_lookup_table(
        lookup_table_pvt=lookup_table_pvt
    )

    tvd, bvhce, ogri, wgrp = sample_res_grids_to_common_nodes(
        xyz_structure=GRID_TVD,
        xyz_bvhce=GRID_BVHCE,
        xyz_ogri=GRID_OGRI,
        xyz_wgrp=GRID_WGRP,
    )
    pi = pi__delaware(tvd=tvd)
    ti = ti__delaware(tvd=tvd)
    psat = psat__delaware(ogri=ogri, ti=ti, psat_lookup_table=lookup_table_psat)
    fluid_type = fluid_type__delaware(
        ogri=ogri,
        ti=ti,
        lookup_table_critical_temperature=lookup_table_crit_point_ogri_temp,
    )
    reservoir_type = reservoir_type__delaware(pi=pi, psat=psat)
    ratio_rsi_rvi = ratio_rsi_rvi__delaware(
        reservoir_type=reservoir_type, ogri=ogri, pi=pi, psat=psat
    )
    swi = swi__delaware(template=tvd)
    cf = cf__delaware(template=tvd)
    pbar_ab = pbar_ab__delaware(template=tvd)
    bgd_i = (
        pvt_rf__delaware(
            lookup_table_pvt=lookup_table_pvt,
            lookup_table_pvt_nodes=lookup_table_pvt_nodes,
            lookup_variable="bgd",
            return_variable="bgd_i",
            ogri_or_yield=ratio_rsi_rvi,
            temperature=ti,
            pressure=pi,
        )
        * 1000
        / 5.615
    )
    bo_i = pvt_rf__delaware(
        lookup_table_pvt=lookup_table_pvt,
        lookup_table_pvt_nodes=lookup_table_pvt_nodes,
        lookup_variable="bo",
        return_variable="bo_i",
        ogri_or_yield=ratio_rsi_rvi,
        temperature=ti,
        pressure=pi,
    )
    rv_i = (
        pvt_rf__delaware(
            lookup_table_pvt=lookup_table_pvt,
            lookup_table_pvt_nodes=lookup_table_pvt_nodes,
            lookup_variable="ratio_rs_rv",
            return_variable="rv_i",
            ogri_or_yield=ratio_rsi_rvi,
            temperature=ti,
            pressure=pi,
        )
        / 1000
    )
    rs_i = (
        pvt_rf__delaware(
            lookup_table_pvt=lookup_table_pvt,
            lookup_table_pvt_nodes=lookup_table_pvt_nodes,
            lookup_variable="rs",
            return_variable="rs_i",
            ogri_or_yield=ratio_rsi_rvi,
            temperature=ti,
            pressure=pi,
        )
        / 1000
    )
    bgd_ab = (
        pvt_rf__delaware(
            lookup_table_pvt=lookup_table_pvt,
            lookup_table_pvt_nodes=lookup_table_pvt_nodes,
            lookup_variable="bgd",
            return_variable="bgd_ab",
            ogri_or_yield=ratio_rsi_rvi,
            temperature=ti,
            pressure=pbar_ab,
        )
        * 1000
        / 5.615
    )
    bo_ab = pvt_rf__delaware(
        lookup_table_pvt=lookup_table_pvt,
        lookup_table_pvt_nodes=lookup_table_pvt_nodes,
        lookup_variable="bo",
        return_variable="bo_ab",
        ogri_or_yield=ratio_rsi_rvi,
        temperature=ti,
        pressure=pbar_ab,
    )
    rv_ab = (
        pvt_rf__delaware(
            lookup_table_pvt=lookup_table_pvt,
            lookup_table_pvt_nodes=lookup_table_pvt_nodes,
            lookup_variable="ratio_rs_rv",
            return_variable="rv_ab",
            ogri_or_yield=ratio_rsi_rvi,
            temperature=ti,
            pressure=pbar_ab,
        )
        / 1000
    )
    rs_ab = (
        pvt_rf__delaware(
            lookup_table_pvt=lookup_table_pvt,
            lookup_table_pvt_nodes=lookup_table_pvt_nodes,
            lookup_variable="rs",
            return_variable="rs_ab",
            ogri_or_yield=ratio_rsi_rvi,
            temperature=ti,
            pressure=pbar_ab,
        )
        / 1000
    )
    co_i = pvt_rf__delaware(
        lookup_table_pvt=lookup_table_pvt,
        lookup_table_pvt_nodes=lookup_table_pvt_nodes,
        lookup_variable="co",
        return_variable="co_i",
        ogri_or_yield=ogri,
        temperature=ti,
        pressure=pi,
    )
    cg_i = pvt_rf__delaware(
        lookup_table_pvt=lookup_table_pvt,
        lookup_table_pvt_nodes=lookup_table_pvt_nodes,
        lookup_variable="cg",
        return_variable="cg_i",
        ogri_or_yield=ogri,
        temperature=ti,
        pressure=pi,
    )
    muo_i = pvt_rf__delaware(
        lookup_table_pvt=lookup_table_pvt,
        lookup_table_pvt_nodes=lookup_table_pvt_nodes,
        lookup_variable="muo",
        return_variable="muo_i",
        ogri_or_yield=ogri,
        temperature=ti,
        pressure=pi,
    )
    mug_i = pvt_rf__delaware(
        lookup_table_pvt=lookup_table_pvt,
        lookup_table_pvt_nodes=lookup_table_pvt_nodes,
        lookup_variable="mug",
        return_variable="mug_i",
        ogri_or_yield=ogri,
        temperature=ti,
        pressure=pi,
    )

    sg, so = sg_so_saturations__delaware(
        reservoir_type=reservoir_type,
        fluid_type=fluid_type,
        swi=swi,
        bo_i=bo_i,
        rv_i=rv_i,
        rs_i=rs_i,
        ratio_rsi_rvi=ratio_rsi_rvi,
        bgd_i=bgd_i,
        ogri=ogri,
    )
    salinity = salinity__delaware(template=tvd)
    pvt_term = pvt_term__delaware(
        reservoir_type=reservoir_type,
        fluid_type=fluid_type,
        rv_i=rv_i,
        bgd_i=bgd_i,
        cg_i=cg_i,
        mug_i=mug_i,
        bo_i=bo_i,
        co_i=co_i,
        muo_i=muo_i,
    )
    bti = bti__delaware(swi=swi, so=so, bo_i=bo_i, sg=sg, rv_i=rv_i, bgd_i=bgd_i)
    ooip = ooip__delaware(bvhce=bvhce, bti=bti)
    ogip = ogip__delaware(ooip=ooip, ogri=ogri)
    yield_decline = yield_decline__delaware(ogri=ogri, pi=pi)
    ogrp = ogrp__delaware(ogri=ogri, yield_decline=yield_decline)
    bw = bw_gas_free_water(ti=ti, pbar_ab=pbar_ab)
    cw = cw_meehan(pi=pi, ti=ti, salinity=salinity)
    m = m__delaware(sg=sg, so=so)
    rp = rp__delaware(ogrp=ogrp)
    wor = wor__delaware(wgrp=wgrp, rp=rp)
    rf_oil = rf_oil__delaware(
        bo_ab=bo_ab,
        rs_i=rs_i,
        rv_ab=rv_ab,
        bgd_ab=bgd_ab,
        rs_ab=rs_ab,
        bo_i=bo_i,
        pi=pi,
        pbar_ab=pbar_ab,
        cf=cf,
        cw=cw,
        swi=swi,
        m=m,
        rp=rp,
        rv_i=rv_i,
        bgd_i=bgd_i,
        wor=wor,
        bw=bw,
    )
    rf_gas = rf_gas__delaware(
        rp=rp, rf_oil=rf_oil, bgd_i=bgd_i, m=m, bo_i=bo_i, rv_i=rv_i, rs_i=rs_i
    )
    roip = roip__delaware(ooip=ooip, rf_oil=rf_oil)
    rgip = rgip__delaware(ogip=ogip, rf_gas=rf_gas)

    output_cols = [
        "bti",
        "ogip",
        "ogrp",
        "ooip",
        "pi",
        "rf_gas",
        "rf_oil",
        "rgip",
        "roip",
        "ti",
        "yield_decline",
    ]

    output = pd.concat(
        [
            tvd,
            bvhce,
            ogri,
            wgrp,
            pi,
            ti,
            psat,
            fluid_type,
            reservoir_type,
            ratio_rsi_rvi,
            swi,
            cf,
            pbar_ab,
            bgd_i,
            bo_i,
            rv_i,
            rs_i,
            bgd_ab,
            bo_ab,
            rv_ab,
            rs_ab,
            co_i,
            cg_i,
            muo_i,
            mug_i,
            sg,
            so,
            salinity,
            pvt_term,
            bti,
            ooip,
            ogip,
            yield_decline,
            ogrp,
            m,
            rp,
            bw,
            wor,
            cw,
            rf_oil,
            rf_gas,
            roip,
            rgip,
        ],
        axis=1,
    ).reset_index()
    pass

    


if __name__ == "__main__":
    wolfcamp_petrophysical_modeling()
