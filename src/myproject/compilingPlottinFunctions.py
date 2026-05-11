#AP 2026

#try a different method:
from myproject.redshiftGalaxyAnalysis import GalaxyAnalysis
from myproject.utilities.snapshotEnum import SnapshotEnum
from myproject.utilities.plottingTools import AstroPlotter
from myproject.utilities.ListGalaxyGroup import ListGalaxyGroup
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, NullLocator, MultipleLocator

def asymetryPlots(
    redshifts,
    listRedshiftGG: list[list[ListGalaxyGroup]],
    listGGLabels,
    plotRows=1,
    plotCols=1,
    ax=None,
    fig=None,
    plotter=None,
    title=None,
    massColors=None,
    ):
    """Plot asymmetry curves on either a fresh axis or a provided axis (for multipanel use)."""
    if massColors==None:
        #standard matplotlib colors for up to 10 lines, will cycle if more than 10 mass bins
        massColors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    polar_pairwise_by_type_by_redshift: dict[float, dict[str, dict[str, list]]] = {}
    # dict of snapshot -> type -> {'differences': [], 'binned': [], 'bin_centers': [], 'errorbars': []}

    for snapshot, mass_bins in zip(redshifts, listRedshiftGG):
        polar_pairwise_by_type: dict[str, dict[str, list]] = {}
        extended_list_of_full_sample_polar_differences = []
        extended_full_sample_pairwise_polar_differences_binned = None
        extended_full_sample_polar_bin_centers = None
        extended_full_sample_pairwise_polar_errorbars = None

        for i, mass_bin in enumerate(mass_bins):
            list_pairwise_polar_differences = mass_bin.compute_probablity_distribution_of_polar_differences(parallelize=False)

            pairwise_polar_differences_binned, polar_bin_centers, pairwise_polar_errorbars = ListGalaxyGroup.get_histogram_bins(
                list_pairwise_polar_differences,
                bins=np.arange(0, 185, 10),
                errorbarType='poisson',
                normalize_to_one=True,
            )

            label_key = listGGLabels[i]
            polar_pairwise_by_type[label_key] = {
                'differences': list_pairwise_polar_differences,
                'binned': pairwise_polar_differences_binned,
                'bin_centers': polar_bin_centers,
                'errorbars': pairwise_polar_errorbars,
            }

            extended_list_of_full_sample_polar_differences = (
                extended_list_of_full_sample_polar_differences + list_pairwise_polar_differences
            )

            if extended_full_sample_pairwise_polar_differences_binned is None:
                extended_full_sample_pairwise_polar_differences_binned = np.array(pairwise_polar_differences_binned, dtype=float)
                extended_full_sample_polar_bin_centers = np.array(polar_bin_centers, dtype=float)
                extended_full_sample_pairwise_polar_errorbars = np.array(pairwise_polar_errorbars, dtype=float)
            else:
                extended_full_sample_pairwise_polar_differences_binned = (
                    extended_full_sample_pairwise_polar_differences_binned + np.array(pairwise_polar_differences_binned, dtype=float)
                )
                extended_full_sample_pairwise_polar_errorbars = (
                    extended_full_sample_pairwise_polar_errorbars + np.array(pairwise_polar_errorbars, dtype=float)
                )

        polar_pairwise_by_type['full_sample'] = {
            'differences': extended_list_of_full_sample_polar_differences,
            'binned': extended_full_sample_pairwise_polar_differences_binned,
            'bin_centers': extended_full_sample_polar_bin_centers,
            'errorbars': extended_full_sample_pairwise_polar_errorbars,
        }

        polar_pairwise_by_type_by_redshift[snapshot] = polar_pairwise_by_type

    # Sum points below 90 deg and above/equal 90 deg, propagate poisson errors in quadrature.
    for redshift, polar_pairwise_by_type in polar_pairwise_by_type_by_redshift.items():
        for type_key, stats in polar_pairwise_by_type.items():
            lt90_mask = stats['bin_centers'] < 90.0
            ge90_mask = stats['bin_centers'] >= 90.0
            n_lt90 = np.sum(stats['binned'][lt90_mask])
            n_ge90 = np.sum(stats['binned'][ge90_mask])
            diff = n_lt90 - n_ge90
            sigma_poisson = np.sqrt(
                np.sum(stats['errorbars'][lt90_mask] ** 2) + np.sum(stats['errorbars'][ge90_mask] ** 2)
            )
            polar_pairwise_by_type_by_redshift[redshift][type_key]['diff'] = diff
            polar_pairwise_by_type_by_redshift[redshift][type_key]['diff_err'] = sigma_poisson

    by_mass_diff = {type_key: [] for type_key in listGGLabels}
    by_mass_diff_err = {type_key: [] for type_key in listGGLabels}
    redshift_list = []

    for redshift, polar_pairwise_by_type in polar_pairwise_by_type_by_redshift.items():
        redshift_list.append(redshift)
        for type_key in listGGLabels:
            by_mass_diff[type_key].append(polar_pairwise_by_type[type_key]['diff'])
            by_mass_diff_err[type_key].append(polar_pairwise_by_type[type_key]['diff_err'])

    if ax is None:
        polarPairwiseDifference_plotter = AstroPlotter() if plotter is None else plotter
        polarPairwiseDifference_fig, polarPairwiseDifference_ax = polarPairwiseDifference_plotter.create_figure(
            nrows=plotRows, ncols=plotCols
        )
    else:
        polarPairwiseDifference_ax = ax
        polarPairwiseDifference_fig = fig if fig is not None else ax.figure
        polarPairwiseDifference_plotter = plotter

    for i, type_key in enumerate(listGGLabels):
        y_all_by_mass_diff = np.array(by_mass_diff[type_key], dtype=float)
        mass_bin_mask = y_all_by_mass_diff != 0
        # print(np.array(redshift_list)[mass_bin_mask])
        # print(y_all_by_mass_diff[mass_bin_mask])
        if np.any(mass_bin_mask):
            x_redshift = np.array(redshift_list)[mass_bin_mask] + (i - 0.001) * 0.01
            y_by_mass_diff = y_all_by_mass_diff[mass_bin_mask]
            yerr_by_mass_diff = np.array(by_mass_diff_err[type_key])[mass_bin_mask]
            polarPairwiseDifference_plotter.scatter_plot(
                x_redshift,
                y_by_mass_diff,
                errorBars=yerr_by_mass_diff,
                label=f'{type_key}',
                ax=polarPairwiseDifference_ax,
                overlay_color=massColors[i % len(massColors)],
            )

    polarPairwiseDifference_ax.xaxis.set_major_locator(MultipleLocator(0.1))
    polarPairwiseDifference_ax.xaxis.set_minor_locator(NullLocator())
    polarPairwiseDifference_ax.yaxis.set_minor_locator(AutoMinorLocator())
    polarPairwiseDifference_ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    polarPairwiseDifference_ax.tick_params(axis='y', which='minor', length=3)

    polarPairwiseDifference_ax.set_xlabel('Redshift')
    polarPairwiseDifference_ax.set_ylabel('N(<90º) - N(>=90º)')
    if title is not None:
        polarPairwiseDifference_ax.set_title(title)
    polarPairwiseDifference_ax.legend(fontsize=6)

    return (
        polarPairwiseDifference_plotter,
        polarPairwiseDifference_fig,
        polarPairwiseDifference_ax,
        polar_pairwise_by_type_by_redshift,
    )


def getConsistentInconsistentPolarMRLPlots(
    list_gg: ListGalaxyGroup,
    snapshot: int,
    percentConfidenceMRL: float = 95,
    gg_indices_inconsistent: dict[int, list[str]] = None,
    mrl_values=None,
    random_MRL_values=None,
    numsamples_random_mrl=10000,
    polar_ax=None,
    mrl_ax=None,
    polar_fig=None,
    mrl_fig=None,
    polar_plotter=None,
    mrl_plotter=None,
    polar_title=None,
    mrl_title=None,
    ):
    """
    Optionally draw into provided polar_ax and mrl_ax for multipanel figures.
    This lets you keep all polar panels in one figure and all MRL panels in another.
    """
    if gg_indices_inconsistent is None:
        random_MRL_values = list_gg.compute_MRL_random_distribution_curves_for_LGG(parallelize=False, num_samples=1000)
        mrl_values = list_gg.compute_probablity_distribution_of_MRL_directionality(parallelize=False)
        (fraction, count, total, gg_indices_inconsistent) = GalaxyAnalysis.calculate_fraction_less_than_percentile(
            MRL_values=mrl_values,
            random_MRL_values=random_MRL_values,
            percentile=percentConfidenceMRL,
        )

    consistent_polar_differences = []
    inconsistent_polar_differences = []

    projections_all = ["xy", "yz", "xz"]
    proj_to_idx = {"xy": 0, "yz": 1, "xz": 2}
    inconsistent_set = set(gg_indices_inconsistent)

    high_galaxy_groups = []
    non_high_galaxy_groups = []

    for group_index, galaxyGroup in enumerate(list_gg.getAllGalaxyGroups()):
        inconsistentProjections = gg_indices_inconsistent.get(group_index, [])
        consistentProjections = [p for p in projections_all if p not in inconsistentProjections]

        if group_index in inconsistent_set:
            high_galaxy_groups.append(galaxyGroup)
        else:
            non_high_galaxy_groups.append(galaxyGroup)

        if len(consistentProjections) > 0:
            consistent_vals = ListGalaxyGroup._compute_pairwise_for_group(
                galaxyGroup, projections=consistentProjections
            )
            consistent_polar_differences.extend([v for tup in consistent_vals for v in tup])

        if len(inconsistentProjections) > 0:
            inconsistent_vals = ListGalaxyGroup._compute_pairwise_for_group(
                galaxyGroup, projections=inconsistentProjections
            )
            inconsistent_polar_differences.extend([v for tup in inconsistent_vals for v in tup])

    if polar_ax is None:
        polar_plotter = AstroPlotter() if polar_plotter is None else polar_plotter
        polar_fig, polar_ax = polar_plotter.create_figure(ncols=1, nrows=1, figsize=(8, 6))
    else:
        polar_fig = polar_fig if polar_fig is not None else polar_ax.figure

    bin = np.arange(0, 185, 10)
    xlabel = "Pairwise Polar Difference (degrees)"
    ylabel = "Probability Density"
    title = polar_title if polar_title is not None else (
        f"P($\delta \phi$) Distribution (TNG300-1, TNG-Cluster; {percentConfidenceMRL} CL)"
    )

    consistent_bin_values, consistent_bin_centers, consistent_bin_errorbars = ListGalaxyGroup.get_histogram_bins(
        consistent_polar_differences,
        bins=bin,
        errorbarType="poisson",
        density=True,
    )
    inconsistent_bin_values, inconsistent_bin_centers, inconsistent_bin_errorbars = ListGalaxyGroup.get_histogram_bins(
        inconsistent_polar_differences,
        bins=bin,
        errorbarType="poisson",
        density=True,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    consistentColor = colors[0]
    inconsistentColor = colors[1]

    if polar_plotter is not None:
        polar_plotter.scatter_plot(
            consistent_bin_centers,
            consistent_bin_values,
            errorBars=consistent_bin_errorbars,
            ax=polar_ax,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            label=f"Consistent MRL projections (N={len(consistent_polar_differences)})",
            output_filename=None,
            grid=True,
        )
        polar_plotter.scatter_plot(
            inconsistent_bin_centers,
            inconsistent_bin_values,
            errorBars=inconsistent_bin_errorbars,
            ax=polar_ax,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            label=f"Inconsistent MRL projections (N={len(inconsistent_polar_differences)})",
            output_filename=None,
            grid=True,
        )
    else:
        polar_ax.errorbar(
            consistent_bin_centers,
            consistent_bin_values,
            yerr=consistent_bin_errorbars,
            label=f"Consistent MRL projections (N={len(consistent_polar_differences)})",
            marker='o',
            linestyle='-',
            linewidth=1,
            markersize=3,
            capsize=2,
        )
        polar_ax.errorbar(
            inconsistent_bin_centers,
            inconsistent_bin_values,
            yerr=inconsistent_bin_errorbars,
            label=f"Inconsistent MRL projections (N={len(inconsistent_polar_differences)})",
            marker='o',
            linestyle='-',
            linewidth=1,
            markersize=3,
            capsize=2,
        )
        polar_ax.set_xlabel(xlabel)
        polar_ax.set_ylabel(ylabel)
        polar_ax.set_title(title)
        polar_ax.grid(True, alpha=0.3)

    #make random distribution line at 1/180 on across the y axis
    polar_ax.axhline(1/180, color='gray', linestyle='--')#, label='Random Distribution (1/180)')

    frac_inconsistent = (
        len(inconsistent_polar_differences) / (len(consistent_polar_differences) + len(inconsistent_polar_differences))
        if (len(consistent_polar_differences) + len(inconsistent_polar_differences))
        else 0.0
    )
    polar_ax.text(
        0.95,
        0.05,
        f"Fraction inconsistent pairs: {frac_inconsistent:.2f}",
        transform=polar_ax.transAxes,
        fontsize=7,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    polar_ax.legend(fontsize=8)









    # MRL panel (separate figure/axis, also supports external multipanel axis)
    if mrl_ax is None:
        mrl_plotter = AstroPlotter() if mrl_plotter is None else mrl_plotter
        mrl_fig, mrl_ax = mrl_plotter.create_figure(ncols=1, nrows=1, figsize=(8, 6))
    else:
        mrl_fig = mrl_fig if mrl_fig is not None else mrl_ax.figure

    mrl_values_array = np.asarray(mrl_values, dtype=float) if mrl_values is not None else None

    high_list_galaxy_groups = ListGalaxyGroup(high_galaxy_groups)
    non_high_list_galaxy_groups = ListGalaxyGroup(non_high_galaxy_groups)

    inconsistent_random_MRL_values = []
    consistent_random_MRL_values = []
    if high_list_galaxy_groups.getNumGalaxyGroups() > 0:
        inconsistent_random_MRL_values = high_list_galaxy_groups.compute_MRL_random_distribution_curves_for_LGG(
            parallelize=False, num_samples=numsamples_random_mrl
        )
    if non_high_list_galaxy_groups.getNumGalaxyGroups() > 0:
        consistent_random_MRL_values = non_high_list_galaxy_groups.compute_MRL_random_distribution_curves_for_LGG(
            parallelize=False, num_samples=numsamples_random_mrl
        )

    bin = np.arange(0, 1.05, 0.05)
    xlabel = 'MRL Value'
    ylabel = 'Probability Density'
    title = mrl_title if mrl_title is not None else (
        'MRL Value Distribution for TNG300-1, TNG-Cluster Combined Galaxy Groups, Consistent vs Inconsistent Projections'
    )

    def _empty_histogram():
        centers = 0.5 * (bin[:-1] + bin[1:])
        values = np.zeros_like(centers, dtype=float)
        errors = np.zeros_like(centers, dtype=float)
        return values, centers, errors

    if len(inconsistent_random_MRL_values) > 0:
        inconsistent_random_MRL_bins, inconsistent_random_MRL_bin_centers, inconsistent_random_MRL_errorbars = ListGalaxyGroup.get_histogram_bins(
            inconsistent_random_MRL_values, binsize=0.05, binLow=0, binHigh=1, errorbarType='poisson', density=True
        )
        # Add a (0, 0) point for spline fit to pass through origin
        inconsistent_random_MRL_bin_centers = np.concatenate([[0], inconsistent_random_MRL_bin_centers])
        inconsistent_random_MRL_bins = np.concatenate([[0], inconsistent_random_MRL_bins])
        inconsistent_random_MRL_errorbars = np.concatenate([[0], inconsistent_random_MRL_errorbars])
    else:
        inconsistent_random_MRL_bins, inconsistent_random_MRL_bin_centers, inconsistent_random_MRL_errorbars = _empty_histogram()

    if len(consistent_random_MRL_values) > 0:
        consistent_random_MRL_bins, consistent_random_MRL_bin_centers, consistent_random_MRL_errorbars = ListGalaxyGroup.get_histogram_bins(
            consistent_random_MRL_values, binsize=0.05, binLow=0, binHigh=1, errorbarType='poisson', density=True
        )
        # Add a (0, 0) point for spline fit to pass through origin
        consistent_random_MRL_bin_centers = np.concatenate([[0], consistent_random_MRL_bin_centers])
        consistent_random_MRL_bins = np.concatenate([[0], consistent_random_MRL_bins])
        consistent_random_MRL_errorbars = np.concatenate([[0], consistent_random_MRL_errorbars])
    else:
        consistent_random_MRL_bins, consistent_random_MRL_bin_centers, consistent_random_MRL_errorbars = _empty_histogram()

    if mrl_values_array is None:
        raise ValueError("mrl_values must be provided when gg_indices_inconsistent is provided.")
    if len(mrl_values_array) != 3 * len(list_gg.getAllGalaxyGroups()):
        raise ValueError("MRL_values length does not match 3 * number of groups in filtered_gt14.")

    consistent_MRL_values_masked = np.full_like(mrl_values_array, np.nan)
    inconsistent_MRL_values_masked = np.full_like(mrl_values_array, np.nan)

    for group_index in range(len(list_gg.getAllGalaxyGroups())):
        base = group_index * 3
        group_vals = mrl_values_array[base:base + 3]
        inconsistentProjections = gg_indices_inconsistent.get(group_index, [])

        for proj in projections_all:
            idx = proj_to_idx[proj]
            val = group_vals[idx]
            if not np.isfinite(val):
                continue
            if proj in inconsistentProjections:
                inconsistent_MRL_values_masked[base + idx] = val
            else:
                consistent_MRL_values_masked[base + idx] = val

    # Optional finite-only lists for histogramming
    consistent_MRL_values = consistent_MRL_values_masked[np.isfinite(consistent_MRL_values_masked)].tolist()
    inconsistent_MRL_values = inconsistent_MRL_values_masked[np.isfinite(inconsistent_MRL_values_masked)].tolist()

    consistent_fraction_above_percentile_masked, consistent_count_less_than_percentile_masked, consistent_total_count_masked, _ = GalaxyAnalysis.calculate_fraction_less_than_percentile(
        MRL_values=consistent_MRL_values_masked,
        random_MRL_values=random_MRL_values,
        percentile=percentConfidenceMRL,
    )
    inconsistent_fraction_above_percentile_masked, inconsistent_count_less_than_percentile_masked, inconsistent_total_count_masked, _ = GalaxyAnalysis.calculate_fraction_less_than_percentile(
        MRL_values=inconsistent_MRL_values_masked,
        random_MRL_values=random_MRL_values,
        percentile=percentConfidenceMRL,
    )

    consistent_bin_values_masked, consistent_bin_centers_masked, consistent_bin_errorbars_masked = ListGalaxyGroup.get_histogram_bins(
        consistent_MRL_values, bins=bin, errorbarType='poisson', density=True
    )
    inconsistent_bin_values_masked, inconsistent_bin_centers_masked, inconsistent_bin_errorbars_masked = ListGalaxyGroup.get_histogram_bins(
        inconsistent_MRL_values, bins=bin, errorbarType='poisson', density=True
    )

    if mrl_plotter is not None:
        mrl_plotter.scatter_plot(
            consistent_random_MRL_bin_centers,
            consistent_random_MRL_bins,
            ax=mrl_ax,
            label="Random MRL (consistent groups)",
            overlay_color=consistentColor,
            alpha=0,
            output_filename=None,
            grid=True,
            spline_curvature=True,
            spline_smoothing=0,
        )
        
        mrl_plotter.scatter_plot(
            inconsistent_random_MRL_bin_centers,
            inconsistent_random_MRL_bins,
            ax=mrl_ax,
            label="Random MRL (inconsistent groups)",
            overlay_color=inconsistentColor,
            alpha=0,
            output_filename=None,
            grid=True,
            spline_curvature=True,
            spline_smoothing=0,
        )

        mrl_plotter.scatter_plot(
            consistent_bin_centers_masked,
            consistent_bin_values_masked,
            errorBars=consistent_bin_errorbars_masked,
            ax=mrl_ax,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            label=f"Consistent projections (N={len(consistent_MRL_values)})",
            output_filename=None,
            grid=True,
        )
        mrl_plotter.scatter_plot(
            inconsistent_bin_centers_masked,
            inconsistent_bin_values_masked,
            errorBars=inconsistent_bin_errorbars_masked,
            ax=mrl_ax,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            label=f"Inconsistent projections (N={len(inconsistent_MRL_values)})",
            output_filename=None,
            grid=True,
        )
    else:
        mrl_ax.errorbar(
            consistent_bin_centers_masked,
            consistent_bin_values_masked,
            yerr=consistent_bin_errorbars_masked,
            label=f"Consistent projections (N={len(consistent_MRL_values)})",
            marker='o',
            # linestyle='-',
            linewidth=1,
            markersize=3,
            capsize=2,
        )
        mrl_ax.errorbar(
            inconsistent_bin_centers_masked,
            inconsistent_bin_values_masked,
            yerr=inconsistent_bin_errorbars_masked,
            label=f"Inconsistent projections (N={len(inconsistent_MRL_values)})",
            marker='o',
            # linestyle='-',
            linewidth=1,
            markersize=3,
            capsize=2,
        )
        mrl_ax.plot(
            consistent_random_MRL_bin_centers,
            consistent_random_MRL_bins,
            linestyle='--',
            color=consistentColor,
            linewidth=1,
            label='Random MRL (consistent groups)',
        )
        mrl_ax.plot(
            inconsistent_random_MRL_bin_centers,
            inconsistent_random_MRL_bins,
            linestyle='--',
            color=inconsistentColor,
            linewidth=1,
            label='Random MRL (inconsistent groups)',
        )
        mrl_ax.set_xlabel(xlabel)
        mrl_ax.set_ylabel(ylabel)
        mrl_ax.set_title(title)
        mrl_ax.grid(True, alpha=0.3)

        #if consistent and consistent are near zero at 0.6, trim xlim to zoom in on the region with data
        if (len(consistent_MRL_values) > 0 and len(inconsistent_MRL_values) > 0 and
            np.max(consistent_bin_centers_masked) > 0.6 and np.max(inconsistent_bin_centers_masked) > 0.6 and
            np.max(consistent_bin_values_masked[consistent_bin_centers_masked <= 0.6]) < 1e-1 and
            np.max(inconsistent_bin_values_masked[inconsistent_bin_centers_masked <= 0.6]) < 1e-1):
            mrl_ax.set_xlim(0, 0.6)

    # mrl_ax.annotate(
    #     f"Fraction Inconsistent MRL Projections: {inconsistent_fraction_above_percentile_masked:.4f} : {inconsistent_count_less_than_percentile_masked}/{len(inconsistent_MRL_values)}",
    #     xy=(0.5, 0.0),
    #     xycoords='axes fraction',
    #     xytext=(0, -14),
    #     textcoords='offset points',
    #     fontsize=8,
    #     ha='center',
    #     va='top',
    #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
    #     annotation_clip=False,
    # )
    # mrl_ax.annotate(
    #     f"Fraction Consistent MRL Projections: {consistent_fraction_above_percentile_masked:.4f} : {consistent_count_less_than_percentile_masked}/{len(consistent_MRL_values)}",
    #     xy=(0.5, 0.0),
    #     xycoords='axes fraction',
    #     xytext=(0, -32),
    #     textcoords='offset points',
    #     fontsize=8,
    #     ha='center',
    #     va='top',
    #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
    #     annotation_clip=False,
    # )
    mrl_ax.text(
        0.95,
        0.95,
        f"Fraction inconsistent pairs: {1-consistent_fraction_above_percentile_masked:.4f}",
        transform=mrl_ax.transAxes,
        fontsize=7,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    mrl_ax.legend(fontsize=8)

    return (polar_plotter, polar_fig, polar_ax), (mrl_plotter, mrl_fig, mrl_ax), gg_indices_inconsistent



def getConsistentInconsistentAssymetryPlots(
    listRedshiftGalaxyAnalysis,
    allSnapshotEnums,
    mass_bin_getter,
    mass_bin_label,
    percentageMRL=95,
    ax=None,
    fig=None,
    plotter=None,
    title=None,
    ):
    """Build High-vs-Non-High asymmetry plot on a new axis or a provided multipanel axis."""
    redshifts = []
    redshift_mass_bins: list[list[ListGalaxyGroup]] = []
    if percentageMRL < 1:
        print(f"Interpreting percentageMRL={percentageMRL} as a fraction and converting to percentage.")
        percentageMRL *= 100

    for snapshot_enum in allSnapshotEnums:
        snapshot_num = snapshot_enum[0]
        print(f"processing snapshot: {snapshot_enum}")
        galaxyAnalysisSnapshot = listRedshiftGalaxyAnalysis.getGalaxyAnalysisSnapshot(snapshot_num)
        mass_bin = mass_bin_getter(galaxyAnalysisSnapshot)
        if mass_bin.getNumGalaxyGroups() == 0:
            continue

        random_MRL_values = mass_bin.compute_MRL_random_distribution_curves_for_LGG(
            parallelize=False,
            num_samples=10000,
        )
        MRL_values = mass_bin.compute_probablity_distribution_of_MRL_directionality(parallelize=False)

        (fraction, count, total, gg_indices_inconsistent) = GalaxyAnalysis.calculate_fraction_less_than_percentile(
            MRL_values=MRL_values,
            random_MRL_values=random_MRL_values,
            percentile=percentageMRL,
        )

        high_galaxy_groups = []
        non_high_galaxy_groups = []
        for i, gg in enumerate(mass_bin.getAllGalaxyGroups()):
            if i not in gg_indices_inconsistent:
                non_high_galaxy_groups.append(gg)
                continue
            high_galaxy_groups.append(gg)

        high_list_galaxy_groups = ListGalaxyGroup(high_galaxy_groups)
        non_high_list_galaxy_groups = ListGalaxyGroup(non_high_galaxy_groups)

        redshifts.append(snapshot_enum[1])
        redshift_mass_bins.append([high_list_galaxy_groups, non_high_list_galaxy_groups])

    panel_title = title if title is not None else (
        f"P($\delta \phi$) Difference for Consistent vs. Inconsistent GG ({mass_bin_label})"
    )

    assymetry_plotter, assymetry_fig, assymetry_ax, assymetry_by_type_by_redshift = asymetryPlots(
        redshifts,
        listRedshiftGG=redshift_mass_bins,
        listGGLabels=[
            f'High GG (Inconsistent at {percentageMRL}% CL)',
            f'Non-High GG (Consistent at {percentageMRL}% CL)',
        ],
        plotRows=1,
        plotCols=1,
        ax=ax,
        fig=fig,
        plotter=plotter,
        title=panel_title,
    )

    return assymetry_plotter, assymetry_fig, assymetry_ax, assymetry_by_type_by_redshift




    