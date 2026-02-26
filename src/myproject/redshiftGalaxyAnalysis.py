#AP 2026

from myproject import GalaxyGroup, Subhalo, ListGalaxyGroup, AstroPlotter
from myproject.utilities.joinTime import JoinTime
import h5py as h5
import numpy as np
import os

class GalaxyAnalysis:
    def __init__(self, sim : str = 'TNG300-1', snapshot : int = 99, generalRewrite: bool = False, verbose : bool = False, generalErrorbar:str='poisson'):
        self.sim = sim
        self.snapshot = snapshot
        self.verbose = verbose
        self.generalRewrite = generalRewrite
        self.snapshot_dic = {99: (0, 'z0p0'), 91: (0.1, 'z0p1'), 84: (0.2, 'z0p2'), 78: (0.3, 'z0p3'), 72: (0.4, 'z0p4'), 67: (0.5, 'z0p5'), 59: (0.7, 'z0p7'), 50: (1.0, 'z1p0'), 40: (1.5, 'z1p5'), 33: (2.0, 'z2p0'), 25: (3.0, 'z3p0')}
        # possible_snapshots = list(snapshot_dic.keys())
        if self.sim == 'TNG300-1':
            self.L = 205*1e3 #kpc for TNG300
        else:
            self.L = 680*1e3 #kpc for TNGCluster
        self.halfbox = self.L/2
        self.setDircs()

        self.plotIdentifier = f'{sim}_{self.snapshot_dic[snapshot][1]}'
        
        self.loaded_list_of_galaxy_groups : ListGalaxyGroup = None

        self.generalErrorbar = generalErrorbar
        
        self.load_galaxy_groups()
        self.initializeMassSubgroups()

        
    def computeAllPlots(self):
        self.pairwisePolarDifferencePlot(self.filtered_gt14_list_of_galaxy_groups, self.scratchPlotDirc)
        self.meanResultantLengthPlot(self.filtered_gt14_list_of_galaxy_groups, self.scratchPlotDirc)
        self.redVsBluePairwisePlot(self.filtered_gt14_list_of_galaxy_groups, self.scratchPlotDirc)
        self.member150v50Plot(self.filtered_gt14_list_of_galaxy_groups, self.scratchPlotDirc)
        self.memberL35vG65Plot(self.filtered_gt14_list_of_galaxy_groups, self.scratchPlotDirc)
        # self.centralFoFDistanceOffsets(self.scratchPlotDirc)
        self.probabilityDistributionOf5MassGroups(self.scratchPlotDirc)
        # self.MRLDistributionPlots(self.scratchPlotDirc)

        print("Finished normal")
        self.HighMRLPlots(self.scratchPlotDirc)
        self.plot_satellite_number_distribution_for_all_mass_bins(self.scratchPlotDirc)

    def computeRedShiftPlots(self):
        self.plot_joining_redshift_for_all_mass_bins(self.scratchPlotDirc)
    
    def setDircs(self):
        self.scratchDataDirc = f'/scratch/poulin.al/lopsided/{self.sim}/{self.snapshot_dic[self.snapshot][1]}/data'
        
        self.scratchPlotDirc = f'/scratch/poulin.al/lopsided/{self.sim}/{self.snapshot_dic[self.snapshot][1]}/plots'
        self.localDataDirc = f'/Users/alexpoulin/Library/CloudStorage/OneDrive-NortheasternUniversity/TGB–Data'

    def load_galaxy_groups(self):
        data_file = self.scratchDataDirc + f'/galaxy_data_{self.sim}.hdf5'
        # data_file = localDataDirc + f'/galaxy_data_{sim}.hdf5'
        with h5.File(data_file, 'r') as f:
            self.loaded_list_of_galaxy_groups = ListGalaxyGroup.from_hdf5(f)
        print(f'Loaded galaxy data from {data_file}')
        # print(f"len filtered: {len(filtered_galaxy_groups)}")
        self.filtered_gt14_list_of_galaxy_groups = self.loaded_list_of_galaxy_groups.getFilterSubhalos(minGGMass=1e14)
        return self.loaded_list_of_galaxy_groups
    
    def load_galaxy_groups_for_snapshot_sim(self, snapshot:int, sim:str):
        scratchDataDirc = f'/scratch/poulin.al/lopsided/{sim}/{self.snapshot_dic[snapshot][1]}/data'
        data_file = scratchDataDirc + f'/galaxy_data_{sim}.hdf5'
        with h5.File(data_file, 'r') as f:
            list_of_galaxy_groups = ListGalaxyGroup.from_hdf5(f)
        print(f'Loaded galaxy data for snapshot {snapshot} from {data_file}')
        return list_of_galaxy_groups
        
    def setGeneralRewrite(self, rewrite : bool):
        self.generalRewrite = rewrite
    
    def setSnapshot(self, newsnapshot:int):
        self.snapshot = newsnapshot
        self.setDircs()
        self.load_galaxy_groups()
        self.initializeMassSubgroups()

    def initializeMassSubgroups(self, list_of_galaxy_groups : ListGalaxyGroup = None):
        if list_of_galaxy_groups is not None:
            self.loaded_list_of_galaxy_groups = list_of_galaxy_groups
        #filter to groups with stellar mass 10^{13} < $M_{{200}}$ < 10^{13.5} Msun
        self.filtered_gt13_ls13p5_list_of_galaxy_groups = self.loaded_list_of_galaxy_groups.getFilterSubhalos(maxGGMass=5e13, minGGMass=1e13)

        #filter to groups with stellar mass 10^{13.5} < $M_{{200}}$ < 10^{14} Msun
        self.filtered_gt13p5_ls14_list_of_galaxy_groups = self.loaded_list_of_galaxy_groups.getFilterSubhalos(maxGGMass=1e14, minGGMass=5e13)

        #filter to groups with stellar mass 10^{14} < $M_{{200}}$ < 10^{14.5} Msun
        self.filtered_gt14_ls14p5_list_of_galaxy_groups = self.loaded_list_of_galaxy_groups.getFilterSubhalos(maxGGMass=5e14, minGGMass=1e14)

        #filter to groups with stellar mass 10^{14.5} < $M_{{200}}$ < 10^{15} Msun
        self.filtered_gt14p5_ls15_list_of_galaxy_groups = self.loaded_list_of_galaxy_groups.getFilterSubhalos(maxGGMass=1e15, minGGMass=5e14)

        #filter to groups with stellar mass $M_{{200}}$ > 10^{15} Msun
        self.filtered_gt15_list_of_galaxy_groups = self.loaded_list_of_galaxy_groups.getFilterSubhalos(minGGMass=1e15)
        
        return self.filtered_gt13_ls13p5_list_of_galaxy_groups, self.filtered_gt13p5_ls14_list_of_galaxy_groups, self.filtered_gt14_ls14p5_list_of_galaxy_groups, self.filtered_gt14p5_ls15_list_of_galaxy_groups, self.filtered_gt15_list_of_galaxy_groups
        
    def pairwisePolarDifferencePlot(self, list_of_galaxy_group : ListGalaxyGroup, plot_dirc : str = None):
        list_pairwise_polar_differences = list_of_galaxy_group.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/pairwise_polar_{self.plotIdentifier}', rewrite=self.generalRewrite)
        
        # polar_bin_centers, pairwise_polar_differences = list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=localDataDirc)
        pairwise_polar_differences_binned, polar_bin_centers, pairwise_polar_errorbars = ListGalaxyGroup.get_histogram_bins(list_pairwise_polar_differences, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
        print('Computed pairwise polar differences between satellite galaxies.') if self.verbose else None
                
        prob_polar_plotter = AstroPlotter()
        prob_polar_plotter.scatter_plot(
            polar_bin_centers, 
            pairwise_polar_differences_binned,
            errorBars = pairwise_polar_errorbars,
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Probability Pairwise Polar Difference Distribution for {self.plotIdentifier}',
            # ylim = (0, 0.01),
            output_filename=self.scratchPlotDirc + f'/pairwise_polar/pairwise_polar_difference_{self.plotIdentifier}.png' if plot_dirc is None else plot_dirc + f'/pairwise_polar_difference_{self.plotIdentifier}.png',
            grid=True,
        )
        
        # save bin centers and probabilities, errorbars to text file
        output_data_file = self.scratchDataDirc + f'/pairwise_polar/pairwise_polar_difference_{self.plotIdentifier}.txt'
        np.savetxt(output_data_file, np.column_stack((polar_bin_centers, pairwise_polar_differences_binned, pairwise_polar_errorbars)), header='Pairwise Polar Difference (degrees)    Probability Density    Errorbars')
        
        #save raw pairwise polar differences to text file
        output_data_file = self.scratchDataDirc + f'/pairwise_polar/pairwise_polar_difference_RAW_{self.plotIdentifier}.txt'
        np.savetxt(output_data_file, np.column_stack((list_pairwise_polar_differences)), header='Pairwise Polar Difference (degrees)')
        
        return prob_polar_plotter, polar_bin_centers, pairwise_polar_differences_binned, pairwise_polar_errorbars
        
    def meanResultantLengthPlot(self, list_of_galaxy_group : ListGalaxyGroup, plot_dirc : str = None):
        MRL_values = list_of_galaxy_group.compute_probablity_distribution_of_MRL_directionality(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/MRL_directionality_{self.plotIdentifier}', rewrite=self.generalRewrite)
        
        MRL_directionality, MRL_bin_centers, MRL_errorbars = list_of_galaxy_group.get_histogram_bins(MRL_values, binsize=0.05, binLow=0, binHigh=1, errorbarType='poisson')
        
        prob_MRL_plotter = AstroPlotter()
        prob_MRL_plotter.scatter_plot(
            MRL_bin_centers, 
            MRL_directionality,
            errorBars = MRL_errorbars,
            xlabel='MRL Directionality',
            ylabel='Probability Density',
            title=f'Probability MRL Directionality Distribution for {self.plotIdentifier}',
            # ylim=(0,4)
            output_filename=self.scratchPlotDirc + f'/MRL/MRL_directionality_{self.plotIdentifier}.png' if plot_dirc is None else plot_dirc + f'/MRL_directionality_{self.plotIdentifier}.png',
            grid=True
        )
        
        #save bin centers and probabilities to text file
        output_data_file_MRL = self.scratchDataDirc + f'/MRL_directionality/MRL_directionality_{self.plotIdentifier}.txt'
        # print(f"data to be saved: {np.column_stack((MRL_bin_centers, MRL_directionality))}")
        np.savetxt(output_data_file_MRL, np.column_stack((MRL_bin_centers, MRL_directionality, MRL_errorbars)), header='MRL Directionality    Probability Density    Errorbars')
        
        #save bin MRL values to text file
        output_data_file_MRL = self.scratchDataDirc + f'/MRL_directionality/MRL_directionality_RAW_{self.plotIdentifier}.txt'
        # print(f"data to be saved: {np.vstack(list_of_galaxy_group.MRL_values)}")
        np.savetxt(output_data_file_MRL, np.vstack(list_of_galaxy_group.MRL_values), header='MRL Directionality')
    
    def redVsBlueDistributionPlot(self, list_of_galaxy_groups : ListGalaxyGroup, plot_dirc : str = None):
        #plot the g-r color distribution for red and blue galaxies in the same plot
        #fit a Gaussian to the g-r color distribution for red and blue galaxies and find the intersection point of the two Gaussians to use as a threshold for separating red and blue galaxies
        gMr_values = []
        for galaxy_group in list_of_galaxy_groups.getAllGalaxyGroups():
            for subhalo in galaxy_group.getSubhalos():
                g_mag = subhalo.getGbandMagnitude()
                r_mag = subhalo.getRbandMagnitude()
                if np.isnan(g_mag) or np.isnan(r_mag):
                    print("WARNING... np.nan")
                    continue  # Skip if magnitudes are not available
                g_r_color = g_mag - r_mag
                gMr_values.append(g_r_color)
                
        gMr_values = np.array(gMr_values)
        from scipy.stats import norm
        # Fit Gaussian to the g-r color distribution        
        mu, std = norm.fit(gMr_values)
        # Generate x values for the Gaussian curve
        x = np.linspace(min(gMr_values), max(gMr_values), 1000)
        # Calculate the Gaussian curve values        
        p = norm.pdf(x, mu, std)
        # Find the intersection point of the Gaussian curve with a horizontal line at the minimum between the two peaks to determine the threshold for separating red and blue galaxies
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(p)
        intersection_point_valid=False
        if len(peaks) < 2:
            print("Warning: Less than 2 peaks found in the g-r color distribution, cannot determine intersection point for red vs blue separation.")
            intersection_point = 0.65  # Default to 0.65 if we cannot find a clear separation
        else:
            min_between_peaks = np.argmin(p[peaks[0]:peaks[1]]) + peaks[0]
            intersection_point = x[min_between_peaks]
            print(f"Determined intersection point for red vs blue separation: {intersection_point:.2f}")
            intersection_point_valid=True
        # Plot the g-r color distribution and the Gaussian fit
        color_plotter = AstroPlotter()
        color_fig, color_ax = color_plotter.create_figure()
        color_plotter.histogram(
            gMr_values,
            bins=50,
            density=True,
            alpha=0.6,
            color='gray',
            ax=color_ax
        )
        color_plotter.plot(
            x,
            p,
            color='black',
            ax=color_ax,
            label=f'Gaussian Fit (μ={mu:.2f}, σ={std:.2f})'
        )
        color_plotter.axvline(
            intersection_point,
            color='red',
            linestyle='--',
            ax=color_ax,
            label=f'Intersection Point = {intersection_point:.2f}'
        )
        color_plotter.set_labels(
            xlabel='g-r Color',
            ylabel='Density',
            title=f'g-r Color Distribution with Gaussian Fit for {self.plotIdentifier}'
        )
        color_plotter.add_legend(ax=color_ax)
        #add a text box in the plot with the mean and standard deviation of the g-r color distribution and the intersection point
        color_plotter.add_text_box(
            color_ax, 
            f"Mean (μ) = {mu:.2f}\nStandard Deviation (σ) = {std:.2f}\nIntersection Point = {intersection_point:.2f}", 
            loc='bottom center'
        )
        color_plotter.save(
            self.scratchPlotDirc + f'/color_distribution/color_distribution_{self.plotIdentifier}.png' if plot_dirc is None else plot_dirc + f'/color_distribution_{self.plotIdentifier}.png'
        )
        return intersection_point, intersection_point_valid
    
    def redVsBluePairwisePlot(self, list_of_galaxy_groups : ListGalaxyGroup = None, plot_dirc : str = None):
        intersectionPoint, intersectionPointValid = self.redVsBlueDistributionPlot(list_of_galaxy_groups=list_of_galaxy_groups, plot_dirc=plot_dirc)
        
        filtered_red_list_of_galaxy_groups = list_of_galaxy_groups.getFilterSubhalos(redGalaxies=True, redBluePoint=intersectionPoint)
        print(f'Number of galaxy groups with only red satellites: {filtered_red_list_of_galaxy_groups.getRangeOfNumSubhalos()}')

        filtered_blue_list_of_galaxy_groups = list_of_galaxy_groups.getFilterSubhalos(blueGalaxies=True, redBluePoint=intersectionPoint)
        print(f'Number of galaxy groups with only blue satellites: {filtered_blue_list_of_galaxy_groups.getRangeOfNumSubhalos()}')

        list_pairwise_polar_differences_red = filtered_red_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/pairwise_polar_color/pairwise_polar_red_{self.plotIdentifier}', rewrite=self.generalRewrite)
        list_pairwise_polar_differences_blue = filtered_blue_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/pairwise_polar_color/pairwise_polar_blue_{self.plotIdentifier}', rewrite=self.generalRewrite)
        
        pairwise_polar_differences_red, polar_bin_centers_red, pairwise_polar_red_errorbars = ListGalaxyGroup.get_histogram_bins(list_pairwise_polar_differences_red, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
        pairwise_polar_differences_blue, polar_bin_centers_blue, pairwise_polar_blue_errorbars = ListGalaxyGroup.get_histogram_bins(list_pairwise_polar_differences_blue, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)

        print('Computed pairwise polar differences between red and blue satellite galaxies.')
        prob_polar_red_blue_plotter = AstroPlotter()
        prob_polar_red_blue_fig, prob_polar_red_blue_ax = prob_polar_red_blue_plotter.create_figure()
        prob_polar_red_blue_plotter.scatter_plot(
            polar_bin_centers_red, 
            pairwise_polar_differences_red,
            ax=prob_polar_red_blue_ax, # Use the same axis for overlay
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Probability Pairwise Polar Difference Distribution for Red and Blue Satellites in {self.plotIdentifier}',
            # ylim = (0, 0.01),
            c=['red'],
            errorBars = pairwise_polar_red_errorbars,
            label = "Red Galaxies (g-r) ≥ 0.65",
            output_filename=None,  # Disable saving for combined plot
            grid=True,
        )
        prob_polar_red_blue_plotter.scatter_plot(
            polar_bin_centers_blue, 
            pairwise_polar_differences_blue,
            ax=prob_polar_red_blue_ax,  # Use the same axis for overlay
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Probability Pairwise Polar Difference Distribution for Red and Blue Satellites in {self.plotIdentifier}',
            # ylim = (0, 0.01),
            c=['blue'],
            errorBars = pairwise_polar_blue_errorbars,
            label = "Blue Galaxies (g-r) < 0.65",
            include_legend=True,
            output_filename=self.scratchPlotDirc + f'/pairwise_polar/pairwise_polar_difference_red_blue_{self.plotIdentifier}.png',
            grid=True,
        )
        
        # save as txt file
        output_data_file = self.scratchDataDirc + f'/pairwise_polar_color/pairwise_polar_color_difference_{self.plotIdentifier}.txt'
        arrays = [polar_bin_centers_red, pairwise_polar_differences_red, pairwise_polar_red_errorbars, pairwise_polar_differences_blue, pairwise_polar_blue_errorbars]
        max_len = max(len(arr) for arr in arrays)
        arrays_padded = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in arrays]
        if any(len(arr) != max_len for arr in arrays):
            print(f"Warning: Arrays have different lengths, padding to {max_len} with NaN for saving pairwise polar color difference.")
        print(f"data to be saved: {np.column_stack(arrays_padded)}")
        np.savetxt(output_data_file, np.column_stack(arrays_padded), header='Pairwise Polar Difference (degrees)    Probability Density (Red)    Error Bar (Red)    Probability Density (Blue)    Error Bar (Blue)')
        #save raw pairwise polar differences to text file
        output_data_file = self.scratchDataDirc + f'/pairwise_polar_color/pairwise_polar_color_difference_RAW_{self.plotIdentifier}.txt'
        raw_arrays = [list_pairwise_polar_differences_red, list_pairwise_polar_differences_blue]
        max_raw_len = max(len(arr) for arr in raw_arrays)
        raw_arrays_padded = [np.pad(arr, (0, max_raw_len - len(arr)), constant_values=np.nan) for arr in raw_arrays]
        if any(len(arr) != max_raw_len for arr in raw_arrays):
            print(f"Warning: Raw arrays have different lengths, padding to {max_raw_len} with NaN for saving raw pairwise polar color difference.")
        np.savetxt(output_data_file, np.column_stack(raw_arrays_padded), header='Pairwise Polar Difference (degrees) (Red)    Pairwise Polar Difference (degrees) (Blue)')
        
    def member150v50Plot(self, list_of_galaxy_groups : ListGalaxyGroup = None, plot_dirc : str = None):
        filtered_GT150_list_of_galaxy_groups = list_of_galaxy_groups.getFilterSubhalos(minNumGalaxies=150)
        print(f'Number of galaxy groups with more than 150 satellites: {filtered_GT150_list_of_galaxy_groups.getNumGalaxyGroups()}')
        filtered_LT50_list_of_galaxy_groups = list_of_galaxy_groups.getFilterSubhalos(maxNumGalaxies=50)
        print(f'Number of galaxy groups with less than 50 satellites: {filtered_LT50_list_of_galaxy_groups.getNumGalaxyGroups()}')

        list_pairwise_polar_differences_GT150 = filtered_GT150_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/pairwise_polar_memberNum/pairwise_polar_GT150_{self.plotIdentifier}', rewrite=self.generalRewrite)
        list_pairwise_polar_differences_LT50 = filtered_LT50_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/pairwise_polar_memberNum/pairwise_polar_LT50_{self.plotIdentifier}', rewrite=self.generalRewrite)
        
        print('Computed pairwise polar differences for galaxy groups with >150 and <50 satellites.')
        pairwise_polar_differences_GT150, polar_bin_centers_GT150, pairwise_polar_GT150_errorbars = ListGalaxyGroup.get_histogram_bins(list_pairwise_polar_differences_GT150, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
        pairwise_polar_differences_LT50, polar_bin_centers_LT50, pairwise_polar_LT50_errorbars = ListGalaxyGroup.get_histogram_bins(list_pairwise_polar_differences_LT50, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
        
        prob_polar_GT150_LT50_plotter = AstroPlotter()
        prob_polar_GT150_LT50_fig, prob_polar_GT150_LT50_ax = prob_polar_GT150_LT50_plotter.create_figure()
        prob_polar_GT150_LT50_plotter.scatter_plot(
            polar_bin_centers_GT150, 
            pairwise_polar_differences_GT150,
            errorBars = pairwise_polar_GT150_errorbars,
            ax=prob_polar_GT150_LT50_ax, # Use the same axis for overlay
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Probability Pairwise Polar Difference Distribution for >150 and <50 Satellites in {self.plotIdentifier}',
            # ylim = (0, 0.01),
            label="More than 150 members",
            output_filename=None,  # Disable saving for combined plot
            grid=True,
        )
        prob_polar_GT150_LT50_plotter.scatter_plot(
            polar_bin_centers_LT50, 
            pairwise_polar_differences_LT50,
            errorBars = pairwise_polar_LT50_errorbars,
            ax=prob_polar_GT150_LT50_ax,  # Use the same axis for overlay
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Probability Pairwise Polar Difference Distribution for >150 and <50 Satellites in {self.plotIdentifier}',
            # ylim = (0, 0.01),
            label="Less than 50 members",
            include_legend=True,
            output_filename=self.scratchPlotDirc + f'/pairwise_polar/pairwise_polar_difference_GT150_LT50_{self.plotIdentifier}.png',
            grid=True,
        )
        
        # save as text file
        output_data_file = self.scratchDataDirc + f'/pairwise_polar_memberNum/pairwise_polar_memberNum_difference_{self.plotIdentifier}.txt'
        arrays = [polar_bin_centers_LT50, pairwise_polar_differences_LT50, pairwise_polar_LT50_errorbars, pairwise_polar_differences_GT150, pairwise_polar_GT150_errorbars]
        max_len = max(len(arr) for arr in arrays)
        arrays_padded = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in arrays]
        np.savetxt(output_data_file, np.column_stack(arrays_padded), header='Pairwise Polar Difference (degrees)    Probability Density (<50 Satellites)    Error Bar (<50 Satellites)    Probability Density (>150 Satellites)    Error Bar (>150 Satellites)')
        #save raw pairwise polar differences to text file
        output_data_file = self.scratchDataDirc + f'/pairwise_polar_memberNum/pairwise_polar_memberNum_difference_RAW_{self.plotIdentifier}.txt'
        raw_arrays = [list_pairwise_polar_differences_LT50, list_pairwise_polar_differences_GT150]
        max_raw_len = max(len(arr) for arr in raw_arrays)
        raw_arrays_padded = [np.pad(arr, (0, max_raw_len - len(arr)), constant_values=np.nan) for arr in raw_arrays]
        np.savetxt(output_data_file, np.column_stack(raw_arrays_padded), header='Pairwise Polar Difference (degrees) (<50 Satellites)    Pairwise Polar Difference (degrees) (>150 Satellites)')

    def memberL35vG65Plot(self, list_of_galaxy_groups : ListGalaxyGroup = None, plot_dirc : str = None):
        filtered_LT35R200_list_of_galaxy_groups = list_of_galaxy_groups.getFilterSubhalos(withinXPercentR200=[0,0.35])
        print(f'Number of galaxy groups with satellites within 35% R200: {filtered_LT35R200_list_of_galaxy_groups.getNumGalaxyGroups()}')
        filtered_GT65R200_list_of_galaxy_groups = list_of_galaxy_groups.getFilterSubhalos(withinXPercentR200=[0.65,1.00])
        print(f'Number of galaxy groups with satellites within 65-100% R200: {filtered_GT65R200_list_of_galaxy_groups.getNumGalaxyGroups()}')

        list_pairwise_polar_differences_LT35R200 = filtered_LT35R200_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/pairwise_polar_radius/pairwise_polar_LT35R200_{self.plotIdentifier}', rewrite=self.generalRewrite)
        list_pairwise_polar_differences_GT65R200 = filtered_GT65R200_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/pairwise_polar_radius/pairwise_polar_GT65R200_{self.plotIdentifier}', rewrite=self.generalRewrite)
        
        pairwise_polar_differences_LT35R200, polar_bin_centers_LT35R200, pairwise_polar_LT35R200_errorbars = filtered_LT35R200_list_of_galaxy_groups.get_histogram_bins(list_pairwise_polar_differences_LT35R200, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
        pairwise_polar_differences_GT65R200, polar_bin_centers_GT65R200, pairwise_polar_GT65R200_errorbars = filtered_GT65R200_list_of_galaxy_groups.get_histogram_bins(list_pairwise_polar_differences_GT65R200, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
        
        prob_polar_LT35R200_GT65R200_plotter = AstroPlotter()
        prob_polar_LT35R200_GT65R200_fig, prob_polar_LT35R200_GT65R200_ax = prob_polar_LT35R200_GT65R200_plotter.create_figure()
        prob_polar_LT35R200_GT65R200_plotter.scatter_plot(
            polar_bin_centers_LT35R200, 
            pairwise_polar_differences_LT35R200,
            errorBars = pairwise_polar_LT35R200_errorbars,
            ax=prob_polar_LT35R200_GT65R200_ax, # Use the same axis for overlay
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Probability Pairwise Polar Difference Distribution for 65%<r<100% R200 and 0%<r<35% R200 in {self.plotIdentifier}',
            # ylim = (0, 0.01),
            label="0<r<0.35 R200",
            output_filename=None,  # Disable saving for combined plot
            grid=True,
        )
        prob_polar_LT35R200_GT65R200_plotter.scatter_plot(
            polar_bin_centers_GT65R200, 
            pairwise_polar_differences_GT65R200,
            errorBars = pairwise_polar_GT65R200_errorbars,
            ax=prob_polar_LT35R200_GT65R200_ax,  # Use the same axis for overlay
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Probability Pairwise Polar Difference Distribution for \n 65%<r<100% R200 and 0%<r<35% R200 in {self.plotIdentifier}',
            # ylim = (0, 0.01),
            label="0.65<r<1.00 R200",
            include_legend=True,
            output_filename=self.scratchPlotDirc + f'/pairwise_polar/pairwise_polar_difference_LT35R200_GT65R200_{self.plotIdentifier}.png',
            grid=True,
        )
        
        #save bin centers and probabilities and errorbars to text file
        output_data_file = self.scratchDataDirc + f'/pairwise_polar_radius/pairwise_polar_radius_difference_{self.plotIdentifier}.txt'
        arrays = [polar_bin_centers_LT35R200, pairwise_polar_differences_LT35R200, pairwise_polar_LT35R200_errorbars, pairwise_polar_differences_GT65R200, pairwise_polar_GT65R200_errorbars]
        max_len = max(len(arr) for arr in arrays)
        arrays_padded = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in arrays]
        np.savetxt(output_data_file, np.column_stack(arrays_padded), header='Pairwise Polar Difference (degrees)    Probability Density (<35% R200)    Error Bar (<35% R200)    Probability Density (65-100% R200)    Error Bar (65-100% R200)')
        #save raw pairwise polar differences to text file
        output_data_file = self.scratchDataDirc + f'/pairwise_polar_radius/pairwise_polar_radius_difference_RAW_{self.plotIdentifier}.txt'
        raw_arrays = [list_pairwise_polar_differences_LT35R200, list_pairwise_polar_differences_GT65R200]
        max_raw_len = max(len(arr) for arr in raw_arrays)
        raw_arrays_padded = [np.pad(arr, (0, max_raw_len - len(arr)), constant_values=np.nan) for arr in raw_arrays]
        np.savetxt(output_data_file, np.column_stack(raw_arrays_padded), header='Pairwise Polar Difference (degrees) (<35% R200)    Pairwise Polar Difference (degrees) (65-100% R200)')
    
    def getCentral_FoF_distanceOffset(self, list_of_galaxy_groups : ListGalaxyGroup):
        distanceOffsets = []
        central_positions = []
        FoF_positions = []
        for gg in list_of_galaxy_groups.listGalaxyGroups:
            central_subhalo = gg.getCentralSubhalo()
            fof_center = gg.getPos()
            central_positions.append(central_subhalo.getPosition())
            FoF_positions.append(fof_center)
            
            distanceOffsets.append(np.linalg.norm(central_subhalo.getPosition() - fof_center))
        return np.array(distanceOffsets), np.array(central_positions), np.array(FoF_positions)
        
    def centralFoFDistanceOffsets(self, plot_dirc : str = None):
        #save txt file if most massive != most central
        #with lines:
        # 1: x_cm of the halo
        # 2: y_cm of the halo
        # 3: z_cm of the halo
        # 4: offset of the "most central galaxy" from the halo center of mass
        # 5: ratio of the total masses of the most central and most massive galaxoes
        # 6: offset of the most massive galaxy from the halo center of mass
        with open(self.scratchDataDirc + f'/most_massive_vs_most_central/most_massive_vs_most_central_{self.plotIdentifier}.txt', 'w') as f:
            f.write('x_cm    y_cm    z_cm    offset_most_central    mass_ratio    offset_most_massive most_massive_r_ratio_r200\n')
            
        for galaxy_group in self.loaded_list_of_galaxy_groups.getAllGalaxyGroups():
            if galaxy_group.getMostMassiveSubhalo().getIdx() != galaxy_group.getMostCentralSubhalo().getIdx():
                output_data_file = self.scratchDataDirc + f'/most_massive_vs_most_central/most_massive_vs_most_central_{self.plotIdentifier}.txt'
                with open(output_data_file, 'a') as f:
                    x_cm, y_cm, z_cm = galaxy_group.getPos()
                    most_central_pos = galaxy_group.getMostCentralSubhalo().getPosition()
                    most_massive_pos = galaxy_group.getMostMassiveSubhalo().getPosition()
                    offset_most_central = np.linalg.norm(most_central_pos - galaxy_group.getPos())
                    offset_most_massive = np.linalg.norm(most_massive_pos - galaxy_group.getPos())
                    mass_ratio = galaxy_group.getMostCentralSubhalo().getStellarMass() / galaxy_group.getMostMassiveSubhalo().getStellarMass()
                    most_massive_r_ratio_r200 = np.linalg.norm(galaxy_group.getMostMassiveSubhalo().getPosition()) / galaxy_group.getRCrit200()
                    f.write(f"{x_cm} {y_cm} {z_cm} {offset_most_central} {mass_ratio} {offset_most_massive} {most_massive_r_ratio_r200} \n")
        
        #plot histogram of distance offsets between central subhalo and FoF center for all galaxy groups, and then split by mass bins as well      
        #filter to groups with stellar mass < 10^{14} Msun
        filtered_gt13_ls14_list_of_galaxy_groups = self.loaded_list_of_galaxy_groups.getFilterSubhalos(maxGGMass=1e14)
        
        distances_gt13_ls14 = self.getCentral_FoF_distanceOffset(filtered_gt13_ls14_list_of_galaxy_groups)[0]
        print(distances_gt13_ls14)
        distances_gt14 = self.getCentral_FoF_distanceOffset(self.filtered_gt14_list_of_galaxy_groups)[0]
        
        central_FoF_Plotter = AstroPlotter()
        central_FoF_fig, central_FoF_ax = central_FoF_Plotter.create_figure()
        central_FoF_Plotter.histogram(
            distances_gt13_ls14,
            bins=60, #np.logspace(np.log10(0.1), np.log10(5000), 60),
            ax = central_FoF_ax,  # Use the same axis for overlay
            xlabel='Distance between Central Subhalo and Galaxy Group Position (kpc)',
            ylabel='Density of Galaxy Groups',
            title=f'Distance between Central Subhalo and Galaxy Group Position for {self.plotIdentifier} ($M_{{200}}$<1e14 Msun)',
            label='1e13<$M_{{200}}$<1e14 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=self.scratchPlotDirc + f'/central_disparities/central_FoF_distance_{self.plotIdentifier}_lt14.png' if plot_dirc is None else plot_dirc + f'/central_disparities/central_FoF_distance_{self.plotIdentifier}_lt14.png',
            percentage=True,
            grid=True,
        )
        central_FoF_Plotter.histogram(
            distances_gt14,
            bins=60, #np.logspace(np.log10(0.1), np.log10(5000), 60),
            ax = central_FoF_ax,  # Use the same axis for overlay
            xlabel='Distance between Central Subhalo and Galaxy Group Position (kpc)',
            ylabel='Density of Galaxy Groups',
            title=f'Distance between Central Subhalo and Galaxy Group Position for {self.plotIdentifier}',
            label='$M_{{200}}$>1e14 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=self.scratchPlotDirc + f'/central_disparities/central_FoF_distance_{self.plotIdentifier}_gt14.png' if plot_dirc is None else plot_dirc + f'/central_disparities/central_FoF_distance_{self.plotIdentifier}_gt14.png',
            percentage=True,
            grid=True,
        )
        
        #make pairwise polar difference plots for these two mass bins as well
        list_pairwise_polar_differences_13_14 = filtered_gt13_ls14_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False)
        list_pairwise_polar_differences_gt14 = self.filtered_gt14_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False)
        
        pairwise_polar_differences_13_14, polar_bin_centers_13_14, pairwise_polar_13_14_errorbars = ListGalaxyGroup.get_histogram_bins(list_pairwise_polar_differences_13_14, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
        
        pairwise_polar_differences_gt14, polar_bin_centers_gt14, pairwise_polar_gt14_errorbars = ListGalaxyGroup.get_histogram_bins(list_pairwise_polar_differences_gt14, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
        
        prob_polar_gt14_lt14_plotter = AstroPlotter()
        prob_polar_gt14_lt14_fig, prob_polar_gt14_lt14_ax = prob_polar_gt14_lt14_plotter.create_figure()
        prob_polar_gt14_lt14_plotter.scatter_plot(
            polar_bin_centers_13_14, 
            pairwise_polar_differences_13_14,
            errorBars = pairwise_polar_13_14_errorbars,
            ax=prob_polar_gt14_lt14_ax, # Use the same axis for overlay
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Probability Pairwise Polar Difference Distribution for 1e13<$M_{{200}}$<1e14 and $M_{{200}}$>1e14 Msun in {self.plotIdentifier}',
            # ylim = (0, 0.01),
            label="1e13<$M_{{200}}$<1e14 Msun",
            output_filename=None,  # Disable saving for combined plot
            grid=True,
        )
        prob_polar_gt14_lt14_plotter.scatter_plot(
            polar_bin_centers_gt14, 
            pairwise_polar_differences_gt14,
            errorBars = pairwise_polar_gt14_errorbars,
            ax=prob_polar_gt14_lt14_ax,  # Use the same axis for        overlay
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Probability Pairwise Polar Difference Distribution for 1e13<$M_{{200}}$<1e14 and $M_{{200}}$>1e14 Msun in {self.plotIdentifier}',
            # ylim = (0, 0.01),
            label="$M_{{200}}$>1e14 Msun",
            include_legend=True,
            output_filename=self.scratchPlotDirc + f'/pairwise_polar/pairwise_polar_difference_gt14_lt14_{self.plotIdentifier}.png' if plot_dirc is None else plot_dirc + f'/pairwise_polar/pairwise_polar_difference_gt14_lt14_{self.plotIdentifier}.png',
            grid=True,
        )
        
    def probabilityDistributionOf5MassGroups(self, plot_dirc : str = None):
        #make pairwise polar difference plots for 5 mass bins as well
        
        print(f'Number of galaxy groups 13-13.5 loaded: {self.filtered_gt13_ls13p5_list_of_galaxy_groups.getNumGalaxyGroups()}')
        print(f'Number of galaxy groups 13.5-14 loaded: {self.filtered_gt13p5_ls14_list_of_galaxy_groups.getNumGalaxyGroups()}')
        print(f'Number of galaxy groups 14-14.5 loaded: {self.filtered_gt14_ls14p5_list_of_galaxy_groups.getNumGalaxyGroups()}')
        print(f'Number of galaxy groups 14.5-15 loaded: {self.filtered_gt14p5_ls15_list_of_galaxy_groups.getNumGalaxyGroups()}')
        print(f'Number of galaxy groups >15 loaded: {self.filtered_gt15_list_of_galaxy_groups.getNumGalaxyGroups()}')

        print(f"total number of galaxy groups: {self.filtered_gt13_ls13p5_list_of_galaxy_groups.getNumGalaxyGroups() + self.filtered_gt13p5_ls14_list_of_galaxy_groups.getNumGalaxyGroups() + self.filtered_gt14_ls14p5_list_of_galaxy_groups.getNumGalaxyGroups() + self.filtered_gt14p5_ls15_list_of_galaxy_groups.getNumGalaxyGroups() + self.filtered_gt15_list_of_galaxy_groups.getNumGalaxyGroups()}")

        # #print the num of galaxies in each group where the central is not the most massive
        # num_mostMassiveNotCentral = 0
        # for gg in self.filtered_gt13_ls13p5_list_of_galaxy_groups.listGalaxyGroups:
        #     if gg.getMostMassiveSubhalo().getIdx() != gg.getMostCentralSubhalo().getIdx():
        #         # print(f"GG ID: {gg.getGroupID()}, Num of galaxies: {gg.getNumSubhalos()}")
        #         num_mostMassiveNotCentral +=1
        #     else:
        #         print(gg.getMostMassiveSubhalo().getPosition(), gg.getMostCentralSubhalo().getPosition())
        # print(f"Number of groups with most massive not central in 13-13.5 bin: {num_mostMassiveNotCentral}")
        # num_mostMassiveNotCentral = 0
        # for gg in self.filtered_gt13p5_ls14_list_of_galaxy_groups.listGalaxyGroups:
        #     if gg.getMostMassiveSubhalo().getIdx() != gg.getMostCentralSubhalo().getIdx():
        #         # print(f"GG ID: {gg.getGroupID()}, Num of galaxies: {gg.getNumSubhalos()}")
        #         num_mostMassiveNotCentral +=1
        # print(f"Number of groups with most massive not central in 13.5-14 bin: {num_mostMassiveNotCentral}")
        # num_mostMassiveNotCentral = 0
        # for gg in self.filtered_gt14_ls14p5_list_of_galaxy_groups.listGalaxyGroups:
        #     if gg.getMostMassiveSubhalo().getIdx() != gg.getMostCentralSubhalo().getIdx():
        #         # print(f"GG ID: {gg.getGroupID()}, Num of galaxies: {gg.getNumSubhalos()}")
        #         num_mostMassiveNotCentral +=1
        # print(f"Number of groups with most massive not central in 14-14.5 bin: {num_mostMassiveNotCentral}")
        # num_mostMassiveNotCentral = 0
        # for gg in self.filtered_gt14p5_ls15_list_of_galaxy_groups.listGalaxyGroups:
        #     if gg.getMostMassiveSubhalo().getIdx() != gg.getMostCentralSubhalo().getIdx():
        #         # print(f"GG ID: {gg.getGroupID()}, Num of galaxies: {gg.getNumSubhalos()}")
        #         num_mostMassiveNotCentral +=1
        # print(f"Number of groups with most massive not central in 14.5-15 bin: {num_mostMassiveNotCentral}")
        # num_mostMassiveNotCentral = 0
        # for gg in self.filtered_gt15_list_of_galaxy_groups.listGalaxyGroups:
        #     if gg.getMostMassiveSubhalo().getIdx() != gg.getMostCentralSubhalo().getIdx():
        #         # print(f"GG ID: {gg.getGroupID()}, Num of galaxies: {gg.getNumSubhalos()}")
        #         num_mostMassiveNotCentral +=1
        # print(f"Number of groups with most massive not central in >15 bin: {num_mostMassiveNotCentral}")
    
        #plot the mass ratio distribution (most central / most massive) for these mass bins
        prob_mass_ratio_plotter = AstroPlotter()
        prob_mass_ratio_fig, prob_mass_ratio_ax = prob_mass_ratio_plotter.create_figure()
        bins = np.linspace(0.4, 1, 30)
        prob_mass_ratio_plotter.histogram(
            [gg.getMostCentralSubhalo().getStellarMass() / gg.getMostMassiveSubhalo().getStellarMass() for gg in self.filtered_gt13_ls13p5_list_of_galaxy_groups.getAllGalaxyGroups()],
            bins=bins,
            ax = prob_mass_ratio_ax,
            xlabel='Mass Ratio between Most Central and Most Massive Subhalo',
            ylabel='Density of Galaxy Groups',
            title=f'Mass Ratio between Most Central and Most Massive Subhalo for {self.sim}',
            label='1e13<$M_{{200}}$<1e13.5 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=None,
            percentage=True,
            grid=True,
        )
        prob_mass_ratio_plotter.histogram(
            [gg.getMostCentralSubhalo().getStellarMass() / gg.getMostMassiveSubhalo().getStellarMass() for gg in self.filtered_gt13p5_ls14_list_of_galaxy_groups.getAllGalaxyGroups()],
            bins=bins,
            ax = prob_mass_ratio_ax,
            xlabel='Mass Ratio between Most Central and Most Massive Subhalo',
            ylabel='Density of Galaxy Groups',
            title=f'Mass Ratio between Most Central and Most Massive Subhalo for {self.sim}',
            label='1e13.5<$M_{{200}}$<1e14 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=None,
            percentage=True,
            grid=True,
        )
        prob_mass_ratio_plotter.histogram(
            [gg.getMostCentralSubhalo().getStellarMass() / gg.getMostMassiveSubhalo().getStellarMass() for gg in self.filtered_gt14_ls14p5_list_of_galaxy_groups.getAllGalaxyGroups()],
            bins=bins,
            ax = prob_mass_ratio_ax,
            xlabel='Mass Ratio between Most Central and Most Massive Subhalo',
            ylabel='Density of Galaxy Groups',
            title=f'Mass Ratio between Most Central and Most Massive Subhalo for {self.plotIdentifier}',
            label='1e14<$M_{{200}}$<1e14.5 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=None,
            percentage=True,
            grid=True,
        )
        prob_mass_ratio_plotter.histogram(
            [gg.getMostCentralSubhalo().getStellarMass() / gg.getMostMassiveSubhalo().getStellarMass() for gg in self.filtered_gt14p5_ls15_list_of_galaxy_groups.getAllGalaxyGroups()],
            bins=bins,
            ax = prob_mass_ratio_ax,
            xlabel='Mass Ratio between Most Central and Most Massive Subhalo',
            ylabel='Density of Galaxy Groups',
            title=f'Mass Ratio between Most Central and Most Massive Subhalo for {self.plotIdentifier}',
            label='1e14.5<$M_{{200}}$<1e15 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=None,
            percentage=True,
            grid=True,
        )
        prob_mass_ratio_plotter.histogram(
            [gg.getMostCentralSubhalo().getStellarMass() / gg.getMostMassiveSubhalo().getStellarMass() for gg in self.filtered_gt15_list_of_galaxy_groups.getAllGalaxyGroups()],
            bins=bins,
            ax = prob_mass_ratio_ax,
            xlabel='Mass Ratio ($M_{{central}} / M_{{most massive}}$)',
            ylabel='Density of Galaxy Groups',
            title=f'Mass Ratio between Most Central and Most Massive Subhalo for {self.plotIdentifier}',
            label='$M_{{200}}$>1e15 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=self.scratchPlotDirc + f'/central_disparities/most_massive_vs_most_central_massRatio_{self.plotIdentifier}_massBins.png' if plot_dirc is None else plot_dirc + f'/central_disparities/most_massive_vs_most_central_massRatio_{self.plotIdentifier}_massBins.png',
            percentage=True,
            grid=True,
        )
        
        # plot the r/r200 of the most massive subhalo for these mass bins
        prob_massive_r_ratio_plotter = AstroPlotter()
        prob_massive_r_ratio_fig, prob_massive_r_ratio_ax = prob_massive_r_ratio_plotter.create_figure()
        bins = np.linspace(0, 1, 30)
        prob_massive_r_ratio_plotter.histogram(
            [np.linalg.norm(gg.getMostMassiveSubhalo().getPosition()) / gg.getRCrit200() for gg in self.filtered_gt13_ls13p5_list_of_galaxy_groups.getAllGalaxyGroups()],
            bins=bins,
            ax = prob_massive_r_ratio_ax,
            xlabel='r/r200 of Most Massive Subhalo',
            ylabel='Density of Galaxy Groups',
            title=f'r/r200 of Most Massive Subhalo for {self.plotIdentifier}',
            label='1e13<$M_{{200}}$<1e13.5 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=self.scratchPlotDirc + f'/central_disparities/most_massive_vs_most_central_r_ratio_{self.plotIdentifier}_massBins.png' if plot_dirc is None else plot_dirc + f'/central_disparities/most_massive_vs_most_central_r_ratio_{self.plotIdentifier}_massBins.png',
            percentage=True,
            grid=True,
        )
        prob_massive_r_ratio_plotter.histogram(
            [np.linalg.norm(gg.getMostMassiveSubhalo().getPosition()) / gg.getRCrit200() for gg in self.filtered_gt13p5_ls14_list_of_galaxy_groups.getAllGalaxyGroups()],
            bins=bins,
            ax = prob_massive_r_ratio_ax,
            xlabel='r/r200 of Most Massive Subhalo',
            ylabel='Density of Galaxy Groups',
            title=f'r/r200 of Most Massive Subhalo for {self.plotIdentifier}',
            label='1e13.5<$M_{{200}}$<1e14 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=None,
            percentage=True,
            grid=True,
        )
        prob_massive_r_ratio_plotter.histogram(
            [np.linalg.norm(gg.getMostMassiveSubhalo().getPosition()) / gg.getRCrit200() for gg in self.filtered_gt14_ls14p5_list_of_galaxy_groups.getAllGalaxyGroups()],
            bins=bins,
            ax = prob_massive_r_ratio_ax,
            xlabel='r/r200 of Most Massive Subhalo',
            ylabel='Density of Galaxy Groups',
            title=f'r/r200 of Most Massive Subhalo for {self.plotIdentifier}',
            label='1e14<$M_{{200}}$<1e14.5 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=None,
            percentage=True,
            grid=True,
        )
        prob_massive_r_ratio_plotter.histogram(
            [np.linalg.norm(gg.getMostMassiveSubhalo().getPosition()) / gg.getRCrit200() for gg in self.filtered_gt14p5_ls15_list_of_galaxy_groups.getAllGalaxyGroups()],
            bins=bins,
            ax = prob_massive_r_ratio_ax,
            xlabel='r/r200 of Most Massive Subhalo',
            ylabel='Density of Galaxy Groups',
            title=f'r/r200 of Most Massive Subhalo for {self.plotIdentifier}',
            label='1e14.5<$M_{{200}}$<1e15 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=None,
            percentage=True,
            grid=True,
        )
        prob_massive_r_ratio_plotter.histogram(
            [np.linalg.norm(gg.getMostMassiveSubhalo().getPosition()) / gg.getRCrit200() for gg in self.filtered_gt15_list_of_galaxy_groups.getAllGalaxyGroups()],
            bins=bins,
            ax = prob_massive_r_ratio_ax,
            xlabel='r/r200 of Most Massive Subhalo',
            ylabel='Density of Galaxy Groups',
            title=f'r/r200 of Most Massive Subhalo for {self.plotIdentifier}',
            label='$M_{{200}}$>1e15 Msun',
            ylog=True,
            legend=True,
            linealpha=0.5,
            output_filename=self.scratchPlotDirc + f'/central_disparities/most_massive_r_ratio_r200_{self.plotIdentifier}_massBins.png' if plot_dirc is None else plot_dirc + f'/central_disparities/most_massive_r_ratio_r200_{self.plotIdentifier}_massBins.png',
            percentage=True,
            grid=True,
        )
        
            
        self.plot_median_num_subhalos_by_mass_bins([(self.filtered_gt13_ls13p5_list_of_galaxy_groups, '1e13<$M_{{200}}$<1e13.5 Msun'),
                                       (self.filtered_gt13p5_ls14_list_of_galaxy_groups, '1e13.5<$M_{{200}}$<1e14 Msun'),
                                       (self.filtered_gt14_ls14p5_list_of_galaxy_groups, '1e14<$M_{{200}}$<1e14.5 Msun'),
                                       (self.filtered_gt14p5_ls15_list_of_galaxy_groups, '1e14.5<$M_{{200}}$<1e15 Msun'),
                                       (self.filtered_gt15_list_of_galaxy_groups, '$M_{{200}}$>1e15 Msun')], plot_dirc=plot_dirc) 
        
        # run for each mass bin
        self.plot_pairwise_polar_by_mass_bins_and_centralMassive_status([(self.filtered_gt13_ls13p5_list_of_galaxy_groups, '1e13<$M_{{200}}$<1e13.5 Msun'),
                                                                    (self.filtered_gt13p5_ls14_list_of_galaxy_groups, '1e13.5<$M_{{200}}$<1e14 Msun'),
                                                                    (self.filtered_gt14_ls14p5_list_of_galaxy_groups, '1e14<$M_{{200}}$<1e14.5 Msun'),
                                                                    (self.filtered_gt14p5_ls15_list_of_galaxy_groups, '1e14.5<$M_{{200}}$<1e15 Msun'),
                                                                    (self.filtered_gt15_list_of_galaxy_groups, '$M_{{200}}$>1e15 Msun')], plot_dirc=plot_dirc)
        # plot_pairwise_polar_by_mass_bins_and_centralMassive_status([
        #                                                             (filtered_gt15_list_of_galaxy_groups, '$M_{{200}}$>1e15 Msun')])
        
        #plot for all galaxy groups with central most massive vs not central most massive without splitting into mass bins
        prob_polar_mass_plotter, prob_polar_mass_fig, prob_polar_mass_ax = self.plot_pairwise_polar_by_mass_bins_and_centralMassive_status([(self.loaded_list_of_galaxy_groups, '$M_{{200}}$>1e13 Msun')])
        
        #replot this with different ylims
        prob_polar_mass_ax.set_ylim(0.0050, 0.0065)
        prob_polar_mass_plotter.save_figure(prob_polar_mass_fig, self.scratchPlotDirc + f'/pairwise_polar_difference_allMasses_{self.plotIdentifier}_differentYlim.png') if plot_dirc is None else prob_polar_mass_plotter.save_figure(prob_polar_mass_fig, plot_dirc + f'/pairwise_polar_difference_allMasses_{self.plotIdentifier}_differentYlim.png')

        #overlay
        # polar_bin_centers, pairwise_polar_differences, pairwise_polar_errorbars
        pairwiseplot, polar_bin_centers, pairwise_polar_differences_binned, pairwise_polar_errorbars = self.pairwisePolarDifferencePlot(self.loaded_list_of_galaxy_groups)
        prob_polar_mass_ax.errorbar(polar_bin_centers, pairwise_polar_differences_binned, yerr=pairwise_polar_errorbars, fmt='o', label='All Galaxy Groups', color='black', markersize=3, alpha=0.5)
        prob_polar_mass_ax.legend()
        prob_polar_mass_plotter.save_figure(prob_polar_mass_fig, self.scratchPlotDirc + f'/pairwise_polar_difference_allMasses_{self.plotIdentifier}_overlay.png') if plot_dirc is None else prob_polar_mass_plotter.save_figure(prob_polar_mass_fig, plot_dirc + f'/pairwise_polar_difference_allMasses_{self.plotIdentifier}_overlay.png')
        
    # plot the median number of subhalos within r200 for these mass bins
    def plot_median_num_subhalos_by_mass_bins(self, listGG : list[tuple[ListGalaxyGroup, str]], plot_dirc : str = None):
        median_subhalo_plotter = AstroPlotter()
        median_subhalo_fig, median_subhalo_ax = median_subhalo_plotter.create_figure()
        bins = np.linspace(0, 500, 100)
        mass_bin_labels = []
        for i, (mass_filtered_list, mass_bin_label) in enumerate(listGG):
            median_num_subhalos = []
            for galaxy_group in mass_filtered_list.getAllGalaxyGroups():
                median_num = len(galaxy_group.getSatelliteSubhalos())
                median_num_subhalos.append(median_num)
            mass_bin_labels.append(mass_bin_label)
            print(f"Mass bin: {mass_bin_label}, median number of subhalos: {median_num}")
        
            median_subhalo_plotter.histogram(
                median_num_subhalos,
                bins=bins,
                ax= median_subhalo_ax,
                xlabel='Number of Satellite Subhalos within r200',
                ylabel='Density of Galaxy Groups',
                title=f'Number of Satellite Subhalos within r200 for {self.plotIdentifier}',
                label=mass_bin_label,
                ylog=True,
                legend=True,
                linealpha=0.5,
                output_filename=None,
                percentage=True,
                grid=True,
            )
        median_subhalo_plotter.save_figure(
            fig=median_subhalo_fig,
            filename=self.scratchPlotDirc + f'/median_num_subhalos_by_mass_bins_{self.plotIdentifier}.png' if plot_dirc is None else plot_dirc + f'/median_num_subhalos_by_mass_bins_{self.plotIdentifier}.png'
        )
        
    #plot the pairwise polar differences for these mass bins and split between central is most massive or not
    def plot_pairwise_polar_by_mass_bins_and_centralMassive_status(self, listGG : list[tuple[ListGalaxyGroup, str]], plot_dirc : str = None): #mass_filtered_list : ListGalaxyGroup, mass_bin_label):
        prob_polar_mass_plotter = AstroPlotter()
        prob_polar_mass_fig, prob_polar_mass_ax = prob_polar_mass_plotter.create_figure(ncols=len(listGG), figsize=(8*len(listGG), 6))

        colors = ['blue', 'orange', 'red', 'green', 'purple']                                                          
        for i, (mass_filtered_list, mass_bin_label) in enumerate(listGG):
            central_most_massive_list_of_galaxy_groups = mass_filtered_list.getFilterSubhalos(centralIsMostMassive=True)
            not_central_most_massive_list_of_galaxy_groups = mass_filtered_list.getFilterSubhalos(centralIsMostMassive=False)
            print(f"len GG: {len(central_most_massive_list_of_galaxy_groups.getAllGalaxyGroups()), len(not_central_most_massive_list_of_galaxy_groups.getAllGalaxyGroups())}")

            list_pairwise_polar_differences_centralMassive = central_most_massive_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/pairwise_polar_centralMassive_{mass_bin_label}_{self.plotIdentifier}', rewrite=self.generalRewrite)
            list_pairwise_polar_differences_notCentralMassive = not_central_most_massive_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/pairwise_polar_notCentralMassive_{mass_bin_label}_{self.plotIdentifier}', rewrite=self.generalRewrite)
            list_pairwise_polar_differences_total = mass_filtered_list.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/pairwise_polar_{mass_bin_label}_{self.plotIdentifier}', rewrite=self.generalRewrite)
            
            pairwise_polar_differences_centralMassive, polar_bin_centers_centralMassive, pairwise_polar_centralMassive_errorbars = ListGalaxyGroup.get_histogram_bins(list_pairwise_polar_differences_centralMassive, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
            pairwise_polar_differences_notCentralMassive, polar_bin_centers_notCentralMassive, pairwise_polar_notCentralMassive_errorbars = ListGalaxyGroup.get_histogram_bins(list_pairwise_polar_differences_notCentralMassive, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
            pairwise_polar_differences_total, polar_bin_centers_total, pairwise_polar_total_errorbars = ListGalaxyGroup.get_histogram_bins(list_pairwise_polar_differences_total, bins = np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
        
            if len(listGG) > 1:
                print(f"choosing ax, {i}")
                axToPlot = prob_polar_mass_ax[i]
            else:
                axToPlot = prob_polar_mass_ax
            print(f"len: {len(polar_bin_centers_centralMassive), len(polar_bin_centers_notCentralMassive), len(polar_bin_centers_total)}")
            print(f"first 5 values centralMassive: {pairwise_polar_differences_centralMassive[:5]}, notCentralMassive: {pairwise_polar_differences_notCentralMassive[:5]}, total: {pairwise_polar_differences_total[:5]}")

            prob_polar_mass_plotter.scatter_plot(
                polar_bin_centers_centralMassive, 
                pairwise_polar_differences_centralMassive,
                # errorBars = pairwise_polar_centralMassive_errorbars,
                ax=axToPlot, # Use the same axis for overlay
                overlay_color=colors[0],
                # xlabel='Pairwise Polar Difference (degrees)',
                # ylabel='Probability Density',
                # title=f'Probability Pairwise Polar Difference Distribution for {sim} ({mass_bin_label})',
                # ylim = (0, 0.01),
                label="Central is Most Massive",
                # output_filename=None,  # Disable saving for combined plot
                # grid=True,
            )
            prob_polar_mass_plotter.scatter_plot(
                polar_bin_centers_notCentralMassive, 
                pairwise_polar_differences_notCentralMassive,
                # errorBars = pairwise_polar_notCentralMassive_errorbars,
                ax=axToPlot,  # Use the same axis for overlay
                overlay_color=colors[1], 
                # xlabel='Pairwise Polar Difference (degrees)',
                # ylabel='Probability Density',
                # title=f'Probability Pairwise Polar Difference Distribution for {sim} ({mass_bin_label})',
                # ylim = (0, 0.01),
                label="Central is NOT Most Massive",
                # include_legend=True,
                # output_filename=None,
                # grid=True,
            )
            prob_polar_mass_plotter.scatter_plot(
                polar_bin_centers_total, 
                pairwise_polar_differences_total,
                # errorBars = pairwise_polar_total_errorbars,
                ax=axToPlot,  # Use the same axis for overlay
                overlay_color=colors[2],
                xlabel='Pairwise Polar Difference (degrees)',
                ylabel='Probability Density',
                title=f'Probability Pairwise Polar Difference Distribution for {self.sim} ({mass_bin_label})',
                # ylim = (0, 0.01),
                s=10,
                
                label="All Galaxy Groups",
                include_legend=True,
                # output_filename=scratchPlotDirc + f'/pairwise_polar_difference_total_{mass_bin_label.replace("<","lt").replace(">","gt")}_{sim}.png',
                grid=True,
            )

            
        return prob_polar_mass_plotter, prob_polar_mass_fig, prob_polar_mass_ax
        
    def overlayMRLAndMRLRandom(self, listGG : list[tuple[ListGalaxyGroup, str]], plot_dirc : str = None):
        overlayMRLPlotter = AstroPlotter()
        overlayMRLFig, overlayMRLAx = overlayMRLPlotter.create_figure(ncols=len(listGG), nrows=1, figsize=(8*len(listGG), 6)) if len(listGG) > 1 else overlayMRLPlotter.create_figure()
        
        numsamples=1000

        for i, (listGalaxyGroup, label) in enumerate(listGG):
            MRL_values = listGalaxyGroup.compute_probablity_distribution_of_MRL_directionality(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/MRL_values_{label}_{self.plotIdentifier}', rewrite=self.generalRewrite)
            if not MRL_values or len(MRL_values) == 0:
                print(f"Skipping {label}: No MRL values to plot.")
                continue
            MRL_binned, MRL_bin_centers, MRL_errorbars = ListGalaxyGroup.get_histogram_bins(MRL_values, binsize=0.05, binLow=0, binHigh=1, errorbarType='poisson')

            random_MRL_values = listGalaxyGroup.compute_MRL_random_distribution_curves_for_LGG(parallelize=False, num_samples=numsamples)
            if not random_MRL_values or len(random_MRL_values) == 0:
                print(f"Skipping {label}: No random MRL values to plot.")
                continue
            random_MRL_bins, random_MRL_bin_centers, random_MRL_errorbars = ListGalaxyGroup.get_histogram_bins(random_MRL_values, binsize=0.05, binLow=0, binHigh=1, errorbarType='poisson')

            if len(listGG) > 1:
                print(f"choosing ax, {i}")
                overlayAxToPlot = overlayMRLAx[i]
            else:
                overlayAxToPlot = overlayMRLAx

            overlayMRLPlotter.scatter_plot(
                MRL_bin_centers, 
                MRL_binned,
                # errorBars = MRL_errorbars,
                ax=overlayAxToPlot, # Use the same axis for overlay
                # ylim = (0, 0.01),
                label=f"MRL Directionality ({listGalaxyGroup.getNumGalaxyGroups()} Galaxy Groups in {label})",
                output_filename=None,  # Disable saving for combined plot
                grid=True
            )
            #plot a small verticle line at 99th percentile of radnom MRL_values
            percentile_99_MRL = np.percentile(random_MRL_values, 99)
            # overlayMRLAx.axvline(percentile_99_MRL, linestyle='--', label=f'99th Percentile')

            overlayMRLPlotter.scatter_plot(
                random_MRL_bin_centers, 
                random_MRL_bins,
                # errorBars = random_MRL_errorbars,
                ax=overlayAxToPlot,  # Use the same axis for overlay
                # ylim = (0, 0.01),
                label=f"Random MRL Directionality ({len(random_MRL_values)} Samples)",
                include_legend=True,
                overlay_color='gray',
                # output_filename=scratchPlotDirc + f'/MRL_distribution_curves_overlay_{sim}.png',
                output_filename=None,
                grid=True,
            )

            overlayMRLPlotter.scatter_plot(
                random_MRL_bin_centers, 
                random_MRL_bins,
                # errorBars = random_MRL_errorbars,
                ax=overlayAxToPlot,  # Use the same axis for overlay
                xlabel='MRL Directionality',
                ylabel='Probability Density',
                title=f'MRL Distribution Curves vs Random for {self.sim}, {label}',
                # ylim = (0, 0.01),
                label=f"Random MRL Directionality ({len(random_MRL_values)} Samples)",
                include_legend=True,
                overlay_color='gray',
                alpha=0,
                # output_filename=scratchPlotDirc + f'/MRL_distribution_curves_overlay_{sim}.png',
                output_filename=None,
                grid=True,
                spline_curvature=True,
                spline_smoothing=0,
            )
            print(f"99th percentile MRL: {percentile_99_MRL}")
            print(f"Overall number of MRL values above 99th percentile: {np.sum(np.array(MRL_values) > percentile_99_MRL)} out of {len(MRL_values)}")
            
            print(f"len listGG: {len(listGG)}, len MRL_values: {len(MRL_values)}, len random_MRL_values: {len(random_MRL_values)}")
            #include a text box in the plot with the fraction of MRL values that are less than the 99th percentile of random MRL values
            fraction_less_than_99th_percentile = np.sum(np.array(MRL_values) < percentile_99_MRL) / len(MRL_values)
            overlayMRLPlotter.add_text_box(overlayAxToPlot, f"Fraction of MRL values < 99th percentile of random MRL: {fraction_less_than_99th_percentile:.2f}", loc='bottom center')
            
            # save into .txt file:
            # In table: galaxy id, num of members, mass of cluster, MRL value of that projection, fraction less than the MRL I measured
            with open(self.scratchPlotDirc + f'/MRL_values_and_random_comparison_{label}_{self.plotIdentifier}.txt', 'w') as f:
                f.write("GalaxyGroupID\tNumMembers\tClusterMass\tMRLValue\tFractionLessThanMRL\n")
                for j, gg in enumerate(listGalaxyGroup.getAllGalaxyGroups()):
                    print(f"processing galaxy group {j+1}/{len(listGalaxyGroup.getAllGalaxyGroups())} for fraction", end='\r', flush=True)
                    gg_id = gg.getGroupID()
                    num_members = gg.getNumSubhalos()
                    cluster_mass = gg.getMCrit200()
                    MRL_value = [MRL_values[j * 3], MRL_values[j * 3 + 1], MRL_values[j * 3 + 2]] if j * 3 + 2 < len(MRL_values) else [0, 0, 0]
                    fraction_less_than_MRL =[np.sum(random_MRL_values[j] < MRL_values[j * 3]) / numsamples, np.sum(random_MRL_values[j] < MRL_values[j * 3 + 1]) / numsamples, np.sum(random_MRL_values[j] < MRL_values[j * 3 + 2]) / numsamples] if j * 3 + 2 < len(MRL_values) else [0, 0, 0]
                    f.write(f"{gg_id}\t{num_members}\t{cluster_mass}\t{MRL_value}\t{fraction_less_than_MRL}\n")
                    
            #save into .txt file, all the galaxy groups who's MRL_value is greater than 0.8
            with open(self.scratchPlotDirc + f'/high_MRL_galaxy_groups_{label}_{self.plotIdentifier}.txt', 'w') as f:
                f.write("GalaxyGroupID\tNumMembers\tClusterMass\tMRLValue\n")
                for k, gg in enumerate(listGalaxyGroup.getAllGalaxyGroups()):
                    print(f"processing galaxy group {k+1}/{len(listGalaxyGroup.getAllGalaxyGroups())} for high MRL", end='\r', flush=True)
                    gg_id = gg.getGroupID()
                    num_members = gg.getNumSubhalos()
                    cluster_mass = gg.getMCrit200()
                    MRL_value = [MRL_values[k * 3], MRL_values[k * 3 + 1], MRL_values[k * 3 + 2]] if k * 3 + 2 < len(MRL_values) else [0, 0, 0]
                    if np.any(np.array(MRL_value) > 0.8):
                        f.write(f"{gg_id}\t{num_members}\t{cluster_mass}\t{MRL_value}\n")
                        
            #save bin centers and probabilities and errorbars to text file
            # Save bin centers and probabilities and errorbars to text file, padding with NaN if needed
            with open(self.scratchPlotDirc + f'/MRL_distribution_curves_{label}_{self.plotIdentifier}.txt', 'w') as f:
                f.write("MRLBinCenter\tMRLProbability\tMRLErrorBar\tRandomMRLBinCenter\tRandomMRLProbability\tRandomMRLErrorBar\n")
                arrays = [MRL_bin_centers, MRL_binned, MRL_errorbars, random_MRL_bin_centers, random_MRL_bins, random_MRL_errorbars]
                max_len = max(len(arr) for arr in arrays)
                arrays_padded = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in arrays]
                for m in range(max_len):
                    f.write("\t".join(str(arr[m]) for arr in arrays_padded) + "\n")
            #save raw MRL values and random MRL values to text file, padding with NaN if needed
            with open(self.scratchPlotDirc + f'/MRL_raw_values_{label}_{self.plotIdentifier}.txt', 'w') as f:
                f.write("MRLValue\tRandomMRLValue\n")
                max_raw_len = max(len(MRL_values), len(random_MRL_values))
                mrl_values_padded = np.pad(MRL_values, (0, max_raw_len - len(MRL_values)), constant_values=np.nan)
                random_mrl_values_padded = np.pad(random_MRL_values, (0, max_raw_len - len(random_MRL_values)), constant_values=np.nan)
                for n in range(max_raw_len):
                    f.write(f"{mrl_values_padded[n]}\t{random_mrl_values_padded[n]}\n")
        # overlayAxToPlot.legend()
        overlayMRLPlotter.save_figure(overlayMRLFig, self.scratchPlotDirc + f'/MRL_distribution_curves_overlay_{self.plotIdentifier}.png') if plot_dirc is None else overlayMRLPlotter.save_figure(overlayMRLFig, plot_dirc + f'/MRL_distribution_curves_overlay_{self.plotIdentifier}.png')
        
        #append into overall .txt file:
        with open(self.scratchPlotDirc + f'/MRL_values_and_random_comparison_overall.txt', 'a') as f:
            f.write(f"Simulation: {self.sim}\n")
            f.write("GalaxyGroupID\tNumMembers\tClusterMass\tMRLValue\tFractionLessThanMRL\n")
            for l, (listGalaxyGroup, label) in enumerate(listGG):
                comparison_file = self.scratchPlotDirc + f'/MRL_values_and_random_comparison_{label}_{self.plotIdentifier}.txt'
                if not os.path.exists(comparison_file):
                    print(f"Skipping missing file: {comparison_file}")
                    continue
                with open(comparison_file, 'r') as g:
                    lines = g.readlines()
                    for line in lines[1:]:  # Skip header line
                        f.write(line)
        
    def MRLDistributionPlots(self, plot_dirc : str = None):
        #plot the MRL distribution for different mass bins
        MRL_20_values = ListGalaxyGroup.compute_an_MRL_distribution_curves(parallelize=False, num_samples=10000, num_non_centrals=20)
        MRL_20, MRL_20_bin_edges, MRL_20_errorbars = ListGalaxyGroup.get_histogram_bins(MRL_20_values, binsize=0.005, binLow=0, binHigh=1, errorbarType='poisson')

        MRL_50_values = ListGalaxyGroup.compute_an_MRL_distribution_curves(parallelize=False, num_samples=10000, num_non_centrals=50)
        MRL_50, MRL_50_bin_edges, MRL_50_errorbars = ListGalaxyGroup.get_histogram_bins(MRL_50_values, binsize=0.005, binLow=0, binHigh=1, errorbarType='poisson')

        MRL_100_values = ListGalaxyGroup.compute_an_MRL_distribution_curves(parallelize=False, num_samples=10000, num_non_centrals=100)
        MRL_100, MRL_100_bin_edges, MRL_100_errorbars = ListGalaxyGroup.get_histogram_bins(MRL_100_values, binsize=0.005, binLow=0, binHigh=1, errorbarType='poisson')

        MRL_200_values = ListGalaxyGroup.compute_an_MRL_distribution_curves(parallelize=False, num_samples=10000, num_non_centrals=200)
        MRL_200, MRL_200_bin_edges, MRL_200_errorbars = ListGalaxyGroup.get_histogram_bins(MRL_200_values, binsize=0.005, binLow=0, binHigh=1, errorbarType='poisson')

        # print(len(MRL_20), len(MRL_20_bin_edges))
        # print(MRL_20[MRL_20 > 0])
        #plot with spline curve
        MRL_plotter = AstroPlotter()
        MRL_fig, MRL_ax = MRL_plotter.create_figure()
        MRL_plotter.scatter_plot(
            MRL_20_bin_edges, 
            MRL_20,
            # errorBars = MRL_20_errorbars,
            ax=MRL_ax, # Use the same axis for overlay
            # ylim = (0, 0.01),
            label="20 Non-Centrals",
            alpha=0,
            s=50,
            output_filename=None,  # Disable saving for combined plot
            grid=True,
            spline_curvature=True,
        )
        #plot a small verticle line at 99th percentile of MRL_20_values
        percentile_99_MRL_20 = np.percentile(MRL_20_values, 99)
        # MRL_ax.axvline(percentile_99_MRL_20, color='blue', linestyle='--', label='99th Percentile (20 Non-Centrals)')

        MRL_plotter.scatter_plot(
            MRL_50_bin_edges, 
            MRL_50,
            # errorBars = MRL_50_errorbars,
            ax=MRL_ax,  # Use the same axis for overlay
            # ylim = (0, 0.01),
            label="50 Non-Centrals",
            alpha=0,
            output_filename=None,
            grid=True,
            spline_curvature=True,
        )
        #plot a small verticle line at 99th percentile of MRL_50_values
        percentile_99_MRL_50 = np.percentile(MRL_50_values, 99)
        # MRL_ax.axvline(percentile_99_MRL_50, color='orange', linestyle='--', label='99th Percentile (50 Non-Centrals)')


        MRL_plotter.scatter_plot(
            MRL_100_bin_edges, 
            MRL_100,
            # errorBars = MRL_100_errorbars,
            ax=MRL_ax,  # Use the same axis for overlay
            # ylim = (0, 0.01),
            label="100 Non-Centrals",
            alpha=0,
            output_filename=None,
            grid=True,
            spline_curvature=True,
        )
        #plot a small verticle line at 99th percentile of MRL_100_values
        percentile_99_MRL_100 = np.percentile(MRL_100_values, 99)
        # MRL_ax.axvline(percentile_99_MRL_100, color='red', linestyle='--', label='99th Percentile (100 Non-Centrals)')


        MRL_plotter.scatter_plot(
            MRL_200_bin_edges, 
            MRL_200,
            # errorBars = MRL_500_errorbars,
            ax=MRL_ax,  # Use the same axis for overlay
            xlabel='MRL Directionality',
            ylabel='Probability Density',
            title=f'MRL Distribution Curves',
            # ylim = (0, 0.01),
            label="200 Non-Centrals",
            alpha=0,
            output_filename=self.scratchPlotDirc + f'/MRL_distribution_curves_{self.plotIdentifier}.png' if plot_dirc is None else plot_dirc + f'/MRL_distribution_curves_{self.plotIdentifier}.png',
            include_legend=True,
            grid=True,
            spline_curvature=True,
        )
        #plot a small verticle line at 99th percentile of MRL_200_values
        percentile_99_MRL_200 = np.percentile(MRL_200_values, 99)
        # MRL_ax.axvline(percentile_99_MRL_200, color='green', linestyle='--', label='99th Percentile (200 Non-Centrals)')

        ymin, ymax = MRL_ax.get_ylim()
        MRL_ax.plot([percentile_99_MRL_20, percentile_99_MRL_20], [ymin, ymax*0.1], color='blue', linestyle='--', label='99th Percentile (20 Non-Centrals)')
        print(MRL_ax.get_ylim())
        MRL_ax.plot([percentile_99_MRL_50, percentile_99_MRL_50], [ymin, ymax*0.1], color='orange', linestyle='--', label='99th Percentile (50 Non-Centrals)')
        print(MRL_ax.get_ylim())
        MRL_ax.plot([percentile_99_MRL_100, percentile_99_MRL_100], [ymin, ymax*0.1], color='green', linestyle='--', label='99th Percentile (100 Non-Centrals)')
        MRL_ax.plot([percentile_99_MRL_200, percentile_99_MRL_200], [ymin, ymax*0.1], color='red', linestyle='--', label='99th Percentile (200 Non-Centrals)')
        MRL_ax.legend()
        
        # list_of_galaxy_groups_gt13 = self.loaded_list_of_galaxy_groups.getFilterSubhalos(minGGMass=1e13)

        # MRL_gt14_values = self.filtered_gt14_list_of_galaxy_groups.compute_probablity_distribution_of_MRL_directionality(parallelize=False)
        # random_MRL_values_gt14 = self.filtered_gt14_list_of_galaxy_groups.compute_MRL_random_distribution_curves_for_LGG(parallelize=False, num_samples=10000)
                            
        self.overlayMRLAndMRLRandom([(self.filtered_gt14_list_of_galaxy_groups, '$M_{200}$>1e14 Msun')])
        
        #compute for each group (13-13.5, 13.5-14, 14-14.5, 14.5-15, >15)
        self.overlayMRLAndMRLRandom([(self.filtered_gt13_ls13p5_list_of_galaxy_groups, '$13<M_{200}<13.5$'), (self.filtered_gt13p5_ls14_list_of_galaxy_groups, '$13.5<M_{200}<14$'), (self.filtered_gt14_ls14p5_list_of_galaxy_groups, '$14<M_{200}<14.5$'), (self.filtered_gt14p5_ls15_list_of_galaxy_groups, '$14.5<M_{200}<15$'), (self.filtered_gt15_list_of_galaxy_groups, '$M_{200}>15$')])

    #for each group, plot the probability distribution of a galaxy group's M200
    def M200DistributionPlots(self, listGG : list[tuple[ListGalaxyGroup, str]], plot_dirc : str = None):
        M200_plotter = AstroPlotter()
        M200_fig, M200_ax = M200_plotter.create_figure(ncols=len(listGG), nrows=1, figsize=(8, 6*len(listGG)) if len(listGG) > 1 else M200_plotter.create_figure())
        for i, (listGalaxyGroup, label) in enumerate(listGG):
            M200_values = []
            for gg in listGalaxyGroup.getAllGalaxyGroups():
                print(f"processing galaxy group {gg.getGroupID()} for M200 distribution", end='\r', flush=True)
                M200_values.append(gg.getMCrit200())
            M200_bins, M200_bin_edges, M200_errorbars = ListGalaxyGroup.get_histogram_bins(M200_values, binsize=0.1, binLow=13, binHigh=15, errorbarType='poisson')

            if len(listGG) > 1:
                print(f"choosing ax, {i}")
                axToPlot = M200_ax[i]
            else:
                axToPlot = M200_ax

            M200_plotter.scatter_plot(
                M200_bin_edges, 
                M200_bins,
                # errorBars = M200_errorbars,
                ax=axToPlot, # Use the same axis for overlay
                xlabel='$M_{200}$ (Msun)',
                ylabel='Probability Density',
                title=f'$M_{{200}}$ Distribution for {self.sim} ({label})',
                # ylim = (0, 0.01),
                label=f"{label}",
                output_filename=None,
                grid=True,
            )
        M200_plotter.save_figure(M200_fig, self.scratchPlotDirc + f'/M200_distribution_{self.plotIdentifier}.png' if plot_dirc is None else plot_dirc + f'/M200_distribution_{self.plotIdentifier}.png')


    def HighMRLPlots(self, plot_dirc : str = None):
        # read high_MRL_galaxy_groups_13>$M_{200}>13.5$.txt file and get a list of galaxy group IDs with high MRL values
        high_MRL_galaxy_group_ids = []
        with open(self.scratchPlotDirc + f'/high_MRL_galaxy_groups_$13<M_{{200}}<13.5$_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header line
                parts = line.split('\t')
                if len(parts) > 0:
                    high_MRL_galaxy_group_ids.append(int(parts[0]))
                    
        high_MRL_galaxy_groups = []
        for gg in self.filtered_gt13_ls13p5_list_of_galaxy_groups.getAllGalaxyGroups():
            if gg.getGroupID() in high_MRL_galaxy_group_ids:
                high_MRL_galaxy_groups.append(gg)
                
        high_MRL_list_of_galaxy_groups = ListGalaxyGroup(high_MRL_galaxy_groups)

        #plot the pairwise polar distribution for these high MRL groups
        high_MRL_pairwise_polar_differences = high_MRL_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/high_MRL_pairwise_polar_differences_{self.plotIdentifier}', rewrite=self.generalRewrite)
        high_MRL_pairwise_polar_bins, high_MRL_pairwise_polar_bin_centers, high_MRL_pairwise_polar_errorbars = ListGalaxyGroup.get_histogram_bins(high_MRL_pairwise_polar_differences, bins = np.arange(0, 185, 10), errorbarType='poisson')
        high_MRL_polar_plotter = AstroPlotter()
        high_MRL_polar_fig, high_MRL_polar_ax = high_MRL_polar_plotter.create_figure()
        high_MRL_polar_plotter.scatter_plot(
            high_MRL_pairwise_polar_bin_centers, 
            high_MRL_pairwise_polar_bins,
            errorBars = high_MRL_pairwise_polar_errorbars,
            ax=high_MRL_polar_ax, # Use the same axis for overlay
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Pairwise Polar Difference Distribution for High MRL Galaxy Groups ($13<M_{{200}}<13.5$) in {self.sim}',
            # ylim = (0, 0.01),
            label="High MRL Galaxy Groups",
            output_filename=self.scratchPlotDirc + f'/high_MRL_pairwise_polar_distribution_{self.sim}.png',
            grid=True,
        )

        #plot <35% R200 and >65% R200 for these high MRL groups
        high_MRL_LT35R200_list_of_galaxy_groups = high_MRL_list_of_galaxy_groups.getFilterSubhalos(withinXPercentR200=[0,0.35])
        high_MRL_pairwise_polar_differences_inner = high_MRL_LT35R200_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/high_MRL_pairwise_polar_differences_inner_{self.plotIdentifier}', rewrite=self.generalRewrite)
        high_MRL_pairwise_polar_bins_inner, high_MRL_pairwise_polar_bin_centers_inner, high_MRL_pairwise_polar_errorbars_inner = ListGalaxyGroup.get_histogram_bins(high_MRL_pairwise_polar_differences_inner, bins = np.arange(0, 185, 10), errorbarType='poisson')
        high_MRL_GT65R200_list_of_galaxy_groups = high_MRL_list_of_galaxy_groups.getFilterSubhalos(withinXPercentR200=[0.65,1])
        high_MRL_pairwise_polar_differences_outer = high_MRL_GT65R200_list_of_galaxy_groups.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/high_MRL_pairwise_polar_differences_outer_{self.plotIdentifier}', rewrite=self.generalRewrite)
        high_MRL_pairwise_polar_bins_outer, high_MRL_pairwise_polar_bin_centers_outer, high_MRL_pairwise_polar_errorbars_outer = ListGalaxyGroup.get_histogram_bins(high_MRL_pairwise_polar_differences_outer, bins = np.arange(0, 185, 10), errorbarType='poisson')
        high_MRL_polar_plotter.scatter_plot(
            high_MRL_pairwise_polar_bin_centers_inner, 
            high_MRL_pairwise_polar_bins_inner,
            errorBars = high_MRL_pairwise_polar_errorbars_inner,
            ax=high_MRL_polar_ax, # Use the same axis for overlay
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Pairwise Polar Difference Distribution for High MRL Galaxy Groups ($13<M_{{200}}<13.5$) in {self.sim}',
            # ylim = (0, 0.01),
            label="High MRL Galaxy Groups (<35% R200)",
            output_filename=None,
            grid=True,
        )
        high_MRL_polar_plotter.scatter_plot(
            high_MRL_pairwise_polar_bin_centers_outer, 
            high_MRL_pairwise_polar_bins_outer,
            errorBars = high_MRL_pairwise_polar_errorbars_outer,
            ax=high_MRL_polar_ax, # Use the same axis for overlay
            xlabel='Pairwise Polar Difference (degrees)',
            ylabel='Probability Density',
            title=f'Pairwise Polar Difference Distribution for High MRL Galaxy Groups ($13<M_{{200}}<13.5$) in {self.sim}',
            # ylim = (0, 0.01),
            label="High MRL Galaxy Groups (>65% R200)",
            output_filename=self.scratchPlotDirc + f'/high_MRL_pairwise_polar_distribution_inner_outer_{self.sim}.png',
            include_legend=True,
            grid=True,
        )

        # get the percentage of high_MRL galaxy groups which have only 2 satellites
        for gg in high_MRL_list_of_galaxy_groups.getAllGalaxyGroups():
            print(f"Galaxy Group ID: {gg.getGroupID()}, Num Subhalos: {gg.getNumSubhalos()}")
        high_MRL_2satellite_count = sum(1 for gg in high_MRL_list_of_galaxy_groups.getAllGalaxyGroups() if gg.getNumSubhalos() == 2)
        high_MRL_total_count = len(high_MRL_list_of_galaxy_groups.getAllGalaxyGroups())
        high_MRL_2satellite_percentage = (high_MRL_2satellite_count / high_MRL_total_count) * 100 if high_MRL_total_count > 0 else 0
        print(f"Percentage of High MRL Galaxy Groups with Only 2 Satellites: {high_MRL_2satellite_percentage:.2f}% ({high_MRL_2satellite_count} out of {high_MRL_total_count})")
        #write into file
        with open(self.scratchPlotDirc + f'/high_MRL_2satellite_percentage_{self.sim}.txt', 'w') as f:
            f.write(f"Percentage of High MRL Galaxy Groups with Only 2 Satellites: {high_MRL_2satellite_percentage:.2f}% ({high_MRL_2satellite_count} out of {high_MRL_total_count})\n")
        #save bin centers and probabilities and errorbars to text file
        with open(self.scratchPlotDirc + f'/high_MRL_pairwise_polar_distribution_{self.sim}.txt', 'w') as f:
            f.write("PolarBinCenter\tPolarProbability\tPolarErrorBar\n")
            for m, (polar_bin_center, polar_prob, polar_errorbar) in enumerate(zip(high_MRL_pairwise_polar_bin_centers, high_MRL_pairwise_polar_bins, high_MRL_pairwise_polar_errorbars)):
                f.write(f"{polar_bin_center}\t{polar_prob}\t{polar_errorbar}\n")

    # for each mass bin, get the probability a galaxy group has some number of satellites
    def plot_satellite_number_distribution_by_mass_bins(self, listGG : list[tuple[ListGalaxyGroup, str]]):
        satellite_number_plotter = AstroPlotter()
        satellite_number_fig, satellite_number_ax = satellite_number_plotter.create_figure()
        
        for listGalaxyGroup, label in listGG:
            satellite_numbers = [len(gg.getSatelliteSubhalos()) for gg in listGalaxyGroup.getAllGalaxyGroups()]
            satellite_number_bins, satellite_number_bin_edges, satellite_number_errorbars = ListGalaxyGroup.get_histogram_bins(satellite_numbers, bins='auto', errorbarType='poisson')
            
            satellite_number_plotter.scatter_plot(
                satellite_number_bin_edges, 
                satellite_number_bins,
                errorBars = satellite_number_errorbars,
                ax=satellite_number_ax, # Use the same axis for overlay
                xlabel='Number of Satellites',
                ylabel='Probability Density',
                title=f'Number of Satellites Distribution for {self.sim} Galaxy Groups in {label}',
                # ylim = (0, 0.01),
                label=f"{label}",
                output_filename=None,
                grid=True,
            )
        satellite_number_ax.legend()
        #set ylim min to 0
        satellite_number_ax.set_ylim(bottom=0)
        satellite_number_plotter.save_figure(satellite_number_fig, self.scratchPlotDirc + f'/satellite_number_distribution_by_mass_bins_{self.sim}.png')
        
    def plot_satellite_number_distribution_for_all_mass_bins(self, plot_dirc : str = None):
        list_of_mass_list_galaxy_groups = [self.filtered_gt13_ls13p5_list_of_galaxy_groups, self.filtered_gt13p5_ls14_list_of_galaxy_groups, self.filtered_gt14_ls14p5_list_of_galaxy_groups, self.filtered_gt14p5_ls15_list_of_galaxy_groups, self.filtered_gt15_list_of_galaxy_groups]
        mass_bin_labels = ['$13<M_{200}<13.5$', '$13.5<M_{200}<14$', '$14<M_{200}<14.5$', '$14.5<M_{200}<15$', '$M_{200}>15$']
        list_of_mass_bin_galaxy_groups = list(zip(list_of_mass_list_galaxy_groups, mass_bin_labels))
        self.plot_satellite_number_distribution_by_mass_bins(list_of_mass_bin_galaxy_groups)

    def get_satellite_join_time(self, listGG:list[tuple[ListGalaxyGroup, str]], rewrite:bool = True):
        #Box boundary 
        # L=75000. #kpc
        # halfbox=L/2.   
        # h=0.6774 
        joinTime = JoinTime(self.sim, self.snapshot)

        totalSatellites = sum(gg.getNumSubhalos() for listGalaxyGroup, _ in listGG for gg in listGalaxyGroup.getAllGalaxyGroups())
        processedSatelliteIds = []
        print(f"Total number of satellites to process: {totalSatellites}")
        
        satellitesWihtoutMergerTree = []
        #load satellitesWihtoutMergerTree from file if exists
        # if not rewrite:
        if os.path.exists(self.scratchPlotDirc + f'/satellites_without_merger_tree_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt'):
            with open(self.scratchPlotDirc + f'/satellites_without_merger_tree_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt', 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header line
                    parts = line.split('\t')
                    if len(parts) > 0:
                        satellitesWihtoutMergerTree.append(int(parts[0]))
            print(f"Loaded {len(satellitesWihtoutMergerTree)} satellites without merger tree from file.")
        else:
            print("No existing file for satellites without merger tree found, starting with an empty list.")
            #prep the file with header
            with open(self.scratchPlotDirc + f'/satellites_without_merger_tree_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt', 'w') as f:
                f.write("SatelliteSubhaloID\n")
        
        #get all join times for each mass group
        for listGalaxyGroup, label in listGG:
            if not rewrite:
                #check if file already exists, if so, skip
                for _, label in listGG:
                    if os.path.exists(self.scratchPlotDirc + f'/join_times_and_parameter_changes_{self.sim}_{self.snapshot_dic[self.snapshot][1]}_{label}.txt'):
                        print(f"File join_times_and_parameter_changes_{label}.txt already exists, skipping...")
                        return
            with open(self.scratchPlotDirc + f'/join_times_and_parameter_changes_{self.sim}_{self.snapshot_dic[self.snapshot][1]}_{label}.txt', 'w') as f:
                f.write("GalaxyGroupID\tNumMembers\tClusterMass\tJoiningRedshift\tSeparationAtZ0\tSeparationNormAtZ0\tDeltaGasMass\tDeltaTotalMass\tDeltaDMMass\tDeltaStellarMass\tDeltaVelSq\tJoiningSnap\tClosestApproach\tClosestApproachNorm\tClosestApproachRedshift\tJoinProgID\tDeltaAngularMomentum\tSatelliteMassAtJoining\tHostProgID\n")
                for gg in listGalaxyGroup.getAllGalaxyGroups():
                    # print(f"Processing Galaxy Group ID: {gg.getGroupID()}")
                    gg_id = gg.getGroupID()
                    num_members = gg.getNumSubhalos()
                    cluster_mass = gg.getMCrit200()
                    #get the subhalo ID of the central galaxy, which is the one that joins the host halo
                    central_subhalo : GalaxyGroup = gg.getCentralSubhalo()
                    for i, subhalo in enumerate(gg.getSatelliteSubhalos()):
                        if central_subhalo is not None:
                            # print(f"id: {subhalo.getIdx()}")
                            #check if satellite has no merger tree
                            if subhalo.getIdx() in satellitesWihtoutMergerTree:
                                print(f"Skipping subhalo {subhalo.getIdx()} (previously identified as having no merger tree)")
                                continue
                            print(f"Processing subhalo {subhalo.getIdx()}, progress: {i}/{num_members-1} satellites in this group, total progress: {len(processedSatelliteIds)}/{totalSatellites} satellites", end='\r', flush=True)
                            join_time_info = joinTime.computeJoinTimes(hostID=central_subhalo.getGroupID(), ID=subhalo.getIdx(), L=self.L, halfbox=self.halfbox, fname=self.scratchDataDirc+'/mergerTree')
                            processedSatelliteIds.append(subhalo.getIdx())
                            if join_time_info is None:
                                print(f"Skipping subhalo {subhalo.getIdx()} (no merger tree available)")
                                satellitesWihtoutMergerTree.append(subhalo.getIdx())
                                #write into file
                                with open(self.scratchPlotDirc + f'/satellites_without_merger_tree_{self.sim}_{self.snapshot_dic[self.snapshot][1]}.txt', 'a') as g:
                                    g.write(f"{subhalo.getIdx()}\n")
                                continue
                            f.write(f"{gg_id}\t{num_members}\t{cluster_mass}\t{join_time_info[0]}\t{join_time_info[1]}\t{join_time_info[2]}\t{join_time_info[3]}\t{join_time_info[4]}\t{join_time_info[5]}\t{join_time_info[6]}\t{join_time_info[7]}\t{join_time_info[8]}\t{join_time_info[9]}\t{join_time_info[10]}\t{join_time_info[11]}\t{join_time_info[12]}\t{join_time_info[13]}\t{join_time_info[14]}\n")
        print(f"\nFinished processing all satellites. Total processed: {len(processedSatelliteIds)}. Satellites without merger tree: {len(satellitesWihtoutMergerTree)}")

    #plot the distribution of joining redshifts for each mass bin
    def plot_joining_redshift_distribution_by_mass_bins(self, listGG : list[tuple[ListGalaxyGroup, str]]):
        joining_redshift_plotter = AstroPlotter()
        joining_redshift_fig, joining_redshift_ax = joining_redshift_plotter.create_figure()
        
        self.get_satellite_join_time(listGG)
        
        #load the join times from the files and plot the distribution of joining redshifts for each mass bin
        for listGalaxyGroup, label in listGG:
            joining_redshifts = []
            with open(self.scratchPlotDirc + f'/join_times_and_parameter_changes_{self.sim}_{self.snapshot_dic[self.snapshot][1]}_{label}.txt', 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header line
                    parts = line.split('\t')
                    if len(parts) > 3:
                        join_time_info = [float(x) for x in parts[3:]]  # Extract joining redshift and other info
                        joining_redshifts.append(join_time_info[0])
            
            joining_redshift_bins, joining_redshift_bin_edges, joining_redshift_errorbars = ListGalaxyGroup.get_histogram_bins(joining_redshifts, bins=np.arange(0, 3.5, 0.5), errorbarType='poisson')
            
            joining_redshift_plotter.scatter_plot(
                joining_redshift_bin_edges, 
                joining_redshift_bins,
                errorBars = joining_redshift_errorbars,
                ax=joining_redshift_ax, # Use the same axis for overlay
                xlabel='Joining Redshift',
                ylabel='Probability Density',
                title=f'Joining Redshift Distribution for {self.sim} Galaxy Groups in {label}',
                # ylim = (0, 0.01),
                label=f"{label}",
                output_filename=None,
                grid=True,
            )
        joining_redshift_ax.legend()
        joining_redshift_plotter.save_figure(joining_redshift_fig, self.scratchPlotDirc + f'/joining_redshift_distribution_by_mass_bins_{self.sim}.png')
    
    def plot_joining_redshift_for_all_mass_bins(self, plot_dirc : str = None):
        list_of_mass_list_galaxy_groups = [self.filtered_gt13_ls13p5_list_of_galaxy_groups, self.filtered_gt13p5_ls14_list_of_galaxy_groups, self.filtered_gt14_ls14p5_list_of_galaxy_groups, self.filtered_gt14p5_ls15_list_of_galaxy_groups, self.filtered_gt15_list_of_galaxy_groups]
        mass_bin_labels = ['$13<M_{200}<13.5$', '$13.5<M_{200}<14$', '$14<M_{200}<14.5$', '$14.5<M_{200}<15$', '$M_{200}>15$']
        list_of_mass_bin_galaxy_groups = list(zip(list_of_mass_list_galaxy_groups, mass_bin_labels))
        # self.get_satellite_join_time(list_of_mass_bin_galaxy_groups)

        self.plot_joining_redshift_distribution_by_mass_bins(list_of_mass_bin_galaxy_groups)
        
    def overlay_polar_pairwise_across_redshifts(self, listRedshiftGG:list[list[tuple[ListGalaxyGroup, str]]], plot_dirc:str = None, polar_plotter=None, polar_fig=None, polar_ax=None, mrlOrPolar:str = 'polar', plotRows:int = 0, plotCols:int = 0):
        if polar_plotter is None:
            polar_plotter = AstroPlotter()
        if polar_fig is None or polar_ax is None:
            polar_fig, polar_ax = polar_plotter.create_figure(ncols=len(listRedshiftGG[0]), nrows=1, figsize=(8, 6*len(listRedshiftGG[0]))) if plotRows == 0 or plotCols == 0 else polar_plotter.create_figure(ncols=plotCols, nrows=plotRows, figsize=(8*plotCols, 6*plotRows))
        
        for redshift_index, listGG in enumerate(listRedshiftGG):
            for i, (listGalaxyGroup, label) in enumerate(listGG):
                print(f"COMPUTING FOR redshift index: {redshift_index}, label: {label}")
                if mrlOrPolar == 'mrl':
                    pairwise_polar_differences = listGalaxyGroup.compute_mrl_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/MRL_values_{label}_{self.plotIdentifier}', rewrite=self.generalRewrite)
                else:
                    pairwise_polar_differences = listGalaxyGroup.compute_probablity_distribution_of_polar_differences(parallelize=False, tempSaveDir=f'{self.scratchDataDirc}/pairwise_polar_{label}_{self.plotIdentifier}', rewrite=self.generalRewrite)
                    pairwise_polar_bins, pairwise_polar_bin_centers, pairwise_polar_errorbars = ListGalaxyGroup.get_histogram_bins(pairwise_polar_differences, bins=np.arange(0, 185, 10), errorbarType=self.generalErrorbar)
                xlabel = 'Pairwise Polar Difference (degrees)' if mrlOrPolar == 'polar' else 'MRL Directionality of Pairwise Polar Difference'
                ylabel = 'Probability Density' if mrlOrPolar == 'polar' else 'Probability Density of MRL Directionality'
                title = f'Pairwise Polar Difference Distribution for {self.sim} Galaxy Groups in {label}' if mrlOrPolar == 'polar' else f'MRL Directionality of Pairwise Polar Difference for {self.sim} Galaxy Groups in {label}'
                
                if len(listGG) > 1:
                    print(f"choosing ax, {i}")
                    polar_ax_to_plot = polar_ax[i]
                else:
                    polar_ax_to_plot = polar_ax

                polar_plotter.scatter_plot(
                    pairwise_polar_bin_centers, 
                    pairwise_polar_bins,
                    errorBars = pairwise_polar_errorbars,
                    ax=polar_ax_to_plot, # Use the same axis for overlay
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=title,
                    # ylim = (0, 0.01),
                    label=f"{label}",
                    output_filename=None,
                    grid=True,
                )
        # if len(listGG) == 1:
        #     polar_ax.legend()
        # else:
        #     for ax in polar_ax:
        #         ax.legend()
        # polar_plotter.save_figure(polar_fig, self.scratchPlotDirc + f'/pairwise_polar_difference_distribution_across_redshifts_{self.sim}.png') if plot_dirc is None else polar_plotter.save_figure(polar_fig, plot_dirc + f'/pairwise_polar_difference_distribution_across_redshifts_{self.sim}.png')
        return polar_plotter, polar_fig, polar_ax
        
    def plot_pairwise_polar_difference_across_redshifts(self, plot_dirc:str = None):
        listRedshiftGG: list[list[tuple[ListGalaxyGroup, str]]] = []
        for snapshot_index, snapshot in enumerate(self.snapshot_dic.keys()):
            print(f"GATHERING FOR SNAPSHOT: {snapshot}")
            list_galaxy_group = self.load_galaxy_groups_for_snapshot_sim(snapshot, self.sim)
            list_galaxy_group_gt13_ls13p5, list_galaxy_group_gt13p5_ls14, list_galaxy_group_gt14_ls14p5, list_galaxy_group_gt14p5_ls15, list_galaxy_group_gt15 = self.initializeMassSubgroups(list_galaxy_group)
            group_list = []
            group_list.append((list_galaxy_group_gt13_ls13p5, f'$13<M_{{200}}<13.5$, z={self.snapshot_dic[snapshot][1]}'))
            group_list.append((list_galaxy_group_gt13p5_ls14, f'$13.5<M_{{200}}<14$, z={self.snapshot_dic[snapshot][1]}'))
            group_list.append((list_galaxy_group_gt14_ls14p5, f'$14<M_{{200}}<14.5$, z={self.snapshot_dic[snapshot][1]}'))
            group_list.append((list_galaxy_group_gt14p5_ls15, f'$14.5<M_{{200}}<15$, z={self.snapshot_dic[snapshot][1]}'))
            group_list.append((list_galaxy_group_gt15, f'$M_{{200}}>15$, z={self.snapshot_dic[snapshot][1]}'))
            group_list.append((list_galaxy_group, f'All Masses, z={self.snapshot_dic[snapshot][1]}'))
            listRedshiftGG.append(group_list)

        polar_plotter, polar_fig, polar_ax = self.overlay_polar_pairwise_across_redshifts(listRedshiftGG, plot_dirc=plot_dirc, mrlOrPolar='polar', plotRows=2, plotCols=3)
        # return polar_plotter, polar_fig, polar_ax
        polar_ax[-1].legend(loc='upper right')  # Add legend to the last subplot
        polar_plotter.save_figure(polar_fig, self.scratchPlotDirc + f'/pairwise_polar_difference_distribution_across_redshifts_{self.sim}.png') if plot_dirc is None else polar_plotter.save_figure(polar_fig, plot_dirc + f'/pairwise_polar_difference_distribution_across_redshifts_{self.sim}.png')
        
        mrl_plotter, mrl_fig, mrl_ax = self.overlay_polar_pairwise_across_redshifts(listRedshiftGG, plot_dirc=plot_dirc, mrlOrPolar='mrl', plotRows=2, plotCols=3)
        mrl_ax[-1].legend(loc='upper right')  # Add legend to the last subplot
        mrl_plotter.save_figure(mrl_fig, self.scratchPlotDirc + f'/mrl_directionality_of_pairwise_polar_difference_distribution_across_redshifts_{self.sim}.png') if plot_dirc is None else mrl_plotter.save_figure(mrl_fig, plot_dirc + f'/mrl_directionality_of_pairwise_polar_difference_distribution_across_redshifts_{self.sim}.png')     
        