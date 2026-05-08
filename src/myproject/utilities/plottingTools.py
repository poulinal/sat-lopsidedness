"""
Plotting tools for publication-ready astrophysics figures.

This module provides a unified interface for creating matplotlib plots
with consistent styling suitable for academic publications.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import importlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.collections import LineCollection
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Union, List
from scipy.interpolate import UnivariateSpline


class AstroPlotter:
    """
    A plotting utility class for creating publication-ready astrophysics plots.
    
    Features:
    - Consistent publication-quality styling
    - Support for common astrophysics plot types
    - Proper handling of units and labels
    - Colorbar utilities
    - Multi-panel figure support
    """
    
    def __init__(self, style: str = 'cms', context: str = 'paper'):
        """
        Initialize the AstroPlotter with specified style.
        
        Parameters
        ----------
        style : str, optional
            Style preset ('publication', 'presentation', 'poster', 'cms')
        context : str, optional
            Context for sizing ('paper', 'notebook', 'talk', 'poster')
        """
        self.style = style
        self.context = context
        # Build rcparams dict but do not mutate global rcParams
        self._rcparams = {}
        self._setup_style()
        # small colorblind-friendly palette
        self._palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
        
    def _setup_style(self):
        """Configure matplotlib settings for publication quality."""
        # Build a local rcparams dict rather than mutating global rcParams
        rc = {}

        if self.style.lower() == 'cms':
            hep_spec = importlib.util.find_spec('mplhep')
            if hep_spec is None:
                warnings.warn(
                    "style='cms' requested but mplhep is not installed. Falling back to default publication style.",
                    RuntimeWarning,
                )
            else:
                hep = importlib.import_module('mplhep')
                rc.update(dict(hep.style.CMS))

        # Publication defaults when not using CMS (or when CMS fallback is active)
        if not rc:
            rc['font.family'] = 'serif'
            rc['font.serif'] = ['Times New Roman', 'DejaVu Serif']
            rc['mathtext.fontset'] = 'dejavuserif'

        # Figure settings based on context
        if self.context == 'paper':
            rc['font.size'] = 10
            rc['axes.labelsize'] = 11
            rc['axes.titlesize'] = 12
            rc['xtick.labelsize'] = 9
            rc['ytick.labelsize'] = 9
            rc['legend.fontsize'] = 9
            rc['figure.figsize'] = (6, 4.5)
        elif self.context == 'presentation':
            rc['font.size'] = 14
            rc['axes.labelsize'] = 16
            rc['axes.titlesize'] = 18
            rc['xtick.labelsize'] = 13
            rc['ytick.labelsize'] = 13
            rc['legend.fontsize'] = 13
            rc['figure.figsize'] = (10, 7.5)
        elif self.context == 'poster':
            rc['font.size'] = 18
            rc['axes.labelsize'] = 22
            rc['axes.titlesize'] = 24
            rc['xtick.labelsize'] = 18
            rc['ytick.labelsize'] = 18
            rc['legend.fontsize'] = 18
            rc['figure.figsize'] = (12, 9)

        # Line and marker settings
        rc['lines.linewidth'] = 1.5
        rc['lines.markersize'] = 6
        rc['patch.linewidth'] = 0.5

        # Axes settings
        rc['axes.linewidth'] = 1.0
        rc['axes.grid'] = False
        rc['axes.axisbelow'] = True
        rc['axes.labelpad'] = 4.0

        # Tick settings
        rc['xtick.direction'] = 'in'
        rc['ytick.direction'] = 'in'
        rc['xtick.major.size'] = 5
        rc['xtick.minor.size'] = 3
        rc['ytick.major.size'] = 5
        rc['ytick.minor.size'] = 3
        rc['xtick.major.width'] = 1.0
        rc['xtick.minor.width'] = 0.8
        rc['ytick.major.width'] = 1.0
        rc['ytick.minor.width'] = 0.8
        rc['xtick.top'] = True
        rc['ytick.right'] = True
        rc['xtick.minor.visible'] = True
        rc['ytick.minor.visible'] = True

        # Legend settings
        rc['legend.frameon'] = False
        rc['legend.numpoints'] = 1
        rc['legend.scatterpoints'] = 1

        # Save settings
        rc['savefig.dpi'] = 300
        rc['savefig.bbox'] = 'tight'
        rc['savefig.pad_inches'] = 0.05

        self._rcparams = rc
        
    def create_figure(self, nrows: int = 1, ncols: int = 1, 
                     figsize: Optional[Tuple[float, float]] = None,
                     constrained_layout: Optional[bool] = True,
                     **kwargs) -> Tuple[Figure, Union[Axes, npt.NDArray[np.object_]]]:
        """
        Create a figure with subplots.
        
        Parameters
        ----------
        nrows : int
            Number of rows
        ncols : int
            Number of columns
        figsize : tuple, optional
            Figure size (width, height) in inches
        **kwargs
            Additional arguments passed to plt.subplots
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes or array of Axes
        """
        if figsize is None:
            figsize = self._rcparams.get('figure.figsize', plt.rcParams['figure.figsize'])

        # Apply local style context per-figure so global rcParams are not mutated
        with mpl.rc_context(self._rcparams):
            fig, ax = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=constrained_layout, **kwargs)
        
        # Ensure figure is set as current matplotlib figure for Jupyter auto-display
        plt.sca(ax if isinstance(ax, Axes) else ax.flat[0] if hasattr(ax, 'flat') else ax)

        return fig, ax
    
    def scatter_plot(self, x: np.ndarray, y: np.ndarray, 
                    errorBars: Optional[np.ndarray] = None,
                    c: Optional[np.ndarray] = None,
                    ax: Optional[Axes] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    clabel: Optional[str] = None,
                    title: Optional[str] = None,
                    xlog: bool = False,
                    ylog: bool = False,
                    clog: bool = False,
                    cmap: str = 'viridis',
                    alpha: float = 0.7,
                    s: float = 20,
                    vmin: Optional[float] = None,
                    vmax: Optional[float] = None,
                    xlim: Optional[Tuple[float, float]] = None,
                    ylim: Optional[Tuple[float, float]] = None,
                    colorbar: bool = False,
                    label: Optional[str] = None,
                    include_legend: bool = False,
                    output_filename: Optional[str] = None,
                    grid : bool = False,
                    overlay_color: Optional[str] = None,  # New parameter for overlay color
                    spline_curvature: bool = False,
                    spline_smoothing: Optional[float] = None,  # New parameter for spline smoothing
                    **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a scatter plot.
        
        Parameters
        ----------
        x, y : array-like
            Data coordinates
        c : array-like, optional
            Color values
        ax : Axes, optional
            Axes to plot on
        xlabel, ylabel, clabel : str, optional
            Axis labels
        title : str, optional
            Plot title
        xlog, ylog, clog : bool
            Use logarithmic scale
        cmap : str
            Colormap name
        alpha : float
            Point transparency
        s : float
            Marker size
        vmin, vmax : float, optional
            Color scale limits
        colorbar : bool
            Add colorbar
        include_legend: bool = False,
        **kwargs
            Additional arguments for scatter
            
        Returns
        -------
        fig, ax
        """
        if ax is None:
            print("warning ax is none")
            fig, ax = self.create_figure()
        else:
            # If ax is a numpy array (from subplots), select the first axes
            if isinstance(ax, np.ndarray):
                print("Warning: ax is a numpy array, using the first axes in the array.")
                ax = ax.flat[0]
            fig = ax.figure
        # Determine coloring strategy:
        # - if `c` provided and is array-like: use it with cmap/norm
        # - if `c` is a single color or overlay_color provided: use as solid color
        # detect whether a drawstyle was requested so we can avoid forcing a color (allow color cycling)
        drawstyle_present = 'drawstyle' in kwargs

        scatter_kwargs = dict(alpha=alpha, s=s)

        if c is not None:
            # check if c is a scalar color
            is_scalar_color = False
            try:
                mpl.colors.to_rgba(c)
                is_scalar_color = True
            except Exception:
                is_scalar_color = False

            if is_scalar_color and overlay_color is None:
                scatter_kwargs['color'] = c
                norm = None
            else:
                # treat c as array of values to map
                norm = LogNorm(vmin=vmin, vmax=vmax) if clog else Normalize(vmin=vmin, vmax=vmax)
                scatter_kwargs['c'] = c
                scatter_kwargs['cmap'] = cmap
                scatter_kwargs['norm'] = norm
        else:
            # no c provided -> if drawstyle is requested, prefer leaving color unset so Matplotlib cycles colors
            if drawstyle_present:
                if overlay_color is not None:
                    scatter_kwargs['color'] = overlay_color
                # else: leave color unset to allow axes color cycling
            else:
                scatter_kwargs['color'] = overlay_color if overlay_color is not None else None # else self._palette[0] #use self._palette[0] to force same color

        # Extract drawstyle (for step-style line plotting) so it isn't passed to PathCollection
        drawstyle = None
        if 'drawstyle' in kwargs:
            drawstyle = kwargs.pop('drawstyle')

        # merge any extra kwargs (marker, edgecolors, etc.)
        scatter_kwargs.update(kwargs)

        artist = None
        # If a drawstyle is requested, use a line plot (supports drawstyle).
        # If `c` is an array-like, construct a LineCollection so segments can be colored by `c`.
        if drawstyle is not None:
            line_kwargs = {}
            if 'color' in scatter_kwargs:
                line_kwargs['color'] = scatter_kwargs.get('color')
            line_kwargs['alpha'] = scatter_kwargs.get('alpha', alpha)
            line_kwargs['linewidth'] = scatter_kwargs.get('linewidth', 2)
            marker = scatter_kwargs.get('marker', None)
            if marker is not None:
                line_kwargs['marker'] = marker

            # If c is an array-like (per-point values), use LineCollection to color segments
            cvals = scatter_kwargs.get('c', None)
            is_c_array = False
            if cvals is not None:
                try:
                    arr = np.asarray(cvals)
                    if arr.ndim >= 1 and arr.size > 1:
                        is_c_array = True
                except Exception:
                    is_c_array = False

            if is_c_array:
                # build segments between consecutive points
                pts = np.array([x, y]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                # create LineCollection and map c to segments (length N-1)
                seg_c = np.asarray(cvals)
                if seg_c.size == arr.size and seg_c.size == len(x):
                    seg_c = seg_c[:-1]
                norm = LogNorm(vmin=vmin, vmax=vmax) if clog else Normalize(vmin=vmin, vmax=vmax)
                lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=line_kwargs.get('linewidth', 2), alpha=line_kwargs.get('alpha', 1.0))
                lc.set_array(seg_c)
                ax.add_collection(lc)
                artist = lc
                # fallback: also respect drawstyle by plotting invisible step line for legend/steps appearance
                try:
                    ax.plot(x, y, drawstyle=drawstyle, color=line_kwargs.get('color', None), alpha=0.0)
                except Exception:
                    pass
            else:
                # plot as a line supporting drawstyle (e.g., 'steps-mid')
                (ln,) = ax.plot(x, y, drawstyle=drawstyle, label=label if not spline_curvature else None, **line_kwargs)
                artist = ln
        else:
            # Create scatter plot
            sc = ax.scatter(x, y, **scatter_kwargs)
            artist = sc

        if spline_curvature:
            # Sort data by x for spline fitting
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            # Fit spline and plot
            spline = UnivariateSpline(x_sorted, y_sorted, k=2, s=spline_smoothing if spline_smoothing is not None else 5)
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            # ax.plot(x_smooth, spline(x_smooth),'k--', alpha=0.5, linewidth=3, c=scatter_color, label=label)
            ax.plot(x_smooth, spline(x_smooth),'--', alpha=0.5, linewidth=3, c=scatter_kwargs.get('color', None), label=label)
        
        if errorBars is not None:
            # Determine a reliable color for the error bars that matches the scatter points.
            ecolor = None
            # 1) If user explicitly set a color, use it
            if 'color' in scatter_kwargs and scatter_kwargs.get('color') is not None:
                ecolor = scatter_kwargs.get('color')

            # 2) If artist is a PathCollection (scatter), inspect its facecolors
            if ecolor is None and isinstance(artist, mpl.collections.PathCollection):
                try:
                    fc = artist.get_facecolors()
                    if fc is not None and len(fc) > 0:
                        # fc is an (N,4) array; use the first entry
                        ecolor = fc[0]
                except Exception:
                    ecolor = None

            # 3) If artist is a Line2D (from ax.plot), use its color
            if ecolor is None and hasattr(artist, 'get_color'):
                try:
                    ecolor = artist.get_color()
                except Exception:
                    ecolor = None

            # 4) Fallback: ask the axes for the next color in the cycle (useful when color was unset)
            if ecolor is None:
                try:
                    prop_cycler = ax._get_lines.get_next_color if hasattr(ax, '_get_lines') else None
                except Exception:
                    prop_cycler = None
                try:
                    # Try retrieving last plotted artist color from the axes
                    last_color = None
                    if hasattr(artist, 'get_facecolor'):
                        fc = artist.get_facecolor()
                        if isinstance(fc, np.ndarray) and fc.size:
                            last_color = fc[0]
                    if last_color is not None:
                        ecolor = last_color
                except Exception:
                    pass

            ax.errorbar(x, y, yerr=errorBars, fmt='none', ecolor=ecolor, alpha=0.5, capsize=2)
            
        # Set scales
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        
        # Set limits
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
            
        # Labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            if len(title) > 50:  # If title is long, create a new line at the next space
                split_idx = title.find(' ', 50)
                if split_idx != -1:
                    title = title[:split_idx] + '\n' + title[split_idx + 1:]
            ax.set_title(title)
        if label and not spline_curvature:
            try:
                artist.set_label(label)
            except Exception:
                pass
            
        if grid:
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            
        # Colorbar
        if c is not None and colorbar and isinstance(artist, mpl.collections.PathCollection):
            cbar = plt.colorbar(artist, ax=ax, pad=0.02)
            if clabel:
                cbar.set_label(clabel)
        
        if include_legend and label:
            ax.legend(loc='upper right', fontsize=8)
        
        if output_filename:
            self.save_figure(fig, output_filename)
            
        return fig, ax
    
    def line_plot(self, x: np.ndarray, y: np.ndarray,
                 ax: Optional[Axes] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 title: Optional[str] = None,
                 label: Optional[str] = None,
                 xlog: bool = False,
                 ylog: bool = False,
                 color: Optional[str] = None,
                 linestyle: str = '-',
                 linewidth: Optional[float] = None,
                 marker: Optional[str] = None,
                 **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a line plot.
        
        Parameters
        ----------
        x, y : array-like
            Data coordinates
        ax : Axes, optional
            Axes to plot on
        xlabel, ylabel : str, optional
            Axis labels
        title : str, optional
            Plot title
        label : str, optional
            Line label for legend
        xlog, ylog : bool
            Use logarithmic scale
        color : str, optional
            Line color
        linestyle : str
            Line style
        linewidth : float, optional
            Line width
        marker : str, optional
            Marker style
        **kwargs
            Additional arguments for plot
            
        Returns
        -------
        fig, ax
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.figure
        
        # Create line plot
        ax.plot(x, y, color=color, linestyle=linestyle, 
               linewidth=linewidth, marker=marker, label=label, **kwargs)
        
        # Set scales
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        
        # Labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        
        # Legend
        if label:
            ax.legend()
        
        return fig, ax
    
    def histogram(self, data: np.ndarray,
                 ax: Optional[Axes] = None,
                 bins: Union[int, np.ndarray] = 30,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 title: Optional[str] = None,
                 label: Optional[str] = None,
                 xlog: bool = False,
                 ylog: bool = False,
                 density: bool = False,
                 cumulative: bool = False,
                 percentage: bool = False,  # New option for percentage
                 histtype: str = 'step',
                 linewidth: float = 2,
                 linealpha : float = 1.0,
                 output_filename: Optional[str] = None,
                 legend: bool = False,
                 grid: bool = False,
                 **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a histogram.
        
        Parameters
        ----------
        data : array-like
            Data to histogram
        ax : Axes, optional
            Axes to plot on
        bins : int or array
            Number of bins or bin edges
        xlabel, ylabel : str, optional
            Axis labels
        title : str, optional
            Plot title
        label : str, optional
            Histogram label
        xlog, ylog : bool
            Use logarithmic scale
        density : bool
            Normalize to density
        cumulative : bool
            Plot cumulative distribution
        percentage : bool
            Normalize to percentage (y-axis as percentage of total)
        histtype : str
            Histogram type
        linewidth : float
            Line width
        **kwargs
            Additional arguments for hist
            
        Returns
        -------
        fig, ax
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.figure
        
        if percentage:
            hist_values, bin_edges = np.histogram(data, bins=bins)
            total = sum(hist_values)
            if total > 0:
                hist_values = (hist_values / total) * 100
            # Plot histogram as step
            ax.plot(
                bin_edges[:-1], hist_values, drawstyle='steps-post',
                linewidth=linewidth, label=label, alpha=linealpha, **kwargs
            )
        else:
            # Calculate histogram data
            hist_values, bin_edges, patches = ax.hist(
                data, bins=bins, density=False, cumulative=cumulative,
                histtype=histtype, linewidth=linewidth, label=label, alpha=linealpha, **kwargs
            )

        if percentage:
            ylabel_text = 'Percentage (%)'
        elif density:
            ylabel_text = 'Density'
        else:
            ylabel_text = 'Count'
        
        # Set scales
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        
        # Labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or ylabel_text)
        if title:
            if len(title) > 50:  # If title is long, create a new line for better formatting
                title = title[:50] + '\n' + title[50:]
            ax.set_title(title)
        
        # Legend
        if legend and label:
            ax.legend()
        
        # Grid
        if grid:
            ax.grid(True)
            
        if output_filename:
            self.save_figure(fig, output_filename)
            
        return fig, ax
    
    def density_map(self, data: np.ndarray,
                   ax: Optional[Axes] = None,
                   extent: Optional[List[float]] = None,
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   clabel: Optional[str] = None,
                   title: Optional[str] = None,
                   cmap: str = 'viridis',
                   log_scale: bool = False,
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   colorbar: bool = True,
                   origin: str = 'lower',
                   aspect: str = 'auto',
                   interpolation: str = 'nearest',
                   **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a 2D density map.
        
        Parameters
        ----------
        data : 2D array
            Data to display
        ax : Axes, optional
            Axes to plot on
        extent : list, optional
            [xmin, xmax, ymin, ymax]
        xlabel, ylabel, clabel : str, optional
            Axis labels
        title : str, optional
            Plot title
        cmap : str
            Colormap name
        log_scale : bool
            Use logarithmic color scale
        vmin, vmax : float, optional
            Color scale limits
        colorbar : bool
            Add colorbar
        origin : str
            Origin position
        aspect : str
            Aspect ratio
        interpolation : str
            Interpolation method
        **kwargs
            Additional arguments for imshow
            
        Returns
        -------
        fig, ax
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.figure
        
        # Handle normalization
        norm = None
        if log_scale:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Create image
        im = ax.imshow(data, cmap=cmap, norm=norm, extent=extent,
                      origin=origin, aspect=aspect, 
                      interpolation=interpolation, **kwargs)
        
        # Labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            if len(title) > 50:  # If title is long, create a new line for better formatting
                title = title[:50] + '\n' + title[50:]
            ax.set_title(title)
        
        # Colorbar
        if colorbar:
            cbar = plt.colorbar(im, ax=ax, pad=0.02)
            if clabel:
                cbar.set_label(clabel)
        
        return fig, ax
    
    def contour_plot(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    ax: Optional[Axes] = None,
                    levels: Optional[Union[int, List[float]]] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    clabel: Optional[str] = None,
                    title: Optional[str] = None,
                    filled: bool = False,
                    cmap: str = 'viridis',
                    linewidths: float = 1.5,
                    colorbar: bool = True,
                    **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a contour plot.
        
        Parameters
        ----------
        x, y : 2D arrays
            Coordinate arrays
        z : 2D array
            Data values
        ax : Axes, optional
            Axes to plot on
        levels : int or list, optional
            Contour levels
        xlabel, ylabel, clabel : str, optional
            Axis labels
        title : str, optional
            Plot title
        filled : bool
            Use filled contours
        cmap : str
            Colormap name
        linewidths : float
            Contour line width
        colorbar : bool
            Add colorbar
        **kwargs
            Additional arguments for contour/contourf
            
        Returns
        -------
        fig, ax
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.figure
        
        # Create contours
        if filled:
            cs = ax.contourf(x, y, z, levels=levels, cmap=cmap, **kwargs)
        else:
            cs = ax.contour(x, y, z, levels=levels, cmap=cmap,
                          linewidths=linewidths, **kwargs)
        
        # Labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            if len(title) > 50:  # If title is long, create a new line for better formatting
                title = title[:50] + '\n' + title[50:]
            ax.set_title(title)
        
        # Colorbar
        if colorbar:
            cbar = plt.colorbar(cs, ax=ax, pad=0.02)
            if clabel:
                cbar.set_label(clabel)
        
        return fig, ax
    
    def save_figure(self, fig: Figure, filename: str, 
                   dpi: int = 600, 
                   format: Optional[str] = None,
                   transparent: bool = False,
                   **kwargs):
        """
        Save figure with publication-quality settings.
        
        Parameters
        ----------
        fig : Figure
            Figure to save
        filename : str
            Output filename
        dpi : int
            Resolution in dots per inch
        format : str, optional
            File format (inferred from filename if not provided)
        transparent : bool
            Transparent background
        **kwargs
            Additional arguments for savefig
        """
        # Save respecting explicit extension when provided. If no extension,
        # prefer vector (PDF) and also save a PNG raster for quick previews.
        saved = []
        if filename.lower().endswith('.pdf') or filename.lower().endswith('.png') or filename.lower().endswith('.svg'):
            fig.savefig(filename, dpi=dpi, format=format, transparent=transparent, **kwargs)
            saved.append(filename)
        else:
            # save vector first
            pdf_name = filename + '.pdf'
            png_name = filename + '.png'
            fig.savefig(pdf_name, dpi=dpi, format='pdf', transparent=transparent, **kwargs)
            fig.savefig(png_name, dpi=dpi, format='png', transparent=transparent, **kwargs)
            saved.extend([pdf_name, png_name])

        # return saved filenames for caller to use or inspect
        return saved
    
    def add_text_box(self, ax: Axes, text: str, 
                    loc: str = 'upper right',
                    fontsize: Optional[int] = None,
                    **kwargs):
        """
        Add a text box to the plot.
        
        Parameters
        ----------
        ax : Axes
            Axes to add text to
        text : str
            Text content
        loc : str
            Location ('upper right', 'lower left', 'upper left', 'lower right', 'center', 'outside upper right', 'bottom center')
        fontsize : int, optional
            Font size
        **kwargs
            Additional arguments for text box properties
        """
        props = dict(boxstyle='round', facecolor='white', 
                    alpha=0.8, edgecolor='gray')
        props.update(kwargs)
        
        # Parse location
        loc_dict = {
            'upper right': (0.95, 0.95),
            'upper left': (0.05, 0.95),
            'lower right': (0.95, 0.05),
            'lower left': (0.05, 0.05),
            'center': (0.5, 0.5),
            'outside upper right': (1.05, 0.95),
            # 'bottom center': (0.5, -0.12),
            'bottom center': (0.5, -0.18),
        }
        
        xy = loc_dict.get(loc, (0.95, 0.95))
        ha = 'right' if 'right' in loc else 'left'
        va = 'top' if 'upper' in loc else 'bottom'
        
        if loc == 'center' or loc == 'bottom center':
            ha, va = 'center', 'center'

        #if text too big, decrease fontsize and add new lines for better formatting
        if len(text) > 50:
            if fontsize is None:
                fontsize = 10
            else:
                fontsize = max(fontsize - 2, 8)
            # text = text[:50] + '\n' + text[50:]
        
        ax.text(xy[0], xy[1], text, transform=ax.transAxes,
               fontsize=fontsize, bbox=props, ha=ha, va=va)

    def add_legend(self, ax: Axes, title: Optional[str] = None,
                   loc: str = 'upper right',
                   fontsize: Optional[int] = None,
                   **kwargs):
        """
        Add a legend to the plot.
        
        Parameters
        ----------
        ax : Axes
            Axes to add legend to
        title : str, optional
            Legend title
        loc : str
            Location ('upper right', 'lower left', 'upper left', 'lower right', 'center')
        fontsize : int, optional
            Font size
        **kwargs
            Additional arguments for legend properties
        """
        legend = ax.legend(loc=loc, fontsize=fontsize, title=title, **kwargs)
        if title:
            legend.get_title().set_fontsize(fontsize if fontsize else 10)
            
    def add_annotation(self, text: str, ax: Axes, xy: Tuple[float, float], 
                    xytext: Optional[Tuple[float, float]] = None,
                    arrowprops: Optional[dict] = None,
                    fontsize: Optional[int] = None,
                    **kwargs):
        """
        Add an annotation with an optional arrow.
        
        Parameters
        ----------
        ax : Axes
            Axes to add annotation to
        text : str
            Annotation text
        xy : tuple
            Point (x, y) to annotate
        xytext : tuple, optional
            Position (x, y) for the text (if different from xy)
        arrowprops : dict, optional
            Properties for the arrow (if xytext is provided)
        fontsize : int, optional
            Font size for the annotation text
        fontweight : str, optional
            Font weight for the annotation text (e.g., 'light', 'normal', 'bold')
        **kwargs
            Additional arguments for annotation properties
        """
        if fontsize is not None:
            kwargs['fontsize'] = fontsize

        # Default to a lighter font weight to reduce perceived "thickness"
        # unless the caller explicitly requests a different weight.
        if 'fontweight' not in kwargs and 'weight' not in kwargs:
            kwargs['fontweight'] = 'light'

        ax.annotate(text, xy=xy, xytext=xytext, arrowprops=arrowprops, **kwargs)