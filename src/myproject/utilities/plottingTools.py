"""
Plotting tools for publication-ready astrophysics figures.

This module provides a unified interface for creating matplotlib plots
with consistent styling suitable for academic publications.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from typing import Optional, Tuple, Union, List


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
    
    def __init__(self, style: str = 'publication', context: str = 'paper'):
        """
        Initialize the AstroPlotter with specified style.
        
        Parameters
        ----------
        style : str, optional
            Style preset ('publication', 'presentation', 'poster')
        context : str, optional
            Context for sizing ('paper', 'notebook', 'talk', 'poster')
        """
        self.style = style
        self.context = context
        self._setup_style()
        
    def _setup_style(self):
        """Configure matplotlib settings for publication quality."""
        # Font settings
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        
        # Figure settings based on context
        if self.context == 'paper':
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.labelsize'] = 11
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9
            plt.rcParams['legend.fontsize'] = 9
            plt.rcParams['figure.figsize'] = (6, 4.5)
        elif self.context == 'presentation':
            plt.rcParams['font.size'] = 14
            plt.rcParams['axes.labelsize'] = 16
            plt.rcParams['axes.titlesize'] = 18
            plt.rcParams['xtick.labelsize'] = 13
            plt.rcParams['ytick.labelsize'] = 13
            plt.rcParams['legend.fontsize'] = 13
            plt.rcParams['figure.figsize'] = (10, 7.5)
        elif self.context == 'poster':
            plt.rcParams['font.size'] = 18
            plt.rcParams['axes.labelsize'] = 22
            plt.rcParams['axes.titlesize'] = 24
            plt.rcParams['xtick.labelsize'] = 18
            plt.rcParams['ytick.labelsize'] = 18
            plt.rcParams['legend.fontsize'] = 18
            plt.rcParams['figure.figsize'] = (12, 9)
        
        # Line and marker settings
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['lines.markersize'] = 6
        plt.rcParams['patch.linewidth'] = 0.5
        
        # Axes settings
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.axisbelow'] = True
        plt.rcParams['axes.labelpad'] = 4.0
        
        # Tick settings
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.minor.size'] = 3
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.minor.size'] = 3
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['xtick.minor.width'] = 0.8
        plt.rcParams['ytick.major.width'] = 1.0
        plt.rcParams['ytick.minor.width'] = 0.8
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['xtick.minor.visible'] = True
        plt.rcParams['ytick.minor.visible'] = True
        
        # Legend settings
        plt.rcParams['legend.frameon'] = False
        plt.rcParams['legend.numpoints'] = 1
        plt.rcParams['legend.scatterpoints'] = 1
        
        # Save settings
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.05
        
    def create_figure(self, nrows: int = 1, ncols: int = 1, 
                     figsize: Optional[Tuple[float, float]] = None,
                     **kwargs) -> Tuple[Figure, Union[Axes, np.ndarray]]:
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
            figsize = plt.rcParams['figure.figsize']
        
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        return fig, ax
    
    def scatter_plot(self, x: np.ndarray, y: np.ndarray, 
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
                    colorbar: bool = True,
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
        **kwargs
            Additional arguments for scatter
            
        Returns
        -------
        fig, ax
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.figure
        
        # Handle color normalization
        norm = None
        if c is not None and clog:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        elif c is not None:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Create scatter plot
        sc = ax.scatter(x, y, c=c, cmap=cmap, alpha=alpha, s=s, 
                       norm=norm, **kwargs)
        
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
        
        # Colorbar
        if c is not None and colorbar:
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            if clabel:
                cbar.set_label(clabel)
        
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
                 histtype: str = 'step',
                 linewidth: float = 2,
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
        
        # Create histogram
        ax.hist(data, bins=bins, density=density, cumulative=cumulative,
               histtype=histtype, linewidth=linewidth, label=label, **kwargs)
        
        # Set scales
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        
        # Labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ylabel_text = ylabel
        else:
            ylabel_text = 'Density' if density else 'Count'
        ax.set_ylabel(ylabel_text)
        if title:
            ax.set_title(title)
        
        # Legend
        if label:
            ax.legend()
        
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
            ax.set_title(title)
        
        # Colorbar
        if colorbar:
            cbar = plt.colorbar(cs, ax=ax, pad=0.02)
            if clabel:
                cbar.set_label(clabel)
        
        return fig, ax
    
    def save_figure(self, fig: Figure, filename: str, 
                   dpi: int = 300, 
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
        fig.savefig(filename, dpi=dpi, format=format, 
                   transparent=transparent, **kwargs)
        print(f"Figure saved to: {filename}")
    
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
            Location ('upper right', 'lower left', etc.)
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
            'center': (0.5, 0.5)
        }
        
        xy = loc_dict.get(loc, (0.95, 0.95))
        ha = 'right' if 'right' in loc else 'left'
        va = 'top' if 'upper' in loc else 'bottom'
        
        if loc == 'center':
            ha, va = 'center', 'center'
        
        ax.text(xy[0], xy[1], text, transform=ax.transAxes,
               fontsize=fontsize, bbox=props, ha=ha, va=va)
