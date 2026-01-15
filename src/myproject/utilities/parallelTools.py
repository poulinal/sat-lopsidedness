"""
Parallel processing utilities for computationally intensive operations.
"""

import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from typing import Callable, List, Any, Optional
import sys


def parallel_map(func: Callable, items: List[Any], n_processes: Optional[int] = None, 
                chunk_size: Optional[int] = None, show_progress: bool = True) -> List[Any]:
    """
    Apply a function to items in parallel using multiprocessing.
    
    Parameters
    ----------
    func : Callable
        Function to apply to each item
    items : List
        List of items to process
    n_processes : int, optional
        Number of processes (default: CPU count - 1)
    chunk_size : int, optional
        Chunk size for processing (default: auto)
    show_progress : bool
        Show progress messages
        
    Returns
    -------
    List
        Results from applying func to each item
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    if chunk_size is None:
        chunk_size = max(1, len(items) // (n_processes * 4))
    
    if show_progress:
        print(f"Processing {len(items)} items with {n_processes} processes...")
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(func, items, chunksize=chunk_size)
    
    if show_progress:
        print(f"Completed processing {len(items)} items.")
    
    return results


def parallel_starmap(func: Callable, items: List[tuple], n_processes: Optional[int] = None,
                    chunk_size: Optional[int] = None, show_progress: bool = True) -> List[Any]:
    """
    Apply a function to tuples of arguments in parallel.
    
    Parameters
    ----------
    func : Callable
        Function to apply (should accept unpacked tuple as args)
    items : List[tuple]
        List of argument tuples
    n_processes : int, optional
        Number of processes (default: CPU count - 1)
    chunk_size : int, optional
        Chunk size for processing (default: auto)
    show_progress : bool
        Show progress messages
        
    Returns
    -------
    List
        Results from applying func to each argument tuple
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    if chunk_size is None:
        chunk_size = max(1, len(items) // (n_processes * 4))
    
    if show_progress:
        print(f"Processing {len(items)} items with {n_processes} processes...")
    
    with Pool(processes=n_processes) as pool:
        results = pool.starmap(func, items, chunksize=chunk_size)
    
    if show_progress:
        print(f"Completed processing {len(items)} items.")
    
    return results


class ProgressCounter:
    """Shared counter for tracking progress across processes."""
    
    def __init__(self, total: int):
        self.total = total
        self.counter = mp.Value('i', 0)
        self.lock = mp.Lock()
    
    def increment(self):
        """Increment counter and print progress."""
        with self.lock:
            self.counter.value += 1
            if self.counter.value % max(1, self.total // 100) == 0 or self.counter.value == self.total:
                percent = (self.counter.value / self.total) * 100
                print(f"\rProgress: {self.counter.value}/{self.total} ({percent:.1f}%)", end='', flush=True)
                if self.counter.value == self.total:
                    print()  # New line at end


def get_optimal_processes(n_items: int, min_items_per_process: int = 10) -> int:
    """
    Determine optimal number of processes based on workload.
    
    Parameters
    ----------
    n_items : int
        Number of items to process
    min_items_per_process : int
        Minimum items per process to avoid overhead
        
    Returns
    -------
    int
        Optimal number of processes
    """
    max_processes = mp.cpu_count() - 1
    optimal = min(max_processes, n_items // min_items_per_process)
    return max(1, optimal)
