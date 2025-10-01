import os
import h5py
import torch
import psutil

import numpy as np

from torch import Tensor
from pathlib import Path
from torch.utils.data import Dataset


class MpcDataset(Dataset):
    def __init__(self, root_dir: os.PathLike, transform=None, target_transform=None):
        """
        Arguments:
            root_dir (PathLike): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        print("Depreciated use of dataset class: ", str(self.__class__))
        self.root_dir = Path(root_dir)
        self.dataset_length = len(list(self.root_dir.iterdir()))
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # Load data
        data_dir = self.root_dir / f"{index:06d}"
        x = np.load(data_dir / "input.npy")
        y = np.load(data_dir / "output.npy")
        
        # Apply transforms
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
            
        # Convert to tensors and transfer to correct device
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        
        return x, y


class MpcDatasetHDF5(Dataset):
    """Optimized HDF5 dataset with memory reporting."""
    
    def __init__(self, hdf5_file: Path, transform=None, target_transform=None, cache_in_memory: bool = False, chunk_cache_size: int = 32*1024*1024, verbose=False):
        """
        Arguments:
            hdf5_file (Path): Path to the HDF5 file
            transform: Optional transform for input data
            target_transform: Optional transform for target data
            cache_in_memory (bool): Load entire dataset into RAM
            chunk_cache_size (int): HDF5 chunk cache size in bytes (only used when cache_in_memory=False)
                - 32*1024*1024 (32MB): Good for random access patterns
                - 128*1024*1024 (128MB): Better for larger sequential reads
                - 256*1024*1024 (256MB): Maximum recommended for most cases
        """
        self.hdf5_file = Path(hdf5_file)
        self.transform = transform
        self.target_transform = target_transform
        self.cache_in_memory = cache_in_memory
        self.chunk_cache_size = chunk_cache_size
        
        # Store metadata
        with h5py.File(self.hdf5_file, 'r') as hf:
            self.dataset_length = hf.attrs['num_samples']
            self.num_classes = hf.attrs.get('num_classes', 3)
            self.one_hot = hf.attrs.get('one_hot', False)
            
            # Get shapes and dtypes for memory calculation
            self.input_shape = hf['inputs'].shape # type: ignore
            self.output_shape = hf['outputs'].shape # type: ignore
            self.input_dtype = hf['inputs'].dtype # type: ignore
            self.output_dtype = hf['outputs'].dtype # type: ignore
            
            # Get chunk information
            if verbose:
                input_chunks = hf['inputs'].chunks # type: ignore
                output_chunks = hf['outputs'].chunks # type: ignore
                print(f"\nHDF5 Dataset Info for {hdf5_file.name}:")
                print(f"  Input chunks: {input_chunks} (chunk = {np.prod(input_chunks) * self.input_dtype.itemsize / 1024**2:.2f} MB)") # type: ignore
                print(f"  Output chunks: {output_chunks} (chunk = {np.prod(output_chunks) * self.output_dtype.itemsize / 1024**2:.2f} MB)") # type: ignore
        
        # Initialize file handle storage
        self._file_handle = None
        self._worker_id = None
        
        # Cache entire dataset in memory if requested
        if self.cache_in_memory and not verbose:
            # Load data
            with h5py.File(self.hdf5_file, 'r') as hf:
                self.cached_inputs = np.array(hf['inputs'])
                self.cached_outputs = np.array(hf['outputs'])
        
        elif self.cache_in_memory and verbose:
            print(f"\n  Loading entire dataset into memory from {hdf5_file}...")
            
            # Calculate expected memory usage
            input_memory = np.prod(self.input_shape) * self.input_dtype.itemsize
            output_memory = np.prod(self.output_shape) * self.output_dtype.itemsize
            total_memory = input_memory + output_memory
            
            print(f"    Expected memory usage:")
            print(f"      Inputs:  {self.input_shape} × {self.input_dtype} = {input_memory / 1024**3:.3f} GB")
            print(f"      Outputs: {self.output_shape} × {self.output_dtype} = {output_memory / 1024**3:.3f} GB")
            print(f"      Total:   {total_memory / 1024**3:.3f} GB")
            
            # Track actual memory usage
            process = psutil.Process()
            mem_before = process.memory_info().rss
            
            # Load data
            with h5py.File(self.hdf5_file, 'r') as hf:
                self.cached_inputs = np.array(hf['inputs'])
                self.cached_outputs = np.array(hf['outputs'])
            
            # Report actual memory used
            mem_after = process.memory_info().rss
            actual_memory = mem_after - mem_before
            
            print(f"\n  Actual memory allocated: {actual_memory / 1024**3:.3f} GB")
            print(f"  Loaded {len(self.cached_inputs):,} samples into memory")
            
            # Report compression ratio if different from expected
            if abs(actual_memory - total_memory) > 0.01 * total_memory:
                compression_ratio = total_memory / actual_memory
                print(f"  Effective compression ratio: {compression_ratio:.2f}x")
        
        elif verbose:
            print(f"\n  Using chunk cache: {chunk_cache_size / 1024**2:.1f} MB")
            print(f"  Note: Larger cache improves random access performance")
        
        if verbose: print("\n")
    
    
    def get_memory_usage(self):
        """Return current memory usage of cached data."""
        if not self.cache_in_memory:
            return 0
        
        input_mem = self.cached_inputs.nbytes if hasattr(self, 'cached_inputs') else 0
        output_mem = self.cached_outputs.nbytes if hasattr(self, 'cached_outputs') else 0
        return input_mem + output_mem
    
    
    def _get_file_handle(self):
        """Get file handle with optimized cache settings."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        
        if self._worker_id != worker_id or self._file_handle is None:
            if self._file_handle is not None:
                self._file_handle.close()
            
            # Open with optimized cache settings
            # These parameters control how HDF5 caches chunks in memory
            self._file_handle = h5py.File(
                self.hdf5_file, 'r',
                rdcc_nbytes=self.chunk_cache_size,  # Total cache size
                rdcc_w0=0.75,  # Preemption policy (0-1, lower = LRU)
                rdcc_nslots=10007  # Hash table size (prime number)
            )
            self._worker_id = worker_id
            
        return self._file_handle
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_in_memory:
            x = self.cached_inputs[index]
            y = self.cached_outputs[index]
        else:
            hf = self._get_file_handle()
            x = hf['inputs'][index] # type: ignore
            y = hf['outputs'][index] # type: ignore
            x = np.array(x)
            y = np.array(y)
        
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32 if self.one_hot else torch.long)
        
        return x, y
    
    def __del__(self):
        if self._file_handle is not None:
            self._file_handle.close()