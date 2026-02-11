#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directly call RPG internal functions for protein digestion, and output only Protein and Sequence columns.
Support filtering by peptide length.
requires Rapid Peptides Generator (RPG) package: pip install rpg==2.0.5

Input: FASTA file(s) containing protein sequences
Output: TSV file(s) with two columns: Protein and Peptide
"""

import sys
import os
import time
import logging
from pathlib import Path
from contextlib import contextmanager
import gc
from functools import lru_cache

try:
    from rpg import core
    from rpg import digest
    from rpg import enzyme
    from rpg import RapidPeptidesGenerator as rpg_main
    from rpg.enzymes_definition import AVAILABLE_ENZYMES
    sys.path.insert(0, str(Path.home()))  # Home path
    try:
        from rpg_user import AVAILABLE_ENZYMES_USER
        ALL_ENZYMES = AVAILABLE_ENZYMES + AVAILABLE_ENZYMES_USER
    except ImportError:
        ALL_ENZYMES = AVAILABLE_ENZYMES
except ImportError:
    print("Error: Could not import RPG package. Make sure it's installed: pip install rpg==2.0.5", file=sys.stderr)
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@contextmanager
def timer(description="Execution time"):
    """Context manager for timing operations"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description}: {elapsed:.2f} seconds")

@lru_cache(maxsize=1024)
def process_header(header, short_header=True):
    """Process and cache header transformation for better performance"""
    if short_header and " " in header:
        return header.split(" ")[0]
    return header

def filter_and_write_results(all_seq_digested, output_file, min_length, max_length, short_header=True):
    """Filter peptides by length and write directly to output in one pass
    
    Combines filtering and output to minimize memory usage and intermediate objects
    """
    total_peptides = 0
    total_original = 0
    buffer_size = 5000  # Number of lines to buffer before writing
    buffer = ["Protein\tPeptide\n"]  # Initialize with header
    
    # Open file early if specified to avoid reopening for each write operation
    out_file = None if output_file is None else open(output_file, 'w')
    
    try:
        for one_seq in all_seq_digested:
            for one_enz_res in one_seq:
                for peptide in one_enz_res.peptides:
                    total_original += 1
                    
                    # Filter by peptide length
                    if min_length <= peptide.size <= max_length:
                        # Process header once and cache results
                        header = process_header(peptide.header, short_header)
                        
                        # Add to buffer
                        buffer.append(f"{header}\t{peptide.sequence}\n")
                        total_peptides += 1
                        
                        # Write buffer to file when it reaches buffer_size
                        if len(buffer) >= buffer_size:
                            if output_file:
                                out_file.writelines(buffer)
                            else:
                                sys.stdout.writelines(buffer)
                            buffer = []  # Clear buffer
                            # Trigger garbage collection periodically
                            if total_peptides % (buffer_size * 10) == 0:
                                gc.collect()
        
        # Write any remaining lines in buffer
        if buffer:
            if output_file:
                out_file.writelines(buffer)
            else:
                sys.stdout.writelines(buffer)
    
    finally:
        # Make sure to close the file if opened
        if out_file:
            out_file.close()
    
    # Report retention rate
    if total_original > 0:
        retention_rate = (total_peptides / total_original) * 100
        logger.info(f"Length filter: kept {total_peptides} out of {total_original} peptides (retention rate: {retention_rate:.2f}%)")
    
    return total_peptides

def is_fasta_file(file_path):
    """Check if a file is in FASTA format by examining its first line"""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            # FASTA files start with '>'
            return first_line.startswith('>')
    except Exception:
        return False

def process_directory(input_dir, output_dir, enzyme_id, min_length=7, max_length=30, 
                     max_num_miscleavages=0, processes=None, short_header=True, verbose=True,
                     skip_existing=False):
    """
    Process all FASTA files in a directory and save results to another directory
    
    Args:
        input_dir (str): Input directory containing FASTA files
        output_dir (str): Output directory for TSV results
        enzyme_id (str): Enzyme ID or name
        min_length (int, optional): Minimum peptide length, default 7
        max_length (int, optional): Maximum peptide length, default 30
        max_num_miscleavages (int, optional): Maximum number of miscleavage sites, default 0 (disabled)
        processes (int, optional): Number of parallel processes, default None (use all available CPUs)
        short_header (bool, optional): Whether to keep only the first field of header (split by space), default True
        verbose (bool, optional): Whether to print processing information, default True
        skip_existing (bool, optional): Whether to skip files that already have output files, default False
        
    Returns:
        dict: Dictionary with filenames as keys and peptide counts as values
    """
    # Check if input directory exists
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        error_msg = f"Error: Input directory {input_dir} does not exist or is not a directory"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.info(f"Creating output directory: {output_dir}")
        output_path.mkdir(parents=True, exist_ok=True)
    elif not output_path.is_dir():
        error_msg = f"Error: Output path {output_dir} exists but is not a directory"
        logger.error(error_msg)
        raise NotADirectoryError(error_msg)
    
    results = {}
    total_files = 0
    processed_files = 0
    skipped_files = 0
    
    # Find all files in the input directory
    all_files = list(input_path.glob('*'))
    total_files = len(all_files)
    
    logger.info(f"Found {total_files} files in {input_dir}")
    
    # Process each file
    for file_path in all_files:
        if not file_path.is_file():
            logger.debug(f"Skipping non-file: {file_path}")
            continue
            
        # Check if it's a FASTA file
        if not is_fasta_file(file_path):
            logger.warning(f"Skipping non-FASTA file: {file_path}")
            skipped_files += 1
            continue
        
        # Determine output file path
        output_file = output_path / f"{file_path.stem}.tsv"
        
        # Skip if output file exists and skip_existing is True
        if skip_existing and output_file.exists():
            logger.info(f"Skipping existing output file: {output_file}")
            skipped_files += 1
            continue
        
        try:
            logger.info(f"Processing file {processed_files+1}/{total_files}: {file_path.name}")
            
            # Process the file
            peptide_count = process_fasta(
                input_file=str(file_path),
                enzyme_id=enzyme_id,
                output_file=str(output_file),
                min_length=min_length,
                max_length=max_length,
                max_num_miscleavages=max_num_miscleavages,
                processes=processes,
                short_header=short_header,
                verbose=verbose
            )
            
            # Store result
            results[file_path.name] = peptide_count
            processed_files += 1
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            skipped_files += 1
    
    # Report summary
    logger.info(f"Directory processing complete: {processed_files} files processed, {skipped_files} files skipped")
    return results

def process_fasta(input_file, enzyme_id, output_file=None, min_length=7, max_length=30, 
                 max_num_miscleavages=0, processes=None, short_header=True, verbose=True):
    """
    Process FASTA file, perform enzyme digestion and filter by peptide length
    
    Args:
        input_file (str): Input FASTA file path
        enzyme_id (str): Enzyme ID or name
        output_file (str, optional): Output TSV file path, if not specified, output to stdout
        min_length (int, optional): Minimum peptide length, default 7
        max_length (int, optional): Maximum peptide length, default 30
        max_num_miscleavages (int, optional): Maximum number of miscleavage sites, default 0 (disabled)
        processes (int, optional): Number of parallel processes, default None (use all available CPUs)
        short_header (bool, optional): Whether to keep only the first field of header (split by space), default True
        verbose (bool, optional): Whether to print processing information, default True
        
    Returns:
        int: Number of peptides generated
    """
    # Set log level
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    # Check input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        error_msg = f"Error: Input file {input_file} does not exist"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Optimize process count
    cpu_count = os.cpu_count() or 1
    if processes is None:
        processes = cpu_count
    
    # Apply Windows-specific limit
    if sys.platform == 'win32' and processes > 60:
        logger.warning(f"Windows system detected: Process count {processes} exceeds handle limit, adjusted to 60")
        processes = 60
    elif processes > cpu_count:
        logger.warning(f"Warning: Process count {processes} exceeds system CPU cores {cpu_count}, adjusted")
        processes = cpu_count
    
    logger.info(f"Using {processes} CPU cores")
    
    # Set mass type (use average by default)
    aa_mass = core.AA_MASS_AVERAGE
    water_mass = core.WATER_MASS
    
    # Set pKa values (use IPC2 by default)
    aa_pka = core.AA_PKA_IPC_2
    
    # Create enzyme object
    try:
        with timer("Creating enzyme object"):
            # Check if enzyme ID is valid
            rpg_main.restricted_enzyme_id(enzyme_id)
            
            # Get enzyme to use
            enzymes_to_use = rpg_main.create_enzymes_to_use([enzyme_id], [0])
            
            if not enzymes_to_use:
                error_msg = f"Error: Could not create enzyme {enzyme_id}"
                logger.error(error_msg)
                raise ValueError(error_msg)
    except Exception as e:
        logger.error(f"Error creating enzyme: {str(e)}")
        raise
    
    try:
        logger.info(f"Processing file: {input_file}")
        logger.info(f"Using enzyme: {enzymes_to_use[0].name}")
        logger.info(f"Parallel processes: {processes}")
        logger.info(f"Peptide length filter: {min_length}-{max_length}")
        
        # Perform digestion
        with timer("Digestion process..."):
            all_seq_digested = digest.digest_from_input(
                input_file, "file", enzymes_to_use, "sequential", 
                aa_pka, aa_mass, water_mass, processes
            )
        
        # Calculate theoretical miscleavage sites if needed
        if max_num_miscleavages > 0:
            with timer(f"Calculating theoretical miscleavage sites (max: {max_num_miscleavages})"):
                enzymes_dict = {enzymes_to_use[0].name: max_num_miscleavages}
                digest.theoretical_peptides(all_seq_digested, enzymes_dict)
        
        # Combined filter and output in one pass to reduce memory usage
        with timer("Filtering and writing results"):
            total_peptides = filter_and_write_results(
                all_seq_digested, output_file, min_length, max_length, short_header
            )
        
        # Clean up memory
        del all_seq_digested
        gc.collect()
        
        # Output statistics
        if output_file:
            logger.info(f"Processing complete. Generated {total_peptides} peptides, saved to {output_file}")
        else:
            logger.info(f"Processing complete. Generated {total_peptides} peptides.")
        
        return total_peptides
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Example usage: process a single file
        """
        with timer("Total processing time"):
            peptide_count = process_fasta(
                input_file="C:/Users/Qing/Desktop/original_db/MGYG000000002.faa",
                enzyme_id="42",
                output_file="filtered_results.tsv",
                min_length=7,
                max_length=30,
                max_num_miscleavages=2,
                processes=None,  # Use all available CPUs
                short_header=True,
                verbose=True
            )
        logger.info(f"Generated {peptide_count} peptides")
        """
        
        # Example usage: process a directory
        with timer("Total directory processing time"):
            results = process_directory(
                input_dir="test_data/mix24x/download_genome",
                output_dir="test_data/mix24x/download_genome_digested",
                enzyme_id="42",
                min_length=7,
                max_length=30,
                max_num_miscleavages=2,
                processes=None,  # Use all available CPUs
                short_header=True,
                verbose=True,
                skip_existing=True
            )
        
        total_peptides = sum(results.values())
        logger.info(f"Total peptides generated across all files: {total_peptides}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)