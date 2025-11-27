"""
Vision utilities for file processing.
"""

import os
import glob
from loguru import logger
from ..config import File1Config
from .pdf_converter import PDFConverter


def collect_fig_files(config: File1Config, fig_files_extensions: list = [".png", ".jpg", ".jpeg", ".pdf"], base_dir=None):
    """
    Collect figure files from the specified directory.
    
    Args:
        config: File1Configuration object
        fig_files_extensions: List of file extensions to search for
        base_dir: Base directory to search in (defaults to config.save_path/workspace)
        
    Returns:
        List of figure file paths
    """
    fig_file_list = []
    if base_dir is None:
        base_dir = os.path.join(config.save_path, "workspace")
    for ext in fig_files_extensions:
        fig_file_list.extend(glob.glob(os.path.join(base_dir, "**", f"*{ext}"), recursive=True))
    logger.info(f"Found {len(fig_file_list)} results files: {fig_file_list}")
    return fig_file_list


# Legacy functions for backward compatibility
def convert_pdf_to_merged_image(pdf_file: str):
    """
    Convert all pages of a PDF file into a single merged image in a grid layout.
    
    Args:
        pdf_file: Path to the PDF file
        
    Returns:
        Tuple of (save_tmp_path, img_base64) or (None, None) if error
    """
    return PDFConverter.convert_pdf_to_merged_image(pdf_file)


def convert_pdf_to_separate_images(pdf_file: str):
    """
    Convert each page of a PDF file into separate PNG images.
    
    Args:
        pdf_file: Path to the PDF file
        
    Returns:
        Tuple of (save_tmp_path, img_base64) for the first page or (None, None) if error
    """
    return PDFConverter.convert_pdf_to_separate_images(pdf_file)


def get_fig_base64(fig_file_list, merge_pdf=False):
    """
    Convert a list of figure files (images and PDFs) to base64 format.
    
    Args:
        fig_file_list: List of figure file paths
        merge_pdf: Whether to merge PDF pages into a single image
        
    Returns:
        List of tuples (file_path, base64_data)
    """
    return PDFConverter.get_fig_base64(fig_file_list, merge_pdf)
