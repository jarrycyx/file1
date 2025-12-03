import os
from typing import Optional
from ..config import InclusionConfig


class FileInclusion:
    """
    File inclusion manager that handles file filtering based on blacklist or whitelist mode.
    """
    
    def __init__(self, config: Optional[InclusionConfig] = None):
        """
        Initialize the FileInclusion with configuration.
        
        Args:
            config: InclusionConfig instance. If None, default config will be used.
        """
        self.config = config or InclusionConfig()
    
    def is_included(self, file_path: str) -> bool:
        """
        Check if a file path should be included based on the current mode.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be included, False otherwise
        """
        if self.config.mode == "blacklist":
            return not self._is_blacklisted(file_path)
        elif self.config.mode == "whitelist":
            if os.path.isfile(file_path):
                return self._is_whitelisted(file_path)
            else:
                return not self._is_blacklisted(file_path)
        else:
            # Default to blacklist mode for invalid mode
            return not self._is_blacklisted(file_path)
    
    def _is_blacklisted(self, file_path: str) -> bool:
        """
        Check if a file path is blacklisted.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file path is blacklisted, False otherwise
        """
        # Check if any blacklist pattern is in the file path
        return any(pattern in file_path for pattern in self.config.path_blacklist)
    
    def _is_whitelisted(self, file_path: str) -> bool:
        """
        Check if a file path is whitelisted.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file path is whitelisted, False otherwise
        """
        # Check if the file extension is in the whitelist
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.config.ext_whitelist
    
    def is_no_child_file(self, file_path: str) -> bool:
        """
        Check if a file path is a no child file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file path is a no child file, False otherwise
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.config.graph_parent_ext_blacklist
    
    def is_code_file(self, file_path: str) -> bool:
        """
        Check if a file path is a code file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file path is a code file, False otherwise
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.config.code_extensions
    
    def can_have_children(self, file_path: str) -> bool:
        """
        Check if a file can have children in the file graph.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file can have children, False otherwise
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.config.graph_parent_ext_whitelist
    
    def is_image_file(self, file_path: str) -> bool:
        """
        Check if a file path is an image file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file path is an image file, False otherwise
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.config.image_extensions

