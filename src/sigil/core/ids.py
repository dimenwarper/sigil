"""
Entity ID generation for Sigil.

Provides stable, content-addressed identifiers for functions and candidates.
"""

import ast
import hashlib
import inspect
from typing import Any, Dict, Optional
from pathlib import Path
import subprocess


class FunctionID:
    """
    Content-addressed function identifier.
    
    Format: sigil://<package>@<git-commit>/<module>:<qualname>#<abi-hash>
    """
    
    def __init__(self, package: str, git_commit: str, module: str, 
                 qualname: str, abi_hash: str):
        self.package = package
        self.git_commit = git_commit
        self.module = module
        self.qualname = qualname
        self.abi_hash = abi_hash
    
    @classmethod
    def from_function(cls, func: Any, package_name: Optional[str] = None) -> "FunctionID":
        """Generate FunctionID from a Python function."""
        # Get git commit
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=Path.cwd(),
                text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_commit = "unknown"
        
        # Get package name
        if package_name is None:
            package_name = getattr(func, "__module__", "").split(".")[0] or "unknown"
        
        # Get module and qualname
        module = getattr(func, "__module__", "unknown")
        qualname = getattr(func, "__qualname__", func.__name__)
        
        # Generate ABI hash from function source
        try:
            source = inspect.getsource(func)
            # Parse and normalize AST
            tree = ast.parse(source)
            normalized = ast.dump(tree, annotate_fields=False)
            
            # Include Python version and key dependencies in hash
            import sys
            hash_input = f"{normalized}|python:{sys.version_info[:2]}"
            abi_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        except (OSError, TypeError):
            # Fallback for functions without accessible source
            abi_hash = hashlib.sha256(f"{module}:{qualname}".encode()).hexdigest()[:16]
        
        return cls(package_name, git_commit, module, qualname, abi_hash)
    
    def __str__(self) -> str:
        return f"sigil://{self.package}@{self.git_commit}/{self.module}:{self.qualname}#{self.abi_hash}"
    
    def __repr__(self) -> str:
        return f"FunctionID('{str(self)}')"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, FunctionID):
            return False
        return str(self) == str(other)
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    @classmethod
    def from_str(cls, id_str: str) -> "FunctionID":
        """Parse FunctionID from string representation."""
        if not id_str.startswith("sigil://"):
            raise ValueError("Invalid FunctionID format")
        
        parts = id_str.replace("sigil://", "").split("/")
        if len(parts) != 2:
            raise ValueError("Invalid FunctionID format")
        
        package_commit = parts[0].split("@")
        module_qual = parts[1].split(":")
        
        if len(package_commit) != 2 or len(module_qual) != 2:
            raise ValueError("Invalid FunctionID format")
        
        package = package_commit[0]
        commit = package_commit[1]
        module = module_qual[0]
        qualname_hash = module_qual[1].split("#")
        
        if len(qualname_hash) != 2:
            raise ValueError("Invalid FunctionID format")
        
        qualname = qualname_hash[0]
        abi_hash = qualname_hash[1]
        
        return cls(package, commit, module, qualname, abi_hash)


class CandidateID:
    """
    Content-addressed candidate identifier.
    
    Format: diff://b3:<digest>
    """
    
    def __init__(self, digest: str):
        self.digest = digest
    
    @classmethod
    def from_diff(cls, diff_text: str) -> "CandidateID":
        """Generate CandidateID from unified diff text."""
        # Use blake3 if available, otherwise SHA-256
        try:
            import blake3
            digest = blake3.blake3(diff_text.encode()).hexdigest()
        except ImportError:
            digest = hashlib.sha256(diff_text.encode()).hexdigest()
        
        return cls(digest)
    
    @classmethod
    def from_source(cls, original_source: str, new_source: str) -> "CandidateID":
        """Generate CandidateID from source code comparison."""
        import difflib
        
        diff_lines = list(difflib.unified_diff(
            original_source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            n=3
        ))
        diff_text = "".join(diff_lines)
        
        return cls.from_diff(diff_text)
    
    def __str__(self) -> str:
        return f"diff://b3:{self.digest}"
    
    def __repr__(self) -> str:
        return f"CandidateID('{str(self)}')"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, CandidateID):
            return False
        return self.digest == other.digest
    
    def __hash__(self) -> int:
        return hash(self.digest)
    
    @classmethod 
    def from_str(cls, id_str: str) -> "CandidateID":
        """Parse CandidateID from string representation."""
        if not id_str.startswith("diff://b3:"):
            raise ValueError("Invalid CandidateID format")
        
        digest = id_str.replace("diff://b3:", "")
        return cls(digest)