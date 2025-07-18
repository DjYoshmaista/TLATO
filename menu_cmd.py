from pathlib import Path
from abc import ABC, abstractmethod

# Main Menu System using Command Pattern
class MenuCommand(ABC):
    """Abstract base class for menu commands"""
    
    @abstractmethod
    def can_execute(self) -> bool:
        """Check if command can be executed"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get command description"""
        pass
    
    @abstractmethod
    def execute(self) -> OperationResult:
        """Execute the command"""
        pass

def get_project_root() -> Path:
    # Get the project root directory
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'src').exists() or (parent / 'requirements.txt').exists() or (parent / '.git').exists():
            return parent
    return Path.cwd()