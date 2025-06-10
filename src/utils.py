import torch

def is_notebook():
    """
    Check if we're running in a Jupyter notebook environment (including VSCode Jupyter extension)
    """
    try:
        # Check if we're in an IPython environment
        from IPython.core.getipython import get_ipython
        ipython = get_ipython()
        
        if ipython is None:
            return False
            
        # Check if it's a Jupyter notebook (including VSCode Jupyter extension)
        if ipython.__class__.__name__ in ['ZMQInteractiveShell', 'TerminalIPythonApp']:
            return True
            
        # Additional check for VSCode Jupyter extension
        if hasattr(ipython, 'kernel'):
            return True
            
        return False
    except ImportError:
        return False

def get_best_torch_device():
    return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )