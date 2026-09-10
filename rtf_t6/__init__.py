"""RT-Focuser T=6 temporal video deblurring package."""

__all__ = ["RT_Focuser_Standard", "RTFocuserT6", "load_rtfocuser_pretrained"]


def __getattr__(name):
    # Path preparation and protocol validation do not require importing torch.
    if name == 'load_rtfocuser_pretrained':
        from .checkpoint import load_rtfocuser_pretrained
        return load_rtfocuser_pretrained
    if name in ('RT_Focuser_Standard', 'RTFocuserT6'):
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
