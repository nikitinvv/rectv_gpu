from pkg_resources import get_distribution, DistributionNotFound

from rectv_gpu.rectv import *
from rectv_gpu.solver import *
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass