__all__ = ['dev', 'prod']
import os
from .common import *
# Load the appropriate configuration file based on the environment
if os.environ.get('ENV') == 'dev':
    from .dev import *

elif os.environ.get('ENV') == 'prod':
    from .prod import *
else:
    raise ValueError('Environment variable ENV not set')