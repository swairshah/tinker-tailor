import sys
from unittest.mock import MagicMock

class MockTinker:
    class ServiceClient:
        def __init__(self, **kwargs): pass
        def get_server_capabilities(self): 
            mock = MagicMock()
            m = MagicMock()
            m.model_name = "Qwen/Qwen3.5-4B"
            mock.supported_models = [m]
            return mock
        def create_rest_client(self): return MagicMock()

sys.modules['tinker'] = MockTinker
sys.modules['tinker_cookbook'] = MagicMock()
sys.modules['tinker_cookbook.supervised'] = MagicMock()
sys.modules['tinker_cookbook.rl'] = MagicMock()
sys.modules['tinker_cookbook.renderers'] = MagicMock()
sys.modules['tinker_cookbook.supervised.data'] = MagicMock()
sys.modules['tinker_cookbook.supervised.types'] = MagicMock()
sys.modules['tinker_cookbook.recipes'] = MagicMock()
sys.modules['tinker_cookbook.recipes.math_rl'] = MagicMock()
sys.modules['certifi'] = MagicMock()
