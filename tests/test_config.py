def test_config():
    from llm_tuner.config import SFTConfig
    cfg = SFTConfig()
    assert cfg.model_name == "Qwen/Qwen3.5-4B"
