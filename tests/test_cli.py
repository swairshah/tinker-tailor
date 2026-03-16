import sys
from unittest.mock import patch

def test_cli_help(capsys):
    from llm_tuner.cli import main
    with patch.object(sys, 'argv', ['llm-tuner', '--help']):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0
        
        captured = capsys.readouterr()
        assert "Qwen3.5-4B Tinker toolkit" in captured.out
