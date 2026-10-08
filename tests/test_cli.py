import json

from infraguard.cli import main


def test_generate_cli(tmp_path, capsys):
    output = tmp_path / "dataset.csv.gz"
    assert main(["generate", "--devices", "2", "--steps", "120", "--output", str(output)]) == 0
    assert output.is_file()
    assert "240 synthetic records" in capsys.readouterr().out
