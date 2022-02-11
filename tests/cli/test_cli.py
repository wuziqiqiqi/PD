import subprocess
import pytest
from click.testing import CliRunner
import clease
from clease.cli.main import clease_cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def run_cmd(runner):
    def _run_cmd(cli_cls=None, opts=None):
        # Default to run with the clease_cli object
        cli_cls = cli_cls or clease_cli
        result = runner.invoke(cli_cls, opts)
        assert result.exit_code == 0
        return result

    return _run_cmd


def test_entry_point():
    """Test the entry point from the install worked.
    Remaining tests are run with the click CliRunner"""
    result = subprocess.run("clease --version", shell=True, check=True)
    assert result.returncode == 0


def test_version(run_cmd):
    opts = "--version"
    p = run_cmd(opts=opts)
    assert str(clease.__version__) in p.stdout


def test_reconfigure(run_cmd, bc_setting):
    bc_setting.save("test_settings.json")
    opts = "reconfigure test_settings.json"
    p = run_cmd(opts=opts)
    assert "reconfiguration completed" in p.stdout
    assert "updating" in p.stdout
