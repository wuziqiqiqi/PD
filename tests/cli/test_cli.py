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


def test_print_cluster_table(run_cmd, bc_setting, mocker):
    # Ensure we don't accidentally boot up a new GUI window.
    mocked_view = mocker.patch("ase.visualize.view")

    assert mocked_view.call_count == 0

    bc_setting.save("test_settings.json")
    opts = "clusters test_settings.json"
    p = run_cmd(opts=opts)
    assert mocked_view.call_count == 0
    assert "Cluster Name" in p.stdout
    assert "Radius" in p.stdout
    assert "c0" in p.stdout
    assert "c1" in p.stdout
    assert "c2_d" in p.stdout
    assert "c3_d" in p.stdout

    # Disable printing the table
    opts = "clusters test_settings.json -t"
    p = run_cmd(opts=opts)
    assert mocked_view.call_count == 0
    assert "Cluster Name" not in p.stdout

    opts = "clusters test_settings.json -g"
    p = run_cmd(opts=opts)
    assert mocked_view.call_count == 1
    assert "Cluster Name" in p.stdout
