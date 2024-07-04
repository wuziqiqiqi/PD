from ase.build import bulk
from clease import ClusterCoverageChecker


def test_cluster_coverage(bc_setting):
    coverage_checker = ClusterCoverageChecker(bc_setting)

    tests = [
        {
            "atoms": bulk("Au", a=4.05),
            "expect": {
                "c2_d0000_0": 1.0 / 12.0,
                "c2_d0001_0": 1.0 / 6.0,
                "c2_d0002_0": 1.0 / 24.0,
            },
        },
        {
            "atoms": bulk("Au", a=4.05) * (2, 1, 1),
            "expect": {
                "c2_d0000_0": 2.0 / 12.0,
                "c2_d0001_0": 1.0 / 6.0,
                "c2_d0002_0": 2.0 / 24.0,
            },
        },
    ]

    for test in tests:
        coverage = coverage_checker.coverage(test["atoms"])
        expect = test["expect"]
        assert all(v == coverage[k] for k, v in expect.items())

    # Sanity check: max_coverage should be the same as coverage for 3x3x3
    atoms = bulk("Au", a=4.05) * (3, 3, 3)
    max_cov = coverage_checker.max_coverage()

    # Just print report to make sure it works
    coverage_checker.print_report()
    assert max_cov == coverage_checker.coverage(atoms)
