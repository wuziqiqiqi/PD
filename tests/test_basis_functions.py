import pytest
from clease.basis_function import Polynomial, Trigonometric, BinaryLinear

all_bfs = (Polynomial, Trigonometric, BinaryLinear)


@pytest.fixture(params=all_bfs)
def bf_fun(request):
    """Fixture to run a test on all the avaialble basis functions"""
    return request.param


@pytest.mark.parametrize(
    "test",
    [
        {
            "bf": Polynomial(["Au", "Cu", "X"]),
            "expect": {"name": "polynomial", "unique_elements": ["Au", "Cu", "X"]},
        },
        {
            "bf": Trigonometric(["Au", "Cu", "X"]),
            "expect": {"name": "trigonometric", "unique_elements": ["Au", "Cu", "X"]},
        },
        {
            "bf": BinaryLinear(["Au", "Cu", "X"]),
            "expect": {
                "name": "binary_linear",
                "unique_elements": ["Au", "Cu", "X"],
                "redundant_element": "Au",
            },
        },
        {
            "bf": BinaryLinear(["Au", "Cu", "X"], redundant_element="X"),
            "expect": {
                "name": "binary_linear",
                "unique_elements": ["Au", "Cu", "X"],
                "redundant_element": "X",
            },
        },
    ],
)
def test_todict(test):
    dct_rep = test["bf"].todict()
    assert dct_rep == test["expect"]


@pytest.mark.parametrize(
    "test",
    [
        {
            "bf": BinaryLinear(["Au", "Cu", "X"], redundant_element="X"),
            "full_name": ("c4_d0012_1_1000", "c3_d0001_4_111"),
            "ans": ("c4_d0012_1_CuAuAuAu", "c3_d0001_4_CuCuCu"),
        },
        {
            "bf": BinaryLinear(["Au", "Cu", "Zn", "Ag"], redundant_element="Cu"),
            "full_name": ("c4_d0001_10_1210", "c3_d0991_10_010"),
            "ans": ("c4_d0001_10_AuZnAuAg", "c3_d0991_10_AgAuAg"),
        },
        {
            "bf": Polynomial(["Au", "Cu", "X"]),
            "full_name": ("c2_d0001_99_01", "c4_d0991_10_0122"),
            "ans": ("c2_d0001_99_01", "c4_d0991_10_0122"),
        },
    ],
)
def test_customize_full_cluster_name(test):
    bf = test["bf"]
    for i in range(len(test["ans"])):
        name = bf.customize_full_cluster_name(test["full_name"][i])
        assert name == test["ans"][i]


def test_num_unique_elements(bf_fun):
    with pytest.raises(ValueError):
        # Two of same symbols
        bf_fun(["Au", "Au"])
    with pytest.raises(ValueError):
        # Just 1 symbols
        bf_fun(["Au"])

    assert bf_fun(["Au", "Ag"]).num_unique_elements == 2
    assert bf_fun(["Au", "Au", "Ag"]).num_unique_elements == 2

    # Test setting directly
    bf = bf_fun(["Au", "Ag"])
    assert bf.num_unique_elements == 2
    bf.unique_elements = ["Au", "Au", "Ag"]
    assert bf.num_unique_elements == 2
    bf.unique_elements = ["Au", "Ag", "Cu"]
    assert bf.num_unique_elements == 3


@pytest.mark.parametrize("bf_fun2", all_bfs)
def test_equality(bf_fun, bf_fun2):
    ele = ["Au", "Ag"]
    bf1 = bf_fun(ele)
    bf2 = bf_fun2(ele)

    # We have same atoms
    if isinstance(bf1, type(bf2)):
        # Same type of BF
        assert bf1 == bf2
    else:
        # Different type of BF
        assert bf1 != bf2

    # Different atoms, always unequal
    bf2.unique_elements = ["Au", "Zn"]
    assert bf1 != bf2

    # Test some things it should never be equal to
    assert bf1 != "some_string"
    assert bf1 != []
    assert bf1 != None
    assert bf1 not in [True, False]


@pytest.mark.parametrize("must_implement", ["get_spin_dict", "get_basis_functions"])
def test_get_must_implements(bf_fun, must_implement):
    """Check that subclasses implemented the necessary items."""
    bf = bf_fun(["Au", "Ag", "Cu", "C"])
    getattr(bf, must_implement)()


def test_save_load_roundtrip(bf_fun, make_tempfile, compare_dict):
    file = make_tempfile("bf.json")
    bf = bf_fun(["Au", "Ag", "Cu"])
    with open(file, "w") as fd:
        bf.save(fd)
    with open(file) as fd:
        bf_loaded = bf_fun.load(fd)
    assert type(bf) is type(bf_loaded)
    compare_dict(bf.todict(), bf_loaded.todict())
