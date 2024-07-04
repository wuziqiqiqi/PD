import json
from typing import Any, Dict
import attr
from ase.io import jsonio as aseio
from ase.utils import reader, writer

__all__ = ("encode", "decode", "read_json", "write_json", "jsonable")


def clease_default(obj: Any) -> Any:
    """Function for identifying CLEASE object types
    in the Encoder"""
    if hasattr(obj, "clease_objtype"):
        d = obj.todict()
        d["__clease_objtype__"] = obj.clease_objtype
        return d
    raise TypeError("Not a CLEASE object")


class CleaseEncoder(json.JSONEncoder):
    """Wrapper around jsonio encoder, for encoding clease object types.
    Includes the ase JSON encoder as well, for simple encoding of
    ASE and NumPy objects.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoders = [clease_default, aseio.MyEncoder().default]

    # Arguments differ, because they call the variable "o" instead of "obj"
    # in JSONEncoder. It doesn't look as nice.
    def default(self, obj):
        # pylint: disable=arguments-renamed
        for encoder in self.encoders:
            try:
                res = encoder(obj)
            except TypeError:
                # Encoders raise TypeError if they cannot deal with it
                continue
            return res

        # Will raise a TypeError
        return super().default(obj)


encode = CleaseEncoder().encode


def object_hook(dct):
    """Wrapper around ASE jsonio object hook"""
    if "__clease_objtype__" in dct:
        objtype = dct.pop("__clease_objtype__")
        return create_clease_object(objtype, dct)
    return aseio.object_hook(dct)


# Note: we disable cyclic imports, as imports here are defered until much later,
# so it's not actually an issue.
# pylint: disable=import-outside-toplevel,cyclic-import
def create_clease_object(objtype, dct):
    if objtype == "concentration":
        from .settings import Concentration

        obj = Concentration.from_dict(dct)
    elif objtype == "basisfunction":
        # It could be multiple types of basis functions
        from .basis_function import basis_function_from_dict

        obj = basis_function_from_dict(dct)
    elif objtype == "ce_settings":
        from .settings import ClusterExpansionSettings

        obj = ClusterExpansionSettings.from_dict(dct)
    elif objtype == "cluster_list":
        from .cluster.cluster_list import ClusterList

        obj = ClusterList.from_dict(dct)
    elif objtype == "cluster":
        from .cluster.cluster import Cluster

        obj = Cluster.from_dict(dct)
    elif objtype == "cluster_fingerprint":
        from .cluster.cluster_fingerprint import ClusterFingerprint

        obj = ClusterFingerprint.from_dict(dct)
    elif objtype == "figure":
        from .datastructures.figure import Figure

        obj = Figure.from_dict(dct)
    elif objtype == "four_vector":
        from .datastructures.four_vector import FourVector

        obj = FourVector.from_dict(dct)
    elif objtype == "system_change":
        from .datastructures.system_changes import SystemChange

        obj = SystemChange.from_dict(dct)
    elif objtype == "mc_step":
        from .datastructures.mc_step import MCStep

        obj = MCStep.from_dict(dct)
    else:
        raise ValueError(f"Cannot load object type: {objtype}")
    assert obj.clease_objtype == objtype
    return obj


cleasedecode = json.JSONDecoder(object_hook=object_hook).decode


def decode(txt, always_array=False):
    obj = cleasedecode(txt)
    obj = aseio.fix_int_keys_in_dicts(obj)
    if always_array:
        obj = aseio.numpyfy(obj)
    return obj


@reader
def read_json(fd, always_array=True):
    dct = decode(fd.read(), always_array=always_array)
    return dct


@writer
def write_json(fd, obj):
    fd.write(encode(obj))


def jsonable(name):
    """Similar to the ASE ase.utils.jsonable decorator, but instead adds a clease_objtype.

    Adds the following methods to the class:
        cls.save
        cls.load
    `load` is added as a classmethod, as it returns a new instance of that class type
    """

    # Define helper methods for saving/loading
    def save_json(self, fd):
        """Method for writing class object to a JSON file."""
        write_json(fd, self)

    @classmethod
    def load_json(cls, fd, **kwargs):
        """Method for loading class object from JSON"""
        obj = read_json(fd, **kwargs)
        assert isinstance(obj, cls)  # pylint: disable=isinstance-second-argument-not-valid-type
        return obj

    def jsonableclass(cls):
        """Wrapper function which adds the save/load methods to the class"""
        assert hasattr(cls, "todict")

        # Define the class attributes
        cls.clease_objtype = name
        cls.save = save_json
        cls.load = load_json
        return cls

    return jsonableclass


class AttrSavable:
    """Mixin for saving simple attrs classes"""

    def todict(self) -> Dict[str, Any]:
        """Convert into a dictionary representation."""
        # Disable recursive, since the json module takes care of that
        return attr.asdict(self, recurse=False)

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]):
        """Load an instance of the class from a dictionary."""
        return cls(**dct)
