try:
    from numba import jit as numba_jit
    from numba import jitclass as numba_jitclass
    from numba import int32
    jit = numba_jit
    jitclass = numba_jitclass
    from clease import _logger

except ImportError:
    _logger("Numba not installed. We recommend installing Numba.")

    # Numba is not installed
    def dummy_jit(**options):
        def decorate_func(func):
            def wrapper(*args):
                return func(*args)
            return wrapper
        return decorate_func

    def dummy_jitclass(spec):
        def decorator(cls):
            return cls
        return decorator

    jit = dummy_jit
    jitclass = dummy_jitclass

    class Int32Cls:
        def __getitem__(self, val):
            pass

    int32 = Int32Cls()
