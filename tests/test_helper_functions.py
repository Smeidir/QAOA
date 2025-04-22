from qaoa.utils.helper_functions import to_bitstring, to_bitstring_str

def test_bitstring_conversion():
    for i in range(8):
        b = to_bitstring(i, 3)
        s = to_bitstring_str(i, 3)
        assert isinstance(b, list)
        assert isinstance(s, str)
        assert int(s, 2) == i