"""
Python 2/3 compatability shim
"""

try:
    # Python 2
    get_char = unichr
    strings = (str, type(get_char(300)))
except NameError:
    # Python 3
    get_char = chr
    strings = str
