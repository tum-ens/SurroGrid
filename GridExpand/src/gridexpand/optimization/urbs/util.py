
try:
    isinstance("", basestring)  # noqa: F821 - Python 2 probe, NameError on Python 3

    def is_string(s):
        return isinstance(s, basestring)  # noqa: F821 - Python 2 only
except NameError:

    def is_string(s):
        return isinstance(s, str)  # Python 2
