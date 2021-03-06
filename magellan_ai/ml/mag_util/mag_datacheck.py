# coding: utf-8

from __future__ import absolute_import, division, \
    print_function, unicode_literals


def is_number(s):
    """check input 's' si numberic type or not"""
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
