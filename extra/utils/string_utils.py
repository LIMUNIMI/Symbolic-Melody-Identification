#!/usr/bin/env python

"""
Common string-related functionality
"""

def ensure_unicode(s):
    return s if type(s) == unicode else unicode(s,'utf8')

if __name__ == '__main__':
    pass
