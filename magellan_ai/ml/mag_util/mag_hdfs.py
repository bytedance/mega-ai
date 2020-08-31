# coding: utf-8

from __future__ import absolute_import, division, \
    print_function, unicode_literals
import os


def hdfs_to_local(hdfs_path, local_path, is_txt=True):
    """copy hdfs file to local
    param:
    * hdfs_path: hdfs file or dir
    * local_path: local file or dir
    return:
    * res: result message
    """

    res = ''
    if is_txt:
        f = os.popen("hadoop dfs -text {} > {}".format(hdfs_path, local_path))
        res = f.read()
    else:
        f = os.popen("hadoop dfs -get {} {}".format(hdfs_path, local_path))
        res = f.read()

    if '' == res:
        res = 'ok'

    return res
