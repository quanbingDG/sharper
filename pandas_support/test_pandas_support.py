# -*- coding: utf-8 -*-
# @Time : 2020/11/9 9:13 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
import pandas as pd
import numpy as np
from unittest import TestCase
from pandas_support import PandasSupport as PS


# @File : test_pandas_support.py
class TestPandasSupport(TestCase):
    def setUp(self) -> None:
        self._test_frame = pd.DataFrame(np.array([1, 2, 3, 4]).reshape(2, 2), columns=['i1', 'i2'])

    def test_check_cols(self):
        self.assertEqual(PS.check_cols(['col1', 'col2'], ['col1']), True)
        self.assertEqual(PS.check_cols(['col1', 'col2'], ['col']), False)
        self.assertEqual(PS.check_cols(['col1', 'col2'], ['col1', 'col3']), False)
        self.assertEqual(PS.check_cols(['col1', 'col2'], 'col1'), True)

    def test_add_ratio(self):
        self.assertEqual(PS.add_ratio(self._test_frame, ['i1']).columns.__len__(), 3)
        self.assertEqual(PS.add_ratio(self._test_frame, ['i1'], csum=True).columns.__len__(), 4)

    def test_add_csum(self):
        self.assertEqual(PS.add_csum(self._test_frame, 'i1').columns.__len__(), 3)
