# Copyright (c) 2023 CINN Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from op_mapper_test import OpMapperTest, logger
import paddle


class TestFloorDivideOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'X': self.random([3], low=1, high=10, dtype='int32'),
            'Y': self.random([3], low=1, high=10, dtype='int32')
        }

    def set_op_type(self):
        return "elementwise_floordiv"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='X',
            shape=self.feed_data['X'].shape,
            dtype=self.feed_data['X'].dtype)
        y = paddle.static.data(
            name='Y',
            shape=self.feed_data['Y'].shape,
            dtype=self.feed_data['Y'].dtype)
        return {'X': [x], 'Y': [y]}

    def set_op_attrs(self):
        return {}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['X'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestFloorDivideCase1(TestFloorDivideOp):
    def init_input_data(self):
        self.feed_data = {
            'X': self.random([3, 4], low=1, high=10, dtype='int64'),
            'Y': self.random([3, 4], low=1, high=10, dtype='int64')
        }


class TestFloorDivideCase2(TestFloorDivideOp):
    def init_input_data(self):
        self.feed_data = {
            'X': self.random([2, 3, 4], low=1, high=10, dtype='int64'),
            'Y': self.random([3, 4], low=1, high=10, dtype='int64')
        }


if __name__ == "__main__":
    unittest.main()
