import re
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from rdt.errors import Error
from rdt.transformers.categorical import (
    CustomLabelEncoder, FrequencyEncoder, LabelEncoder, OneHotEncoder)

RE_SSN = re.compile(r'\d\d\d-\d\d-\d\d\d\d')


class TestFrequencyEncoder:

    def test___setstate__(self):
        """Test the ``__set_state__`` method.

        Validate that the ``__dict__`` attribute is correctly udpdated when

        Setup:
            - create an instance of a ``FrequencyEncoder``.

        Side effect:
            - it updates the ``__dict__`` attribute of the object.
        """
        # Setup
        transformer = FrequencyEncoder()

        # Run
        transformer.__setstate__({
            'intervals': {
                None: 'abc'
            }
        })

        # Assert
        assert transformer.__dict__['intervals'][np.nan] == 'abc'

    def test___init__(self):
        """Passed arguments must be stored as attributes."""
        # Run
        transformer = FrequencyEncoder(add_noise='add_noise_value')

        # Asserts
        assert transformer.add_noise == 'add_noise_value'

    def test_is_transform_deterministic(self):
        """Test the ``is_transform_deterministic`` method.

        Validate the method returns the opposite boolean value of the ``add_noise`` parameter.

        Setup:
            - initialize a ``FrequencyEncoder`` with ``add_noise = True``.

        Output:
            - the boolean value which is the opposite of ``add_noise``.
        """
        # Setup
        transformer = FrequencyEncoder(add_noise=True)

        # Run
        output = transformer.is_transform_deterministic()

        # Assert
        assert output is False

    def test__get_intervals(self):
        """Test the ``_get_intervals`` method.

        Validate that the intervals for each categorical value are correct.

        Input:
            - a pandas series containing categorical values.

        Output:
            - a tuple, where the first element describes the intervals for each
            categorical value (start, end).
        """
        # Run
        data = pd.Series(['foo', 'bar', 'bar', 'foo', 'foo', 'tar'])
        result = FrequencyEncoder._get_intervals(data)

        # Asserts
        expected_intervals = {
            'foo': (
                0,
                0.5,
                0.25,
                0.5 / 6
            ),
            'bar': (
                0.5,
                0.8333333333333333,
                0.6666666666666666,
                0.05555555555555555
            ),
            'tar': (
                0.8333333333333333,
                0.9999999999999999,
                0.9166666666666666,
                0.027777777777777776
            )
        }
        expected_means = pd.Series({
            'foo': 0.25,
            'bar': 0.6666666666666666,
            'tar': 0.9166666666666666
        })
        expected_starts = pd.DataFrame({
            'category': ['foo', 'bar', 'tar'],
            'start': [0, 0.5, 0.8333333333333333]
        }).set_index('start')

        assert result[0] == expected_intervals
        pd.testing.assert_series_equal(result[1], expected_means)
        pd.testing.assert_frame_equal(result[2], expected_starts)

    def test__get_intervals_nans(self):
        """Test the ``_get_intervals`` method when data contains nan's.

        Validate that the intervals for each categorical value are correct, when passed
        data containing nan values.

        Input:
            - a pandas series cotaining nan values and categorical values.

        Output:
            - a tuple, where the first element describes the intervals for each
            categorical value (start, end).
        """
        # Setup
        data = pd.Series(['foo', np.nan, None, 'foo', 'foo', 'tar'])

        # Run
        result = FrequencyEncoder._get_intervals(data)

        # Assert
        expected_intervals = {
            'foo': (
                0,
                0.5,
                0.25,
                0.5 / 6
            ),
            np.nan: (
                0.5,
                0.8333333333333333,
                0.6666666666666666,
                0.05555555555555555
            ),
            'tar': (
                0.8333333333333333,
                0.9999999999999999,
                0.9166666666666666,
                0.027777777777777776
            )
        }
        expected_means = pd.Series({
            'foo': 0.25,
            np.nan: 0.6666666666666666,
            'tar': 0.9166666666666666
        })
        expected_starts = pd.DataFrame({
            'category': ['foo', np.nan, 'tar'],
            'start': [0, 0.5, 0.8333333333333333]
        }).set_index('start')

        assert result[0] == expected_intervals
        pd.testing.assert_series_equal(result[1], expected_means)
        pd.testing.assert_frame_equal(result[2], expected_starts)

    def test__fit_intervals(self):
        # Setup
        transformer = FrequencyEncoder()

        # Run
        data = pd.Series(['foo', 'bar', 'bar', 'foo', 'foo', 'tar'])
        transformer._fit(data)

        # Asserts
        expected_intervals = {
            'foo': (
                0,
                0.5,
                0.25,
                0.5 / 6
            ),
            'bar': (
                0.5,
                0.8333333333333333,
                0.6666666666666666,
                0.05555555555555555
            ),
            'tar': (
                0.8333333333333333,
                0.9999999999999999,
                0.9166666666666666,
                0.027777777777777776
            )
        }
        expected_means = pd.Series({
            'foo': 0.25,
            'bar': 0.6666666666666666,
            'tar': 0.9166666666666666
        })
        expected_starts = pd.DataFrame({
            'category': ['foo', 'bar', 'tar'],
            'start': [0, 0.5, 0.8333333333333333]
        }).set_index('start')

        assert transformer.intervals == expected_intervals
        pd.testing.assert_series_equal(transformer.means, expected_means)
        pd.testing.assert_frame_equal(transformer.starts, expected_starts)

    def test__get_value_add_noise_false(self):
        # Setup
        transformer = FrequencyEncoder(add_noise=False)
        transformer.intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
            np.nan: (0.5, 1.0, 0.75, 0.5 / 6),
        }

        # Run
        result_foo = transformer._get_value('foo')
        result_nan = transformer._get_value(np.nan)

        # Asserts
        assert result_foo == 0.25
        assert result_nan == 0.75

    @patch('rdt.transformers.categorical.norm')
    def test__get_value_add_noise_true(self, norm_mock):
        # setup
        norm_mock.rvs.return_value = 0.2745

        transformer = FrequencyEncoder(add_noise=True)
        transformer.intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
        }

        # Run
        result = transformer._get_value('foo')

        # Asserts
        assert result == 0.2745

    def test__reverse_transform_array(self):
        """Test reverse_transform a numpy.array"""
        # Setup
        data = pd.Series(['foo', 'bar', 'bar', 'foo', 'foo', 'tar'])
        rt_data = np.array([-0.6, 0.5, 0.6, 0.2, 0.1, -0.2])
        transformer = FrequencyEncoder()

        # Run
        transformer._fit(data)
        result = transformer._reverse_transform(rt_data)

        # Asserts
        expected_intervals = {
            'foo': (
                0,
                0.5,
                0.25,
                0.5 / 6
            ),
            'bar': (
                0.5,
                0.8333333333333333,
                0.6666666666666666,
                0.05555555555555555
            ),
            'tar': (
                0.8333333333333333,
                0.9999999999999999,
                0.9166666666666666,
                0.027777777777777776
            )
        }

        assert transformer.intervals == expected_intervals

        expect = pd.Series(['foo', 'bar', 'bar', 'foo', 'foo', 'foo'])
        pd.testing.assert_series_equal(result, expect)

    def test__transform_user_warning(self):
        """Test the ``_transform`` method generates the correct user warning.

        When asked to transform data not seen during the fit, a UserWarning should be raised.

        Setup:
            - create an instance of the ``FrequencyEncoder``, where ``means`` is a list
            of floats and ``intervals`` is the appropriate dictionary.

        Input:
            - a pandas series containing a np.nan.

        Output:
            - a numpy array containing the transformed data.

        Raises:
            - a UserWarning with the correct message.
        """
        # Setup
        data = pd.Series([1, 2, 3, 4, np.nan])
        transformer = FrequencyEncoder()
        transformer.means = [0.125, 0.375, 0.625, 0.875]
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            1: (0.75, 1.0, 0.875, 0.041666666666666664),
        }

        # Run
        warning_msg = re.escape(
            'The data contains 1 new categories that were not '
            'seen in the original data (examples: {nan}). Assigning '
            'them random values. If you want to model new categories, '
            'please fit the transformer again with the new data.'
        )
        with pytest.warns(UserWarning, match=warning_msg):
            transformed = transformer._transform(data)

        # Assert
        expected = pd.Series([0.875, 0.625, 0.375, 0.125])
        np.testing.assert_array_equal(transformed[:4], expected)

        assert transformed[4] in transformer.means

    def test__transform_by_category_called(self):
        """Test that the `_transform_by_category` method is called.

        When the number of rows is greater than the number of categories, expect
        that the `_transform_by_category` method is called.

        Setup:
            The categorical transformer is instantiated with 4 categories.

        Input:
            - data with 5 rows.

        Output:
            - the output of `_transform_by_category`.

        Side effects:
            - `_transform_by_category` will be called once.
        """
        # Setup
        data = pd.Series([1, 3, 3, 2, 1])

        categorical_transformer_mock = Mock()
        categorical_transformer_mock.means = pd.Series([0.125, 0.375, 0.625, 0.875])

        # Run
        transformed = FrequencyEncoder._transform(categorical_transformer_mock, data)

        # Asserts
        categorical_transformer_mock._transform_by_category.assert_called_once_with(data)
        assert transformed == categorical_transformer_mock._transform_by_category.return_value

    def test__transform_by_category(self):
        """Test the `_transform_by_category` method with numerical data.

        Expect that the correct transformed data is returned.

        Setup:
            The categorical transformer is instantiated with 4 categories and intervals.

        Input:
            - data with 5 rows.

        Ouptut:
            - the transformed data.
        """
        # Setup
        data = pd.Series([1, 3, 3, 2, 1])
        transformer = FrequencyEncoder()
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            1: (0.75, 1.0, 0.875, 0.041666666666666664),
        }

        # Run
        transformed = transformer._transform_by_category(data)

        # Asserts
        expected = np.array([0.875, 0.375, 0.375, 0.625, 0.875])
        assert (transformed == expected).all()

    def test__transform_by_category_nans(self):
        """Test the ``_transform_by_category`` method with data containing nans.

        Validate that the data is transformed correctly when it contains nan's.

        Setup:
            - the categorical transformer is instantiated, and the appropriate ``intervals``
            attribute is set.

        Input:
            - a pandas series containing nan's.

        Output:
            - a numpy array containing the transformed data.
        """
        # Setup
        data = pd.Series([np.nan, 3, 3, 2, np.nan])
        transformer = FrequencyEncoder()
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            np.nan: (0.75, 1.0, 0.875, 0.041666666666666664),
        }

        # Run
        transformed = transformer._transform_by_category(data)

        # Asserts
        expected = np.array([0.875, 0.375, 0.375, 0.625, 0.875])
        assert (transformed == expected).all()

    @patch('rdt.transformers.categorical.norm')
    def test__transform_by_category_add_noise_true(self, norm_mock):
        """Test the ``_transform_by_category`` method when ``add_noise`` is True.

        Validate that the data is transformed correctly when ``add_noise`` is True.

        Setup:
            - the categorical transformer is instantiated with ``add_noise`` as True,
            and the appropriate ``intervals`` attribute is set.
            - the ``intervals`` attribute is set to a a dictionary of intervals corresponding
            to the elements of the passed data.
            - set the ``side_effect`` of the ``rvs_mock`` to the appropriate function.

        Input:
            - a pandas series.

        Output:
            - a numpy array containing the transformed data.

        Side effect:
            - ``rvs_mock`` should be called four times, one for each element of the
            intervals dictionary.
        """
        # Setup
        def rvs_mock_func(loc, scale, **kwargs):
            return loc

        norm_mock.rvs.side_effect = rvs_mock_func

        data = pd.Series([1, 3, 3, 2, 1])
        transformer = FrequencyEncoder(add_noise=True)
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            1: (0.75, 1.0, 0.875, 0.041666666666666664),
        }

        # Run
        transformed = transformer._transform_by_category(data)

        # Assert
        expected = np.array([0.875, 0.375, 0.375, 0.625, 0.875])
        assert (transformed == expected).all()
        norm_mock.rvs.assert_has_calls([
            call(0.125, 0.041666666666666664, size=0),
            call(0.375, 0.041666666666666664, size=2),
            call(0.625, 0.041666666666666664, size=1),
            call(0.875, 0.041666666666666664, size=2),
        ])

    def test__transform_by_row_called(self):
        """Test that the `_transform_by_row` method is called.

        When the number of rows is less than or equal to the number of categories,
        expect that the `_transform_by_row` method is called.

        Setup:
            The categorical transformer is instantiated with 4 categories.
        Input:
            - data with 4 rows
        Output:
            - the output of `_transform_by_row`
        Side effects:
            - `_transform_by_row` will be called once
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])

        categorical_transformer_mock = Mock()
        categorical_transformer_mock.means = pd.Series([0.125, 0.375, 0.625, 0.875])

        # Run
        transformed = FrequencyEncoder._transform(categorical_transformer_mock, data)

        # Asserts
        categorical_transformer_mock._transform_by_row.assert_called_once_with(data)
        assert transformed == categorical_transformer_mock._transform_by_row.return_value

    def test__transform_by_row(self):
        """Test the `_transform_by_row` method with numerical data.

        Expect that the correct transformed data is returned.

        Setup:
            The categorical transformer is instantiated with 4 categories and intervals.
        Input:
            - data with 4 rows
        Ouptut:
            - the transformed data
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])
        transformer = FrequencyEncoder()
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            1: (0.75, 1.0, 0.875, 0.041666666666666664),
        }

        # Run
        transformed = transformer._transform_by_row(data)

        # Asserts
        expected = np.array([0.875, 0.625, 0.375, 0.125])
        assert (transformed == expected).all()

    @patch('psutil.virtual_memory')
    def test__reverse_transform_by_matrix_called(self, psutil_mock):
        """Test that the `_reverse_transform_by_matrix` method is called.

        When there is enough virtual memory, expect that the
        `_reverse_transform_by_matrix` method is called.

        Setup:
            The categorical transformer is instantiated with 4 categories. Also patch the
            `psutil.virtual_memory` function to return a large enough `available_memory`.
        Input:
            - numerical data with 4 rows
        Output:
            - the output of `_reverse_transform_by_matrix`
        Side effects:
            - `_reverse_transform_by_matrix` will be called once
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])

        categorical_transformer_mock = Mock()
        categorical_transformer_mock.means = pd.Series([0.125, 0.375, 0.625, 0.875])

        virtual_memory = Mock()
        virtual_memory.available = 4 * 4 * 8 * 3 + 1
        psutil_mock.return_value = virtual_memory

        # Run
        reverse = FrequencyEncoder._reverse_transform(categorical_transformer_mock, data)

        # Asserts
        reverse_arg = categorical_transformer_mock._reverse_transform_by_matrix.call_args[0][0]
        np.testing.assert_array_equal(reverse_arg, data.clip(0, 1))
        assert reverse == categorical_transformer_mock._reverse_transform_by_matrix.return_value

    @patch('psutil.virtual_memory')
    def test__reverse_transform_by_matrix(self, psutil_mock):
        """Test the _reverse_transform_by_matrix method with numerical data.

        Expect that the transformed data is correctly reverse transformed.

        Setup:
            The categorical transformer is instantiated with 4 categories and means. Also patch
            the ``psutil.virtual_memory`` function to return a large enough ``available_memory``.
        Input:
            - transformed data with 4 rows
        Ouptut:
            - the original data
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])
        transformed = pd.Series([0.875, 0.625, 0.375, 0.125])

        transformer = FrequencyEncoder()
        transformer.starts = pd.DataFrame({'category': [4, 3, 2, 1]}, index=[0., 0.25, 0.5, 0.75])
        transformer.dtype = data.dtype

        virtual_memory = Mock()
        virtual_memory.available = 4 * 4 * 8 * 3 + 1
        psutil_mock.return_value = virtual_memory

        # Run
        reverse = transformer._reverse_transform_by_matrix(transformed)

        # Assert
        pd.testing.assert_series_equal(data, reverse)

    @patch('psutil.virtual_memory')
    def test__reverse_transform_by_category_called(self, psutil_mock):
        """Test that the `_reverse_transform_by_category` method is called.

        When there is not enough virtual memory and the number of rows is greater than the
        number of categories, expect that the `_reverse_transform_by_category` method is called.

        Setup:
            The categorical transformer is instantiated with 4 categories. Also patch the
            `psutil.virtual_memory` function to return an `available_memory` of 1.
        Input:
            - numerical data with 5 rows
        Output:
            - the output of `_reverse_transform_by_category`
        Side effects:
            - `_reverse_transform_by_category` will be called once
        """
        # Setup
        transform_data = pd.Series([1, 3, 3, 2, 1])

        categorical_transformer_mock = Mock()
        categorical_transformer_mock.means = pd.Series([0.125, 0.375, 0.625, 0.875])

        virtual_memory = Mock()
        virtual_memory.available = 1
        psutil_mock.return_value = virtual_memory

        # Run
        reverse = FrequencyEncoder._reverse_transform(
            categorical_transformer_mock, transform_data)

        # Asserts
        reverse_arg = categorical_transformer_mock._reverse_transform_by_category.call_args[0][0]
        np.testing.assert_array_equal(reverse_arg, transform_data.clip(0, 1))
        assert reverse == categorical_transformer_mock._reverse_transform_by_category.return_value

    @patch('psutil.virtual_memory')
    def test__reverse_transform_by_category(self, psutil_mock):
        """Test the _reverse_transform_by_category method with numerical data.

        Expect that the transformed data is correctly reverse transformed.

        Setup:
            The categorical transformer is instantiated with 4 categories, and the means
            and intervals are set for those categories. Also patch the `psutil.virtual_memory`
            function to return an `available_memory` of 1.
        Input:
            - transformed data with 5 rows
        Ouptut:
            - the original data
        """
        data = pd.Series([1, 3, 3, 2, 1])
        transformed = pd.Series([0.875, 0.375, 0.375, 0.625, 0.875])

        transformer = FrequencyEncoder()
        transformer.means = pd.Series([0.125, 0.375, 0.625, 0.875], index=[4, 3, 2, 1])
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            1: (0.75, 1.0, 0.875, 0.041666666666666664),
        }
        transformer.dtype = data.dtype

        virtual_memory = Mock()
        virtual_memory.available = 1
        psutil_mock.return_value = virtual_memory

        reverse = transformer._reverse_transform_by_category(transformed)

        pd.testing.assert_series_equal(data, reverse)

    def test__get_category_from_start(self):
        """Test the ``_get_category_from_start`` method.

        Setup:
            - instantiate a ``FrequencyEncoder``, and set the attribute ``starts``
            to a pandas dataframe with ``set_index`` as ``'start'``.

        Input:
            - an integer, an index from data.

        Output:
            - a category from the data.
        """
        # Setup
        transformer = FrequencyEncoder()
        transformer.starts = pd.DataFrame({
            'start': [0.0, 0.5, 0.7],
            'category': ['a', 'b', 'c']
        }).set_index('start')

        # Run
        category = transformer._get_category_from_start(2)

        # Assert
        assert category == 'c'

    @patch('psutil.virtual_memory')
    def test__reverse_transform_by_row_called(self, psutil_mock):
        """Test that the `_reverse_transform_by_row` method is called.

        When there is not enough virtual memory and the number of rows is less than or equal
        to the number of categories, expect that the `_reverse_transform_by_row` method
        is called.

        Setup:
            The categorical transformer is instantiated with 4 categories. Also patch the
            `psutil.virtual_memory` function to return an `available_memory` of 1.
        Input:
            - numerical data with 4 rows
        Output:
            - the output of `_reverse_transform_by_row`
        Side effects:
            - `_reverse_transform_by_row` will be called once
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])

        categorical_transformer_mock = Mock()
        categorical_transformer_mock.means = pd.Series([0.125, 0.375, 0.625, 0.875])
        categorical_transformer_mock.starts = pd.DataFrame(
            [0., 0.25, 0.5, 0.75], index=[4, 3, 2, 1], columns=['category'])
        categorical_transformer_mock._normalize.return_value = data

        virtual_memory = Mock()
        virtual_memory.available = 1
        psutil_mock.return_value = virtual_memory

        # Run
        reverse = FrequencyEncoder._reverse_transform(categorical_transformer_mock, data)

        # Asserts
        reverse_arg = categorical_transformer_mock._reverse_transform_by_row.call_args[0][0]
        np.testing.assert_array_equal(reverse_arg, data.clip(0, 1))
        assert reverse == categorical_transformer_mock._reverse_transform_by_row.return_value

    @patch('psutil.virtual_memory')
    def test__reverse_transform_by_row(self, psutil_mock):
        """Test the _reverse_transform_by_row method with numerical data.

        Expect that the transformed data is correctly reverse transformed.

        Setup:
            The categorical transformer is instantiated with 4 categories, and the means, starts,
            and intervals are set for those categories. Also patch the `psutil.virtual_memory`
            function to return an `available_memory` of 1.
        Input:
            - transformed data with 4 rows
        Ouptut:
            - the original data
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])
        transformed = pd.Series([0.875, 0.625, 0.375, 0.125])

        transformer = FrequencyEncoder()
        transformer.means = pd.Series([0.125, 0.375, 0.625, 0.875], index=[4, 3, 2, 1])
        transformer.starts = pd.DataFrame(
            [4, 3, 2, 1], index=[0., 0.25, 0.5, 0.75], columns=['category'])
        transformer.intervals = {
            4: (0, 0.25, 0.125, 0.041666666666666664),
            3: (0.25, 0.5, 0.375, 0.041666666666666664),
            2: (0.5, 0.75, 0.625, 0.041666666666666664),
            1: (0.75, 1.0, 0.875, 0.041666666666666664),
        }
        transformer.dtype = data.dtype

        virtual_memory = Mock()
        virtual_memory.available = 1
        psutil_mock.return_value = virtual_memory

        # Run
        reverse = transformer._reverse_transform(transformed)

        # Assert
        pd.testing.assert_series_equal(data, reverse)


class TestOneHotEncoder:

    def test__prepare_data_empty_lists(self):
        # Setup
        ohe = OneHotEncoder()
        data = [[], [], []]

        # Assert
        with pytest.raises(ValueError, match='Unexpected format.'):
            ohe._prepare_data(data)

    def test__prepare_data_nested_lists(self):
        # Setup
        ohe = OneHotEncoder()
        data = [[[]]]

        # Assert
        with pytest.raises(ValueError, match='Unexpected format.'):
            ohe._prepare_data(data)

    def test__prepare_data_list_of_lists(self):
        # Setup
        ohe = OneHotEncoder()

        # Run
        data = [['a'], ['b'], ['c']]
        out = ohe._prepare_data(data)

        # Assert
        expected = np.array(['a', 'b', 'c'])
        np.testing.assert_array_equal(out, expected)

    def test__prepare_data_pandas_series(self):
        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 'b', 'c'])
        out = ohe._prepare_data(data)

        # Assert
        expected = pd.Series(['a', 'b', 'c'])
        np.testing.assert_array_equal(out, expected)

    def test_get_output_sdtypes(self):
        """Test the ``get_output_sdtypes`` method.

        Validate that the ``_add_prefix`` method is properly applied to the ``output_sdtypes``
        dictionary. For this class, the ``output_sdtypes`` dictionary is described as:

        {
            'value1': 'float',
            'value2': 'float',
            ...
        }

        The number of items in the dictionary is defined by the ``dummies`` attribute.

        Setup:
            - initialize a ``OneHotEncoder`` and set:
                - the ``dummies`` attribute to a list.
                - the ``column_prefix`` attribute to a string.

        Output:
            - the ``output_sdtypes`` dictionary, but with ``self.column_prefix``
            added to the beginning of the keys of the ``output_sdtypes`` dictionary.
        """
        # Setup
        transformer = OneHotEncoder()
        transformer.column_prefix = 'abc'
        transformer.dummies = [1, 2]

        # Run
        output = transformer.get_output_sdtypes()

        # Assert
        expected = {
            'abc.value0': 'float',
            'abc.value1': 'float'
        }
        assert output == expected

    def test__fit_dummies_no_nans(self):
        """Test the ``_fit`` method without nans.

        Check that ``self.dummies`` does not
        contain nans.

        Input:
        - Series with values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 2, 'c'])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, ['a', 2, 'c'])

    def test__fit_dummies_nans(self):
        """Test the ``_fit`` method without nans.

        Check that ``self.dummies`` contain ``np.nan``.

        Input:
        - Series with values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 2, 'c', None])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, ['a', 2, 'c', np.nan])

    def test__fit_no_nans(self):
        """Test the ``_fit`` method without nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        should be activated

        Input:
        - Series with values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 'b', 'c'])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, ['a', 'b', 'c'])
        np.testing.assert_array_equal(ohe._uniques, ['a', 'b', 'c'])
        assert ohe._dummy_encoded
        assert not ohe._dummy_na

    def test__fit_no_nans_numeric(self):
        """Test the ``_fit`` method without nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        should be deactivated

        Input:
        - Series with values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series([1, 2, 3])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, [1, 2, 3])
        np.testing.assert_array_equal(ohe._uniques, [1, 2, 3])
        assert not ohe._dummy_encoded
        assert not ohe._dummy_na

    def test__fit_nans(self):
        """Test the ``_fit`` method with nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        and NA should be activated.

        Input:
        - Series with containing nan values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 'b', None])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, ['a', 'b', np.nan])
        np.testing.assert_array_equal(ohe._uniques, ['a', 'b'])
        assert ohe._dummy_encoded
        assert ohe._dummy_na

    def test__fit_nans_numeric(self):
        """Test the ``_fit`` method with nans.

        Check that the settings of the transformer
        are properly set based on the input. Encoding
        should be deactivated and NA activated.

        Input:
        - Series with containing nan values
        """

        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series([1, 2, np.nan])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, [1, 2, np.nan])
        np.testing.assert_array_equal(ohe._uniques, [1, 2])
        assert not ohe._dummy_encoded
        assert ohe._dummy_na

    def test__fit_single(self):
        # Setup
        ohe = OneHotEncoder()

        # Run
        data = pd.Series(['a', 'a', 'a'])
        ohe._fit(data)

        # Assert
        np.testing.assert_array_equal(ohe.dummies, ['a'])

    def test__transform_no_nan(self):
        """Test the ``_transform`` method without nans.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation.

        Input:
        - Series with values
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', 'c'])
        ohe._uniques = ['a', 'b', 'c']
        ohe._num_dummies = 3

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_no_nan_categorical(self):
        """Test the ``_transform`` method without nans.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation
        using the categorical branch.

        Input:
        - Series with categorical values
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', 'c'])
        ohe._uniques = ['a', 'b', 'c']
        ohe._indexer = [0, 1, 2]
        ohe._num_dummies = 3
        ohe._dummy_encoded = True

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_nans_encoded(self):
        """Test the ``_transform`` method with nans.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation. Null
        values should be represented by the same encoding.

        Input:
        - Series with values containing nans
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series([np.nan, None, 'a', 'b'])
        ohe._uniques = ['a', 'b']
        ohe._dummy_na = True
        ohe._num_dummies = 2

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_nans_categorical(self):
        """Test the ``_transform`` method with nans.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation using
        the categorical branch. Null values should be
        represented by the same encoding.

        Input:
        - Series with categorical values containing nans
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series([np.nan, None, 'a', 'b'])
        ohe._uniques = ['a', 'b']
        ohe._indexer = [0, 1]
        ohe._dummy_na = True
        ohe._num_dummies = 2
        ohe._dummy_encoded = True

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_single_column(self):
        """Test the ``_transform`` with one category.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation
        where it should be a single column.

        Input:
        - Series with a single category
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'a', 'a'])
        ohe._uniques = ['a']
        ohe._num_dummies = 1

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([
            [1],
            [1],
            [1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_single_categorical(self):
        """Test the ``_transform`` with one category.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation
        using the categorical branch where it should
        be a single column.

        Input:
        - Series with a single category
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'a', 'a'])
        ohe._uniques = ['a']
        ohe._indexer = [0]
        ohe._num_dummies = 1
        ohe._dummy_encoded = True

        # Run
        out = ohe._transform_helper(data)

        # Assert
        expected = np.array([
            [1],
            [1],
            [1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_zeros(self):
        """Test the ``_transform`` with unknown category.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation
        where it should be a column of zeros.

        Input:
        - Series with unknown values
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        pd.Series(['a'])
        ohe._uniques = ['a']
        ohe._num_dummies = 1

        # Run
        out = ohe._transform_helper(pd.Series(['b', 'b', 'b']))

        # Assert
        expected = np.array([
            [0],
            [0],
            [0]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_zeros_categorical(self):
        """Test the ``_transform`` with unknown category.

        The values passed to ``_transform`` should be
        returned in a one-hot encoding representation
        using the categorical branch where it should
        be a column of zeros.

        Input:
        - Series with categorical and unknown values
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        pd.Series(['a'])
        ohe._uniques = ['a']
        ohe._indexer = [0]
        ohe._num_dummies = 1
        ohe.dummy_encoded = True

        # Run
        out = ohe._transform_helper(pd.Series(['b', 'b', 'b']))

        # Assert
        expected = np.array([
            [0],
            [0],
            [0]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_unknown_nan(self):
        """Test the ``_transform`` with unknown and nans.

        This is an edge case for ``_transform`` where
        unknowns should be zeros and nans should be
        the last entry in the column.

        Input:
        - Series with unknown and nans
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        pd.Series(['a'])
        ohe._uniques = ['a']
        ohe._dummy_na = True
        ohe._num_dummies = 1

        # Run
        out = ohe._transform_helper(pd.Series(['b', 'b', np.nan]))

        # Assert
        expected = np.array([
            [0, 0],
            [0, 0],
            [0, 1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_no_nans(self):
        """Test the ``transform`` without nans.

        In this test ``transform`` should return an identity
        matrix representing each item in the input.

        Input:
        - Series with categorical values
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', 'c'])
        ohe._fit(data)

        # Run
        out = ohe._transform(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_nans(self):
        """Test the ``transform`` with nans.

        In this test ``transform`` should return an identity matrix
        representing each item in the input as well as nans.

        Input:
        - Series with categorical values and nans
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', None])
        ohe._fit(data)

        # Run
        out = ohe._transform(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_single_column_filled_with_ones(self):
        """Test the ``transform`` on a single category.

        In this test ``transform`` should return a column
        filled with ones.

        Input:
        - Series with a single categorical value
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'a', 'a'])
        ohe._fit(data)

        # Run
        out = ohe._transform(data)

        # Assert
        expected = np.array([
            [1],
            [1],
            [1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_unknown(self):
        """Test the ``transform`` with unknown data.

        In this test ``transform`` should raise a warning due to the attempt
        of transforming data with previously unseen categories.

        Input:
        - Series with unknown categorical values
        Output:
        - one-hot encoding of the input, with the unseen category encoded as 0s
        """
        # Setup
        ohe = OneHotEncoder()
        fit_data = pd.Series([1, 2, 3, np.nan])
        ohe._fit(fit_data)

        # Run
        warning_msg = re.escape(
            'The data contains 1 new categories that were not '
            'seen in the original data (examples: {4.0}). Creating '
            'a vector of all 0s. If you want to model new categories, '
            'please fit the transformer again with the new data.'
        )
        with pytest.warns(UserWarning, match=warning_msg):
            transform_data = pd.Series([1, 2, np.nan, 4])
            out = ohe._transform(transform_data)

        # Assert
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        np.testing.assert_array_equal(out, expected)

    def test__transform_numeric(self):
        """Test the ``transform`` on numeric input.

        In this test ``transform`` should return a matrix
        representing each item in the input as one-hot encodings.

        Input:
        - Series with numeric input
        Output:
        - one-hot encoding of the input
        """
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series([1, 2])
        ohe._fit(data)

        expected = np.array([
            [1, 0],
            [0, 1],
        ])

        # Run
        out = ohe._transform(data)

        # Assert
        assert not ohe._dummy_encoded
        np.testing.assert_array_equal(out, expected)

    def test__reverse_transform_no_nans(self):
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', 'c'])
        ohe._fit(data)

        # Run
        transformed = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        out = ohe._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'b', 'c'])
        pd.testing.assert_series_equal(out, expected)

    def test__reverse_transform_nans(self):
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'b', None])
        ohe._fit(data)

        # Run
        transformed = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        out = ohe._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'b', None])
        pd.testing.assert_series_equal(out, expected)

    def test__reverse_transform_single(self):
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'a', 'a'])
        ohe._fit(data)

        # Run
        transformed = np.array([
            [1],
            [1],
            [1]
        ])
        out = ohe._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'a', 'a'])
        pd.testing.assert_series_equal(out, expected)

    def test__reverse_transform_1d(self):
        # Setup
        ohe = OneHotEncoder()
        data = pd.Series(['a', 'a', 'a'])
        ohe._fit(data)

        # Run
        transformed = pd.Series([1, 1, 1])
        out = ohe._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'a', 'a'])
        pd.testing.assert_series_equal(out, expected)


class TestLabelEncoder:

    def test___init__(self):
        """Passed arguments must be stored as attributes."""
        # Run
        transformer = LabelEncoder(add_noise='add_noise_value', order_by='alphabetical')

        # Asserts
        assert transformer.add_noise == 'add_noise_value'
        assert transformer.order_by == 'alphabetical'

    def test___init___bad_order_by(self):
        """Test that the ``__init__`` raises error if ``order_by`` is a bad value.

        Input:
            - ``order_by`` will be set to an unexpected string.

        Expected behavior:
            - An error should be raised.
        """
        # Run / Assert
        message = (
            "order_by must be one of the following values: None, 'numerical_value' or "
            "'alphabetical'"
        )
        with pytest.raises(Error, match=message):
            LabelEncoder(order_by='bad_value')

    def test__order_categories_alphabetical(self):
        """Test the ``_order_categories`` method when ``order_by`` is 'alphabetical'.

        Setup:
            - Set ``order_by`` to 'alphabetical'.

        Input:
            - numpy array of strings that are unordered.

        Output:
            - Same numpy array but with the strings alphabetically ordered.
        """
        # Setup
        transformer = LabelEncoder(order_by='alphabetical')
        arr = np.array(['one', 'two', 'three', 'four'])

        # Run
        ordered = transformer._order_categories(arr)

        # Assert
        np.testing.assert_array_equal(ordered, np.array(['four', 'one', 'three', 'two']))

    def test__order_categories_alphabetical_with_nans(self):
        """Test the ``_order_categories`` method when ``order_by`` is 'alphabetical'.

        Setup:
            - Set ``order_by`` to 'alphabetical'.

        Input:
            - numpy array of strings that are unordered and have nans.

        Output:
            - Same numpy array but with the strings alphabetically ordered and nan at the end.
        """
        # Setup
        transformer = LabelEncoder(order_by='alphabetical')
        arr = np.array(['one', 'two', 'three', np.nan, 'four'], dtype='object')

        # Run
        ordered = transformer._order_categories(arr)

        # Assert
        expected = np.array(['four', 'one', 'three', 'two', np.nan], dtype='object')
        pd.testing.assert_series_equal(pd.Series(ordered), pd.Series(expected))

    def test__order_categories_alphabetical_error(self):
        """Test the ``_order_categories`` method when ``order_by`` is 'alphabetical'.

        If ``order_by`` is 'alphabetical' but the data isn't a string, then an error should
        be raised.

        Setup:
            - Set ``order_by`` to 'alphabetical'.

        Input:
            - numpy array of strings that are unordered.

        Output:
            - Same numpy array but with the strings alphabetically ordered.
        """
        # Setup
        transformer = LabelEncoder(order_by='alphabetical')
        arr = np.array([1, 2, 3, 4])

        # Run / Assert
        message = "The data must be of type string if order_by is 'alphabetical'."
        with pytest.raises(Error, match=message):
            transformer._order_categories(arr)

    def test__order_categories_numerical(self):
        """Test the ``_order_categories`` method when ``order_by`` is 'numerical_value'.

        Setup:
            - Set ``order_by`` to 'numerical_value'.

        Input:
            - numpy array of numbers that are unordered.

        Output:
            - Same numpy array but with the numbers ordered.
        """
        # Setup
        transformer = LabelEncoder(order_by='numerical_value')
        arr = np.array([5, 3.11, 100, 67.8, np.nan, -2.5])

        # Run
        ordered = transformer._order_categories(arr)

        # Assert
        np.testing.assert_array_equal(ordered, np.array([-2.5, 3.11, 5, 67.8, 100, np.nan]))

    def test__order_categories_numerical_error(self):
        """Test the ``_order_categories`` method when ``order_by`` is 'numerical_value'.

        If the array is made up of strings that can't be converted to floats, and `order_by`
        is 'numerical_value', then we should raise an error.

        Setup:
            - Set ``order_by`` to 'numerical_value'.

        Input:
            - numpy array of strings.

        Expected behavior:
            - Error should be raised.
        """
        # Setup
        transformer = LabelEncoder(order_by='numerical_value')
        arr = np.array(['one', 'two', 'three', 'four'])

        # Run / Assert
        message = ("The data must be numerical if order_by is 'numerical_value'.")
        with pytest.raises(Error, match=message):
            transformer._order_categories(arr)

    def test__order_categories_numerical_different_dtype_error(self):
        """Test the ``_order_categories`` method when ``order_by`` is 'numerical_value'.

        If the array is made up of a dtype that is not numeric and can't be converted to a float,
        and `order_by` is 'numerical_value', then we should raise an error.

        Setup:
            - Set ``order_by`` to 'numerical_value'.

        Input:
            - numpy array of booleans.

        Expected behavior:
            - Error should be raised.
        """
        # Setup
        transformer = LabelEncoder(order_by='numerical_value')
        arr = np.array([True, False, False, True])

        # Run / Assert
        message = ("The data must be numerical if order_by is 'numerical_value'.")
        with pytest.raises(Error, match=message):
            transformer._order_categories(arr)

    def test__fit(self):
        """Test the ``_fit`` method.

        Validate that a unique integer representation for each category of the data is stored
        in the ``categories_to_values`` attribute, and the reverse is stored in the
        ``values_to_categories`` attribute .

        Setup:
            - create an instance of the ``LabelEncoder``.

        Input:
            - a pandas series.

        Side effects:
            - set the ``values_to_categories`` dictionary to the appropriate value.
            - set ``categories_to_values`` dictionary to the appropriate value.
        """
        # Setup
        data = pd.Series([1, 2, 3, 2, 1])
        transformer = LabelEncoder()

        # Run
        transformer._fit(data)

        # Assert
        assert transformer.values_to_categories == {0: 1, 1: 2, 2: 3}
        assert transformer.categories_to_values == {1: 0, 2: 1, 3: 2}

    def test__transform(self):
        """Test the ``_transform`` method.

        Validate that each category of the passed data is replaced with its corresponding
        integer value.

        Setup:
            - create an instance of the ``LabelEncoder``, where ``categories_to_values``
            and ``values_to_categories`` are set to dictionaries.

        Input:
            - a pandas series.

        Output:
            - a numpy array containing the transformed data.
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])
        transformer = LabelEncoder()
        transformer.categories_to_values = {1: 0, 2: 1, 3: 2}
        transformer.values_to_categories = {0: 1, 1: 2, 2: 3}

        # Run
        warning_msg = re.escape(
            'The data contains 1 new categories that were not '
            'seen in the original data (examples: {4}). Assigning '
            'them random values. If you want to model new categories, '
            'please fit the transformer again with the new data.'
        )
        with pytest.warns(UserWarning, match=warning_msg):
            transformed = transformer._transform(data)

        # Assert
        expected = pd.Series([0., 1., 2.])
        pd.testing.assert_series_equal(transformed[:-1], expected)

        assert 0 <= transformed[3] <= 2

    def test__transform_add_noise(self):
        """Test the ``_transform`` method with ``add_noise``.

        Validate that the method correctly transforms the categories when ``add_noise`` is True.

        Setup:
            - create an instance of the ``LabelEncoder``, where ``categories_to_values``
            and ``values_to_categories`` are set to dictionaries.
            - set ``add_noise`` to True.

        Input:
            - a pandas series.

        Output:
            - a numpy array containing the transformed data.
        """
        # Setup
        data = pd.Series([1, 2, 3, 4])
        transformer = LabelEncoder(add_noise=True)
        transformer.categories_to_values = {1: 0, 2: 1, 3: 2}
        transformer.values_to_categories = {0: 1, 1: 2, 2: 3}

        # Run
        transformed = transformer._transform(data)

        # Assert
        assert 0 <= transformed[0] < 1
        assert 1 <= transformed[1] < 2
        assert 2 <= transformed[2] < 3
        assert 0 <= transformed[3] < 3

    def test__transform_unseen_categories(self):
        """Test the ``_transform`` method with multiple unseen categories.

        Validate that each category of the passed data is replaced with its corresponding
        integer value.

        Setup:
            - create an instance of the ``LabelEncoder``, where ``categories_to_values``
            and ``values_to_categories`` are set to dictionaries.

        Input:
            - a pandas series.

        Output:
            - a numpy array containing the transformed data.
        """
        # Setup
        fit_data = pd.Series(['a', 2, True])
        transformer = LabelEncoder()
        transformer.categories_to_values = {'a': 0, 2: 1, True: 2}
        transformer.values_to_categories = {0: 'a', 1: 2, 2: True}

        # Run
        with pytest.warns(UserWarning):
            transform_data = pd.Series(['a', 2, True, np.nan, np.nan, np.nan, 'b', False, 3])
            transformed = transformer._transform(transform_data)

        # Assert
        expected = pd.Series([0., 1., 2.])
        pd.testing.assert_series_equal(transformed[:3], expected)

        assert all([0 <= value < len(fit_data) for value in transformed[3:]])

    def test__reverse_transform_clips_values(self):
        """Test the ``_reverse_transform`` method with values not in map.

        If a value that is not in ``values_to_categories`` is passed
        to ``reverse_transform``, then the value should be clipped to
        the range of the dict's keys.

        Input:
        - array with values outside of dict
        Output:
        - categories corresponding to closest key in the dict
        """
        # Setup
        transformer = LabelEncoder()
        transformer.values_to_categories = {0: 'a', 1: 'b', 2: 'c'}
        data = pd.Series([0, 1, 10])

        # Run
        out = transformer._reverse_transform(data)

        # Assert
        pd.testing.assert_series_equal(out, pd.Series(['a', 'b', 'c']))

    def test__reverse_transform_add_noise(self):
        """Test the ``_reverse_transform`` method with ``add_noise``.

        Test that the method correctly reverse transforms the data
        when ``add_noise`` is set to True.

        Input:
            - pd.Series
        Output:
            - corresponding categories
        """
        # Setup
        transformer = LabelEncoder(add_noise=True)
        transformer.values_to_categories = {0: 'a', 1: 'b', 2: 'c'}
        data = pd.Series([0.5, 1.0, 10.9])

        # Run
        out = transformer._reverse_transform(data)

        # Assert
        pd.testing.assert_series_equal(out, pd.Series(['a', 'b', 'c']))


class TestCustomLabelEncoder:

    def test___init__(self):
        """The the ``__init__`` method.

        Passed arguments must be stored as attributes.
        """
        # Run
        transformer = CustomLabelEncoder(order=['b', 'c', 'a', None], add_noise='add_noise_value')

        # Asserts
        assert transformer.add_noise == 'add_noise_value'
        pd.testing.assert_series_equal(transformer.order, pd.Series(['b', 'c', 'a', np.nan]))

    def test__fit(self):
        """Test the ``_fit`` method.

        Validate that a unique integer representation for each category of the data is stored
        in the ``categories_to_values`` attribute, and the reverse is stored in the
        ``values_to_categories`` attribute. The order should match the ``self.order`` indices.

        Setup:
            - create an instance of the ``CustomLabelEncoder``.

        Input:
            - a pandas series.

        Side effects:
            - set the ``values_to_categories`` dictionary to the appropriate value.
            - set ``categories_to_values`` dictionary to the appropriate value.
        """
        # Setup
        data = pd.Series([1, 2, 3, 2, np.nan, 1])
        transformer = CustomLabelEncoder(order=[2, 3, np.nan, 1])

        # Run
        transformer._fit(data)

        # Assert
        expected_values_to_categories = {0: 2, 1: 3, 2: np.nan, 3: 1}
        expected_categories_to_values = {2: 0, 3: 1, 1: 3, np.nan: 2}
        for key, value in transformer.values_to_categories.items():
            assert value == expected_values_to_categories[key] or pd.isna(value)

        for key, value in transformer.categories_to_values.items():
            assert value == expected_categories_to_values.get(key) or pd.isna(key)

    def test__fit_error(self):
        """Test the ``_fit`` method checks that data is in ``self.order``.

        If the data being fit is not in ``self.order`` an error should be raised.

        Setup:
            - create an instance of the ``CustomLabelEncoder``.

        Input:
            - a pandas series.

        Side effects:
            - set the ``values_to_categories`` dictionary to the appropriate value.
            - set ``categories_to_values`` dictionary to the appropriate value.
        """
        # Setup
        data = pd.Series([1, 2, 3, 2, 1, 4])
        transformer = CustomLabelEncoder(order=[2, 1])

        # Run / Assert
        message = re.escape(
            "Unknown categories '[3, 4]'. All possible categories must be defined in the "
            "'order' parameter."
        )
        with pytest.raises(Error, match=message):
            transformer._fit(data)
