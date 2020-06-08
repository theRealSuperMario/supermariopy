import pytest
import numpy as np


class Test_denseposelib:
    @pytest.mark.parametrize(
        "in_shape,out_shape", [((256, 256), (128, 128)), ((3, 256, 256), (3, 128, 128))]
    )
    def test_resize_labels(self, in_shape, out_shape):
        from supermariopy.denseposelib import resize_labels

        labels = np.random.randint(0, 10, in_shape)
        resized = resize_labels(labels, out_shape[-2:])
        assert resized.shape == out_shape

    def test_compute_iou(self):
        from supermariopy.denseposelib import compute_iou

        A = np.ones((10, 10, 1), dtype=np.int)
        B = np.ones((10, 10, 1), dtype=np.int)
        B[:5, :5] = 0

        iou, unique_labels = compute_iou(A, B)

        assert (float(iou[unique_labels == 1])) == 0.75

        A = np.ones((10, 10), dtype=np.int)
        B = np.ones((10, 10), dtype=np.int)
        B[:5, :5] = 0

        iou, unique_labels = compute_iou(A, B)

        assert (float(iou[unique_labels == 1])) == 0.75

    def test_calculate_iou_df(self):
        from supermariopy.denseposelib import calculate_iou_df

        A = np.ones((10, 10), dtype=np.int)
        B = np.ones((10, 10), dtype=np.int)
        B[:5, :5] = 0
        B[5:, 5:] = 1
        B[5:, :5] = 2

        predicted = np.stack([A] * 10, axis=0)
        target = np.stack([B] * 10, axis=0)
        label_names = ["zeros", "ones", "twos", "threes"]

        df = calculate_iou_df(predicted, target, label_names)

        assert np.allclose(df.zeros, np.zeros((10,)))
        assert np.allclose(df.ones, np.ones((10,)) * 0.5)
        assert np.allclose(df.twos, np.zeros((10,)))
        assert np.allclose(df.threes, np.ones((10,)) * -1.0)

    def test_calculate_overall_iou_from_df(self):
        from supermariopy.denseposelib import (
            calculate_overall_iou_from_df,
            calculate_iou_df,
        )

        A = np.ones((10, 10), dtype=np.int)
        B = np.ones((10, 10), dtype=np.int)
        B[:5, :5] = 0
        B[5:, 5:] = 1
        B[5:, :5] = 2

        predicted = np.stack([A] * 10, axis=0)
        target = np.stack([B] * 10, axis=0)
        label_names = ["zeros", "ones", "twos", "threes"]

        df = calculate_iou_df(predicted, target, label_names)
        df_mean = calculate_overall_iou_from_df(df)

        print(df_mean)
        np.testing.assert_almost_equal(df_mean["overall"], np.array([0.5 / 3]))

    def test_remap_parts(self):
        predicted_labels = np.zeros((1, 100, 100), dtype=np.int)
        true_labels = np.zeros((1, 100, 100), dtype=np.int)

        g = 1
        for i in range(10):
            for j in range(10):
                predicted_labels[
                    :, (i * 10) : (i + 1) * 10, (j * 10) : (j + 1) * 10
                ] = g
                g += 1

        for i in range(5):
            true_labels[:, (i * 20) : (i + 1) * 20, (i * 20) : (i + 1) * 20] = i + 1

        from supermariopy.denseposelib import remap_parts
        from supermariopy.metrics import compute_best_iou_remapping

        remap_dict = compute_best_iou_remapping(predicted_labels, true_labels)
        remapped_labels = remap_parts(predicted_labels, remap_dict)
        assert np.all(remapped_labels == true_labels)
