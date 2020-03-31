from torch_mimicry.training import metric_log


class TestMetricLog:
    def setup(self):
        self.log_data = metric_log.MetricLog()

    def test_add_metric(self):
        # Singular metric
        self.log_data.add_metric('singular', 1.0124214)
        assert self.log_data['singular'] == 1.0124

        # Multiple metrics under same group
        self.log_data.add_metric('errD', 1.00001, group='loss')
        self.log_data.add_metric('errG', 1.0011, group='loss')

        assert self.log_data.get_group_name(
            'errD') == self.log_data.get_group_name('errG')

    def teardown(self):
        del self.log_data


if __name__ == "__main__":
    test = TestMetricLog()
    test.setup()
    test.test_add_metric()
    test.teardown()
