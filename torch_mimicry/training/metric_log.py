"""
MetricLog object for intelligently logging data to display them more intuitively.
"""


class MetricLog:
    """
    A dictionary-like object that logs data, and includes an extra dict to map the metrics
    to its group name, if any, and the corresponding precision to print out.

    Attributes:
        metrics_dict (dict): A dictionary mapping to another dict containing
            the corresponding value, precision, and the group this metric belongs to.
    """
    def __init__(self, **kwargs):
        self.metrics_dict = {}

    def add_metric(self, name, value, group=None, precision=4):
        """
        Logs metric to internal dict, but with an additional option
        of grouping certain metrics together.

        Args:
            name (str): Name of metric to log.
            value (Tensor/Float): Value of the metric to log.
            group (str): Name of the group to classify different metrics together.
            precision (int): The number of floating point precision to represent the value.

        Returns:
            None
        """
        # Grab tensor values only
        try:
            value = value.item()
        except AttributeError:
            value = value

        self.metrics_dict[name] = dict(value=value,
                                       group=group,
                                       precision=precision)

    def __getitem__(self, key):
        return round(self.metrics_dict[key]['value'],
                     self.metrics_dict[key]['precision'])

    def get_group_name(self, name):
        """
        Obtains the group name of a particular metric. For example, errD and errG
        which represents the discriminator/generator losses could fall under a
        group name called "loss".

        Args:
            name (str): The name of the metric to retrieve group name.

        Returns:
            str: A string representing the group name of the metric.
        """
        return self.metrics_dict[name]['group']

    def keys(self):
        """
        Dict like functionality for retrieving keys.
        """
        return self.metrics_dict.keys()

    def items(self):
        """
        Dict like functionality for retrieving items.
        """
        return self.metrics_dict.items()
