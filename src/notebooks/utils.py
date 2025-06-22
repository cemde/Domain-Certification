from tueplots.bundles import iclr2024


class Styles:
    def __init__(self):
        # _quarter_style = iclr2024(family="sans-serif", usetex=False, nrows=1)
        # _quarter_style["figure.figsize"] = ((_quarter_style["figure.figsize"][0] / 2, _quarter_style["figure.figsize"]),)
        self.full = iclr2024(family="sans-serif", usetex=False, nrows=1, ncols=1)
        self.half = iclr2024(family="sans-serif", usetex=False, nrows=1, ncols=2)
        self.third = iclr2024(family="sans-serif", usetex=False, nrows=1, ncols=3)
        self.half_double_height = iclr2024(family="sans-serif", usetex=False, nrows=1, ncols=2)
        self.full_double_height = iclr2024(family="sans-serif", usetex=False, nrows=1, ncols=1)

        height = 1.33
        self.full["figure.figsize"] = (5.5, height)
        self.half["figure.figsize"] = (5.5 / 2, height)
        self.third["figure.figsize"] = (5.5 / 3, height)
        self.half_double_height["figure.figsize"] = (5.5 / 2, height * 2.4)
        self.full_double_height["figure.figsize"] = (5.5, height * 2.4)

        axes_labelsize = 7
        self.full["axes.labelsize"] = axes_labelsize
        self.half["axes.labelsize"] = axes_labelsize
        self.third["axes.labelsize"] = axes_labelsize
        self.half_double_height["axes.labelsize"] = axes_labelsize
        self.full_double_height["axes.labelsize"] = axes_labelsize

        # tighter layout for axes
        additional_setings = {
            "xtick.major.pad": -4,  # Decrease spacing between x-axis tick labels and the figure
            "ytick.major.pad": -4,  # Decrease spacing between y-axis tick labels and the figure
            "axes.labelpad": 0,  # Move x-axis label closer to the plot
        }

        self.full.update(additional_setings)
        self.half.update(additional_setings)
        self.third.update(additional_setings)
        self.full_double_height.update(additional_setings)

        # self.quarter = _quarter_style

    def _variable_height(self, ncols: int, height: float):
        style = iclr2024(family="sans-serif", usetex=False, nrows=1, ncols=ncols)
        style["figure.figsize"] = (5.5 / ncols, height)
        return style

    def third_variable_height(self, height: float):
        return self._variable_height(3, height)

    def half_variable_height(self, height: float):
        return self._variable_height(2, height)

    def full_variable_height(self, height: float):
        return self._variable_height(1, height)


styles = Styles()
