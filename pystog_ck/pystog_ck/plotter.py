import matplotlib.pyplot as plt


class Plotter:
    """A class to handle plotting of data."""

    @classmethod
    def update_data(
            cls, x, y, data_label, colors,
            x_label, y_label, x_unit, y_unit,
            title=None, tick_ls=18, tick_lg=10,
            axis_ls=20, top_m=0.95, right_m=0.97,
            left_m=0.1, bottom_m=None):
        """Update the class variable with new data and configs."""
        cls.x = x
        cls.y = y
        cls.data_label = data_label
        cls.colors = colors
        cls.x_label = x_label
        cls.y_label = y_label
        cls.x_unit = x_unit
        cls.y_unit = y_unit
        cls.title = title
        cls.tick_ls = tick_ls
        cls.tick_lg = tick_lg
        cls.axis_ls = axis_ls
        cls.top_m = top_m
        cls.right_m = right_m
        cls.left_m = left_m
        cls.bottom_m = bottom_m

    @classmethod
    def plot(cls):
        """Plot the data with specified configs."""
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, xi in enumerate(cls.x):
            ax.plot(
                xi, cls.y[i], label=cls.data_label[i],
                color=cls.colors[i]
            )

        if cls.title:
            ax.set_title(cls.title, fontsize=cls.axis_ls)

        ax.set_xlabel(f"{cls.x_label} ({cls.x_unit})", fontsize=cls.axis_ls)
        if cls.y_unit:
            y_unit_str = f" ({cls.y_unit})"
        else:
            y_unit_str = ""
        ax.set_ylabel(f"{cls.y_label}{y_unit_str}", fontsize=cls.axis_ls)
        ax.tick_params(axis='x', labelsize=cls.tick_ls)
        ax.tick_params(axis='x', length=cls.tick_lg)
        ax.tick_params(axis='y', labelsize=cls.tick_ls)
        ax.tick_params(axis='y', length=cls.tick_lg)
        ax.legend(fontsize=cls.tick_ls)
        fig.subplots_adjust(top=cls.top_m)
        fig.subplots_adjust(right=cls.right_m)
        fig.subplots_adjust(left=cls.left_m)

        return fig, ax
