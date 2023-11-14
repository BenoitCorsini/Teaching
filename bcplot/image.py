import matplotlib.patches as patches

from .figure import Figure


class Image(Figure):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_shape(self, shape_name, **kwargs):
        patch = getattr(patches, shape_name)(**kwargs)
        self.ax.add_patch(patch)
        return patch

    def grid(self, n_blocks=10, **kwargs):
        columns = []
        rows = []
        kwargs['capstyle'] = 'butt'
        delta = 2/n_blocks
        for i in range(0, n_blocks+1):
            pos = i*delta - 1
            column_patch = patches.Polygon(
                [[pos,-1], [pos,1]],
                **kwargs
            )
            self.ax.add_patch(column_patch)
            columns.append(column_patch)
            row_patch = patches.Polygon(
                [[-1,pos], [1,pos]],
                **kwargs
            )
            self.ax.add_patch(row_patch)
            rows.append(row_patch)
        return columns, rows