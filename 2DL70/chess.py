import sys
import numpy as np
import numpy.random as npr

sys.path.append('../')
from bcplot import BCPlot as plot

N_TILES = 8
CMAP = plot.get_cmap([
    (255/255, 255/255, 255/255),
    ( 64/255, 149/255, 191/255),
    (  0/255,   0/255,   0/255),
])
PARAMS = {
    'xmax' : N_TILES,
    'ymax' : N_TILES,
    'colour_ratio' : 0.8,
    'inner_ratio' : 0.7,
    'disappear_ratio' : 0.3,
    'tile_params' : {
        'lw' : 0,
        'zorder' : 0,
    },
    'grid_params' : {
        'color' : 'black',
        'lw' : 2,
        'zorder' : 1,
    },
    'piece_params' : {
        'lw' : 0,
        'color' : CMAP(0.9),
        'capstyle' : 'round',
        'joinstyle' : 'round',
        'zorder' : 3,
    },
    'bg_piece_params' : {
        'lw' : 4,
        'color' : CMAP(0.1),
        'capstyle' : 'round',
        'joinstyle' : 'round',
        'zorder' : 2,
    },
    'times' : {
        'initial' : 0,
        'final' : 1,
    },
}


class ChessPiece(object):

    def __init__(self, char):
        self.char = char
        self.name = self.__class__.__name__.lower()
        self.counting = {}
        self.pos = 0, 0
        self.counting[self.pos] = 1
        self.n_steps = 0

    def reset(self):
        self.counting = {}
        self.pos = 0, 0
        self.counting[self.pos] = 1
        self.n_steps = 0

    @staticmethod
    def clean(neighbours):
        return [(i, j) for (i, j) in neighbours if (i >= 0) and (i < N_TILES) and (j >= 0) and (j < N_TILES)]

    def step(self):
        step = self.neighbours(*self.pos)
        step = step[npr.randint(len(step))]
        if step not in self.counting:
            self.counting[step] = 0
        self.counting[step] += 1
        self.pos = step
        self.n_steps += 1

    def steps(self, jumps):
        for _ in range(jumps):
            self.step()

    def get_path(self, inner_ratio=1, i=0, j=0):
        return plot.path_from_string(
            self.char,
            ratio=inner_ratio,
            x=i + 0.5,
            y=j + 0.5,
        )


class King(ChessPiece):

    def __init__(self, char='\u265A'):
        super().__init__(char)

    def neighbours(self, i, j):
        return self.clean([
            (i, j + 1),
            (i, j - 1),
            (i + 1, j),
            (i - 1, j),
            (i + 1, j + 1),
            (i + 1, j - 1),
            (i - 1, j + 1),
            (i - 1, j - 1),
        ])

class Queen(ChessPiece):

    def __init__(self, char='\u265B'):
        super().__init__(char)

    def neighbours(self, i, j):
        return self.clean(
            [(i, j + k) for k in range(1, N_TILES)]
            + [(i, j - k) for k in range(1, N_TILES)]
            + [(i + k, j) for k in range(1, N_TILES)]
            + [(i - k, j) for k in range(1, N_TILES)]
            + [(i + k, j + k) for k in range(1, N_TILES)]
            + [(i + k, j - k) for k in range(1, N_TILES)]
            + [(i - k, j + k) for k in range(1, N_TILES)]
            + [(i - k, j - k) for k in range(1, N_TILES)]
        )

class Rook(ChessPiece):

    def __init__(self, char='\u265C'):
        super().__init__(char)

    def neighbours(self, i, j):
        return self.clean(
            [(i, j + k) for k in range(1, N_TILES)]
            + [(i, j - k) for k in range(1, N_TILES)]
            + [(i + k, j) for k in range(1, N_TILES)]
            + [(i - k, j) for k in range(1, N_TILES)]
        )

class Bishop(ChessPiece):

    def __init__(self, char='\u265D'):
        super().__init__(char)

    def neighbours(self, i, j):
        return self.clean(
            [(i + k, j + k) for k in range(1, N_TILES)]
            + [(i + k, j - k) for k in range(1, N_TILES)]
            + [(i - k, j + k) for k in range(1, N_TILES)]
            + [(i - k, j - k) for k in range(1, N_TILES)]
        )

class Knight(ChessPiece):

    def __init__(self, char='\u265E'):
        super().__init__(char)

    def neighbours(self, i, j):
        return self.clean([
            (i + 2, j + 1),
            (i + 2, j - 1),
            (i - 2, j + 1),
            (i - 2, j - 1),
            (i + 1, j + 2),
            (i + 1, j - 2),
            (i - 1, j + 2),
            (i - 1, j - 2),
        ])


class ChessPlot(plot):

    def __init__(self):
        super().__init__(**PARAMS)

    def setup_chess(self, piece):
        self.tiles = {}
        self.pieces = {}
        self.bg_pieces = {}
        for i in range(N_TILES):
            for j in range(N_TILES):
                self.tiles[i, j] = self.plot_shape(
                    shape_name='Rectangle',
                    xy=[i, j],
                    height=1,
                    width=1,
                    **self.tile_params
                )
                path = piece.get_path(i=i, j=j, inner_ratio=self.inner_ratio)
                self.pieces[i, j] = self.plot_shape(
                    shape_name='PathPatch',
                    path=path,
                    **self.piece_params
                )
                self.bg_pieces[i, j] = self.plot_shape(
                    shape_name='PathPatch',
                    path=path,
                    **self.bg_piece_params
                )
        self.grid(N_TILES, **self.grid_params)
        self.update_tiles()

    def cmap(self, value):
        return CMAP(value*self.colour_ratio + (1 - self.colour_ratio)/2)

    def update_tiles(self, counting={}, ratio=1):
        counts = np.zeros((N_TILES, N_TILES))
        for (i, j), c in counting.items():
            counts[i, j] = c
        if np.std(counts):
            counts = counts/np.max(counts)
            mean = np.mean(counts)
            if mean > 0.5:
                counts = 0.5*counts/mean
            elif mean < 0.5:
                counts = 1 - (1 - counts)*0.5/(1 - mean)
        else:
            counts = 0.5*np.ones_like(counts)
        for (i, j), tile in self.tiles.items():
            self.tiles[i, j].set_color(self.cmap(ratio*((i + j) % 2) + counts[i, j]*(1 - ratio)))
            self.pieces[i, j].set_visible(False)
            self.bg_pieces[i, j].set_visible(False)

    def file_name(self, piece, secret=False):
        name = f'chess_{piece.name}_{piece.n_steps}'
        if secret:
            name += '_secret'
        return name

    def image(self, piece, secret=False):
        self.reset()
        self.setup_chess(piece)
        self.update_tiles(piece.counting, 0)
        self.pieces[0, 0].set_visible(not secret)
        self.bg_pieces[0, 0].set_visible(not secret)
        self.save_image(name=self.file_name(piece, secret))

    def video(self, piece, steps=0, jumps=1, secret=False):
        self.reset()
        self.setup_chess(piece)
        self.update_tiles(piece.counting, 1)
        self.pieces[piece.pos].set_visible(not secret)
        self.bg_pieces[piece.pos].set_visible(not secret)
        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        n_steps = 1
        for step in range(steps):
            n_steps *= jumps
            ratio = (self.disappear_ratio - step/steps)/self.disappear_ratio
            if ratio < 0:
                ratio = 0
            self.update_tiles(piece.counting, ratio)
            self.pieces[piece.pos].set_visible(not secret)
            self.bg_pieces[piece.pos].set_visible(not secret)
            self.new_frame()
            piece.steps(int(n_steps))
        self.update_tiles(piece.counting, 0)
        self.pieces[0, 0].set_visible(not secret)
        self.bg_pieces[0, 0].set_visible(not secret)
        for _ in range(int(self.fps*self.times['final'])):
            self.new_frame()
        self.save_video(name=self.file_name(piece, secret))

    def run(self, piece, steps=0, jumps=0, secret=False, seed=None):
        npr.seed(seed)
        n_steps = 1
        for step in range(steps):
            n_steps *= jumps
            piece.steps(int(n_steps))
        self.image(piece, secret=secret)
        piece.reset()
        npr.seed(seed)
        self.video(piece, steps=steps, jumps=jumps, secret=secret)
        piece.reset()


if __name__ == '__main__':
    CP = ChessPlot()
    CP.new_param('--seed', type=int, default=None)
    CP.new_param('--steps', type=int, default=180)
    CP.new_param('--jumps', type=int, default=1.05)
    CP.new_param('--secret', type=int, default=0)
    for piece in [King(), Queen(), Rook(), Bishop(), Knight()]:
        CP.run(piece, **CP.get_kwargs())