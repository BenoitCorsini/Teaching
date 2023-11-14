import sys
import argparse
import os
import os.path as osp
import numpy as np
import numpy.random as npr
sys.path.append('../../../')
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib.textpath import TextPath

import probability


N_TILES = 8
CMAP = probability.plot.visual.Visual.get_cmap([
    (255/255, 255/255, 255/255),
    ( 64/255, 149/255, 191/255),
    (  0/255,   0/255,   0/255),
])
PARAMS = {
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


# https://www.rapidtables.com/code/text/unicode-characters.html
class ChessPiece(object):

    def __init__(self, char):
        self.char = char
        self.counting = {}
        self.pos = 0, 0
        self.counting[self.pos] = 1

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

    def steps(self, jumps):
        for _ in range(jumps):
            self.step()

    @classmethod
    def get_path(piece, i=0, j=0, inner_ratio=1):
        path = TextPath((0, 0), piece().char)
        bbox = path.get_extents()
        size = max(bbox.size)
        transform = Affine2D()
        tx, ty =  (size - bbox.size)/2 - bbox.p0
        transform.translate(tx, ty)
        transform.scale(2*inner_ratio/size/N_TILES)
        transform.translate(2*i/N_TILES - 1 + (1 - inner_ratio)/N_TILES, 2*j/N_TILES - 1 + (1 - inner_ratio)/N_TILES)
        path = path.transformed(transform)
        return path

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

class Project(probability.plot.visual.Visual):

    def __init__(self, steps=1, jumps=1, **kwargs):
        super().__init__(**PARAMS)
        self.steps = steps
        self.jumps = jumps

    def tile_to_xy(self, i, j, simple_dim=False):
        if simple_dim:
            return np.array([2*i/N_TILES - 1, 2*j/N_TILES - 1])
        else:
            return np.array([[2*i/N_TILES - 1, 2*j/N_TILES - 1]])

    def setup_chess(self, piece):
        self.frame.set_visible(False)
        self.tiles = {}
        self.pieces = {}
        self.bg_pieces = {}
        for i in range(N_TILES):
            for j in range(N_TILES):
                self.tiles[i, j] = self.plot_shape(
                    shape_name='Rectangle',
                    xy=self.tile_to_xy(i, j, True),
                    height=2/N_TILES,
                    width=2/N_TILES,
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
            # self.tiles[i, j].set_color(self.cmap(ratio*(1 - counts[i,j])*((i + j) % 2) + counts[i, j]))
            self.pieces[i, j].set_visible(False)
            self.bg_pieces[i, j].set_visible(False)

    def image(self, piece, *args, **kwargs):
        self.reset()
        self.setup_chess(piece)
        self.update_tiles(piece.counting, 0)
        self.pieces[0, 0].set_visible(True)
        self.bg_pieces[0, 0].set_visible(True)

        self.save(name='piece_' + piece.__class__.__name__.lower())
        print('Time to plot image: ' + self.time())

    def animate(self, piece, *args, **kwargs):
        self.reset()
        self.setup_chess(piece)
        self.update_tiles(piece.counting, 1)
        self.pieces[0, 0].set_visible(True)
        self.bg_pieces[0, 0].set_visible(True)
        for _ in range(int(self.fps*self.times['initial'])):
            self.new_frame()
        n_steps = 1
        for step in range(self.steps):
            n_steps *= self.jumps
            ratio = (self.disappear_ratio - step/self.steps)/self.disappear_ratio
            if ratio < 0:
                ratio = 0
            self.update_tiles(piece.counting, ratio)
            self.pieces[piece.pos].set_visible(True)
            self.bg_pieces[piece.pos].set_visible(True)
            self.new_frame()
            piece.steps(int(n_steps))
        self.update_tiles(piece.counting, 0)
        self.pieces[0, 0].set_visible(True)
        self.bg_pieces[0, 0].set_visible(True)
        for _ in range(int(self.fps*self.times['final'])):
            self.new_frame()

        sys.stdout.write('\033[F\033[K')
        print(f'Time to create all frames ({self.frame_counter}): ' + self.time())
        print('Making the video...')
        self.frames_to_video(name='piece_' + piece.__class__.__name__.lower())
        sys.stdout.write('\033[F\033[K')
        print('Time to make video: ' + self.time())

    def anonymized(self, piece):
        self.reset()
        self.setup_chess(piece)
        piece.steps(self.steps*self.jumps)
        self.update_tiles(piece.counting, 0)
        self.save(name='piece_' + piece.__class__.__name__.lower() + f'_{self.steps}_anonymized')
        self.pieces[0, 0].set_visible(True)
        self.bg_pieces[0, 0].set_visible(True)

        self.save(name='piece_' + piece.__class__.__name__.lower() + f'_{self.steps}')
        print('Time to plot image: ' + self.time())

    def moves(self, piece):
        self.reset()
        self.setup_chess(piece)
        i, j = 3, 3
        counting = {n : 1 for n in piece.neighbours(i, j)}
        self.update_tiles(counting, 0.5)
        self.pieces[i, j].set_visible(True)
        self.bg_pieces[i, j].set_visible(True)
        self.save(name='moves_' + piece.__class__.__name__.lower())
        self.update_tiles({}, 0.5)
        self.pieces[i, j].set_visible(True)
        self.bg_pieces[i, j].set_visible(True)
        self.save(name='no_moves_' + piece.__class__.__name__.lower())
        print('Time to plot image: ' + self.time())

    def run(self, piece):
        # self.animate(piece)
        # self.image(piece)
        self.anonymized(piece)
        # self.moves(piece)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--jumps', type=int, default=1.05)
    kwargs = vars(parser.parse_args())
    Project(**kwargs).run(King())
    Project(**kwargs).run(Queen())
    Project(**kwargs).run(Rook())
    Project(**kwargs).run(Bishop())
    Project(**kwargs).run(Knight())
