import os

import numpy as np
from imageio import v2 as imageio
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


class LineWidthData(Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop('linewidth', 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72. / self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data)) - trans((0, 0))) * ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def simulation_to_gif(data, config, filename='simulation.gif', skip_frames=1):
    if not os.path.exists('frames'):
        os.makedirs('frames')
    env_size = config['env_size']
    barriers = config.get('barriers', [])

    images = []
    for index, step in enumerate(data[::skip_frames]):
        fig, ax = plt.subplots()
        ax.set_xlim(0, env_size)
        ax.set_ylim(0, env_size)
        ax.set_aspect('equal')

        # Draw barriers
        for barrier in barriers:
            start_x, start_y = barrier['start']
            end_x, end_y = barrier['end']
            thickness = barrier.get('thickness', 1)  # Default thickness
            barrier_line = Line2D([start_x, end_x], [start_y, end_y], linewidth=thickness, color='gray')
            ax.add_line(barrier_line)

        # Draw agents
        for agent_id, obj in step['agents'].items():
            if obj.get('type') == 'circle':
                circle = Circle((obj['location'][0], obj['location'][1]),
                                obj['radius'],
                                fill=True)
                ax.add_patch(circle)
            elif obj.get('type') == 'segment':
                length = obj['length']
                thickness = obj['radius'] #* 2  # Visual thickness of the line
                angle = obj['angle']

                dx = np.cos(angle) * length / 2
                dy = np.sin(angle) * length / 2
                start_point = (obj['location'][0] - dx, obj['location'][1] - dy)
                end_point = (obj['location'][0] + dx, obj['location'][1] + dy)

                line = Line2D([start_point[0], end_point[0]],
                              [start_point[1], end_point[1]],
                              linewidth=thickness,
                              solid_capstyle='round')
                ax.add_line(line)

        ax.set_title(f"Time = {step['time']:.1f}")
        frame_filename = f'frames/frame_{index:04d}.png'
        plt.savefig(frame_filename)
        plt.close(fig)
        images.append(imageio.imread(frame_filename))

    imageio.mimsave(filename, images, duration=0.1, loop=0)

    # Clean up the frames directory
    for frame_filename in os.listdir('frames'):
        os.remove(f'frames/{frame_filename}')
    os.rmdir('frames')

    # Output file saved message
    print(f"GIF saved to {filename}")
