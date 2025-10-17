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


def _ensure_gif_filename(path):
    root, ext = os.path.splitext(path)
    return f"{path}.gif" if ext == "" else path

def simulation_to_gif(
        data,
        config,
        agents_key='agents',
        filename='simulation.gif',
        skip_frames=1,
        out_dir='out'
):
    # prepare output dirs
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, '_frames_tmp')
    os.makedirs(frames_dir, exist_ok=True)

    env_size = config['env_size']
    barriers = config.get('barriers', [])

    images = []
    try:
        for index, step in enumerate(data[::skip_frames]):
            fig, ax = plt.subplots()
            ax.set_xlim(0, env_size)
            ax.set_ylim(0, env_size)
            ax.set_aspect('equal')

            # Draw barriers (world-units thickness)
            for barrier in barriers:
                start_x, start_y = barrier['start']
                end_x, end_y = barrier['end']
                thickness = barrier.get('thickness', 1)
                barrier_line = LineWidthData([start_x, end_x], [start_y, end_y],
                                             linewidth=thickness, color='gray')
                ax.add_line(barrier_line)

            # Draw agents
            for agent_id, obj in step[agents_key].items():
                if obj.get('type') == 'circle':
                    circle = Circle((obj['location'][0], obj['location'][1]),
                                    obj['radius'],
                                    fill=True)
                    ax.add_patch(circle)
                elif obj.get('type') == 'segment':
                    length = obj['length']
                    r = obj['radius']
                    angle = obj['angle']

                    dx = np.cos(angle) * length / 2
                    dy = np.sin(angle) * length / 2
                    start_point = (obj['location'][0] - dx, obj['location'][1] - dy)
                    end_point   = (obj['location'][0] + dx, obj['location'][1] + dy)

                    line = LineWidthData([start_point[0], end_point[0]],
                                         [start_point[1], end_point[1]],
                                         linewidth=2 * r,  # diameter in world units
                                         solid_capstyle='round')
                    ax.add_line(line)

            ax.set_title(f"Time = {step['time']:.1f}")
            frame_filename = os.path.join(frames_dir, f'frame_{index:04d}.png')
            plt.savefig(frame_filename)
            plt.close(fig)
            images.append(imageio.imread(frame_filename))

        # decide output path
        filename = _ensure_gif_filename(filename)
        out_path = filename if os.path.dirname(filename) else os.path.join(out_dir, filename)

        if not images:
            raise ValueError("No frames to write (images list is empty).")

        imageio.mimsave(out_path, images, duration=0.1, loop=0, format="GIF")
        print(f"GIF saved to {out_path}")

    finally:
        # Clean up temp frames even on error
        if os.path.isdir(frames_dir):
            for f in os.listdir(frames_dir):
                try:
                    os.remove(os.path.join(frames_dir, f))
                except FileNotFoundError:
                    pass
            try:
                os.rmdir(frames_dir)
            except OSError:
                # directory not empty or race; ignore
                pass