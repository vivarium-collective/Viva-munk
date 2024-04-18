import os
import shutil
import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt

# from vivarium.plots.agents_multigen import plot_agents_multigen

from pymunk_process.plots.snapshots import (
    make_snapshots_figure,
    make_tags_figure,
    format_snapshot_data,
    get_field_range,
    get_agent_colors,
    get_tag_ranges
)

DEFAULT_HIGHLIGHT_COLOR = [0, 1, 1]
PLOT_WIDTH = 7


def make_snapshot_function(
        data,
        bounds,
        agent_colors=None,
        **kwargs):
    agent_colors = agent_colors or {}
    multibody_agents, multibody_fields = format_snapshot_data(data)

    # make the snapshot plot function
    time_vec = list(multibody_agents.keys())

    # get fields and agent colors
    multibody_field_range = get_field_range(multibody_fields, time_vec)
    multibody_agent_colors = get_agent_colors(multibody_agents)
    multibody_agent_colors.update(agent_colors)

    def plot_single_snapshot(t_index):
        time_indices = np.array([t_index])
        snapshot_time = [time_vec[t_index]]
        fig = make_snapshots_figure(
            time_indices=time_indices,
            snapshot_times=snapshot_time,
            agents=multibody_agents,
            agent_colors=multibody_agent_colors,
            fields=multibody_fields,
            field_range=multibody_field_range,
            n_snapshots=1,
            bounds=bounds,
            default_font_size=12,
            plot_width=PLOT_WIDTH,
            scale_bar_length=0,
            **kwargs)
        return fig

    return plot_single_snapshot, time_vec


def make_tags_function(
        data,
        bounds,
        agent_colors=None,
        tagged_molecules=None,
        tag_colors=None,
        convert_to_concs=False,
        **kwargs
):
    agent_colors = agent_colors or {}
    tag_colors = tag_colors or {}
    multibody_agents, multibody_fields = format_snapshot_data(data)

    # make the snapshot plot function
    time_vec = list(multibody_agents.keys())
    time_indices = np.array(range(0, len(time_vec)))

    # get agent colors, and ranges
    tag_ranges, tag_colors = get_tag_ranges(
        agents=multibody_agents,
        tagged_molecules=tagged_molecules,
        time_indices=time_indices,
        convert_to_concs=convert_to_concs,
        tag_colors=tag_colors)

    # make the function for a single snapshot
    def plot_single_tags(t_index):
        time_index = np.array([t_index])
        snapshot_time = [time_vec[t_index]]
        fig = make_tags_figure(
            time_indices=time_index,
            snapshot_times=snapshot_time,
            agents=multibody_agents,
            agent_colors=agent_colors,
            tagged_molecules=tagged_molecules,
            convert_to_concs=convert_to_concs,
            tag_ranges=tag_ranges,
            tag_colors=tag_colors,
            n_snapshots=1,
            bounds=bounds,
            default_font_size=12,
            plot_width=PLOT_WIDTH,
            scale_bar_length=0,
            **kwargs)
        return fig

    return plot_single_tags, time_vec


def video_from_images(img_paths, out_file):
    # make the video
    img_array = []
    size = None
    for img_file in img_paths:
        img = cv2.imread(img_file)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def make_video(
        data,
        bounds,
        plot_type='fields',
        step=1,
        highlight_agents=None,
        # show_timeseries=None,
        highlight_color=DEFAULT_HIGHLIGHT_COLOR,
        out_dir='out',
        filename='snapshot_vid',
        **kwargs
):
    """Make a video with snapshots across time

    Args:
        plot_type: (str) select either 'fields' or 'tags'. 'fields' is the default
    """
    highlight_agents = highlight_agents or []
    # show_timeseries = show_timeseries or []

    # make images directory, remove if existing
    out_file = os.path.join(out_dir, f'{filename}.mp4')
    out_file2 = os.path.join(out_dir, f'{filename}_timeseries.mp4')
    images_dir = os.path.join(out_dir, f'_images_{plot_type}')
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir)

    agent_colors = None
    if highlight_agents:
        agent_colors = {
            agent_id: highlight_color
            for agent_id in highlight_agents}

    # get the single snapshots function
    if plot_type == 'fields':
        snapshot_fun, time_vec = make_snapshot_function(
            data,
            bounds,
            agent_colors=agent_colors,
            **kwargs)
    elif plot_type == 'tags':
        snapshot_fun, time_vec = make_tags_function(
            data,
            bounds,
            agent_colors=agent_colors,
            **kwargs)


    # make the individual snapshot figures
    img_paths = []
    img_paths_2 = []
    for t_index in range(0, len(time_vec) - 1, step):
        fig_path = os.path.join(images_dir, f"img{t_index}.jpg")
        img_paths.append(fig_path)

        fig = snapshot_fun(t_index)
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close()

    # make the video
    video_from_images(img_paths, out_file)
    video_from_images(img_paths_2, out_file2)

    # delete image folder
    shutil.rmtree(images_dir)

