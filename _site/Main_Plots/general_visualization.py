import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pyanatomogram
import os

# Read database metadata
metadata = pd.read_csv('Human_Mouse_database.csv', index_col=0)
metadata['Datasets'] = 1

human_metadata = metadata.loc[metadata.Organism=='Human', ['Tissue_vis', 'Slides', 'Patients', 'Spots', 'Datasets']]
mouse_metadata = metadata.loc[metadata.Organism=='Mouse', ['Tissue_vis', 'Slides', 'Patients', 'Spots', 'Datasets']]

grouped_human_metadata = human_metadata.groupby('Tissue_vis').sum()
grouped_mouse_metadata = mouse_metadata.groupby('Tissue_vis').sum()

grouped_human_metadata.sort_values(by='Spots', inplace=True)
grouped_mouse_metadata.sort_values(by='Spots', inplace=True)

# Plot list labels
labels = ['Datasets', 'Patients', 'Slides', 'Spots']

fig, axes = plt.subplots(nrows=2, ncols=4)
fig.set_size_inches(20, 9)

cmap = matplotlib.cm.Purples
purple = cmap(1.0)

for i, lab in enumerate(labels):

    curr_series = grouped_human_metadata[lab]
    curr_series.plot(kind='barh', ax=axes[1, i], color=purple)
    axes[1, i].bar_label(axes[1, i].containers[0], padding=3, fontsize='large')

    axes[1, i].set_xlabel(lab, fontsize='xx-large')
    axes[1, i].spines[['right', 'top']].set_visible(False)
    axes[1, i].tick_params(axis='both', labelsize='x-large')
    if i == 0:
        axes[1, i].set_ylabel('Organ', fontsize='xx-large')
    else:
        axes[1, i].set_ylabel(None)

    norm = matplotlib.colors.Normalize()
    norm.autoscale(curr_series)
    norm.vmin = 0 
    anatomogram = pyanatomogram.Anatomogram('homo_sapiens.male')
    anatomogram.highlight_tissues(curr_series.to_dict(), cmap=cmap, norm=norm)
    anatomogram.to_matplotlib(ax=axes[0, i])

    axes[0, i].set_title(lab, fontsize='xx-large')
    cb = matplotlib.pyplot.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0, i])
    cb.set_label(label=None)

fig.suptitle('General Database Statistics Human Data', fontsize=30)
fig.tight_layout()
fig.savefig('human_statistics.png', dpi=300)


### Mouse

fig, axes = plt.subplots(nrows=2, ncols=4)
fig.set_size_inches(20, 9)


for i, lab in enumerate(labels):

    curr_series = grouped_mouse_metadata[lab]
    curr_series.plot(kind='barh', ax=axes[1, i], color=purple)
    axes[1, i].bar_label(axes[1, i].containers[0], padding=3, fontsize='large')

    axes[1, i].set_xlabel(lab, fontsize='xx-large')
    axes[1, i].spines[['right', 'top']].set_visible(False)
    axes[1, i].tick_params(axis='both', labelsize='x-large')
    if i == 0:
        axes[1, i].set_ylabel('Organ', fontsize='xx-large')
    else:
        axes[1, i].set_ylabel(None)

    norm = matplotlib.colors.Normalize()
    norm.autoscale(curr_series)
    norm.vmin = 0 
    anatomogram = pyanatomogram.Anatomogram('mus_musculus.male')
    anatomogram.highlight_tissues(curr_series.to_dict(), cmap=cmap, norm=norm)
    anatomogram.to_matplotlib(ax=axes[0, i])

    axes[0, i].set_title(lab, fontsize='xx-large')
    cb = matplotlib.pyplot.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0, i])
    cb.set_label(label=None)

fig.suptitle('General Database Statistics Mouse Data', fontsize=30)
fig.tight_layout()
fig.savefig('mouse_statistics.png', dpi=300)


