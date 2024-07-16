API
===
Import SpaRED::

    import spared


Datasets
~~~~~~~~~~~~~~~

.. module:: spared.datasets
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    datasets.get_dataset

Filtering
~~~~~~~~~~~~~~~

.. module:: spared.filtering
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    filtering.filter_by_moran
    filtering.filter_dataset
    filtering.get_slide_from_collection
    filtering.get_slides_adata


Gene Features
~~~~~~~~~~~~~~~

.. module:: spared.gene_features
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    gene_features.get_exp_frac
    gene_features.get_glob_exp_frac
    gene_features.compute_moran


Spot Features
~~~~~~~~~~~~~~~

.. module:: spared.spot_features
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    spot_features.compute_patches_embeddings
    spot_features.compute_patches_predictions
    spot_features.compute_dim_red
    spot_features.get_spatial_neighbors


Layer Operations
~~~~~~~~~~~~~~~

.. module:: spared.layer_operations
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    layer_operations.tpm_normalization
    layer_operations.log1p_transformation
    layer_operations.combat_transformation
    layer_operations.get_deltas
    layer_operations.add_noisy_layer
    layer_operations.process_dataset


Denoising
~~~~~~~~~~~~~~~

.. module:: spared.denoising
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    denoising.median_cleaner
    denoising.spackle_cleaner


Graph Operations
~~~~~~~~~~~~~~~

.. module:: spared.graph_operations
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    graph_operations.get_graphs_one_slide
    graph_operations.get_sin_cos_positional_embeddings
    graph_operations.get_graphs


Plotting
~~~~~~~~~~~~~~~

.. module:: spared.plotting
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    plotting.plot_all_slides
    plotting.plot_exp_frac
    plotting.plot_histograms
    plotting.plot_random_patches
    plotting.visualize_moran_filtering
    plotting.visualize_gene_expression
    plotting.plot_clusters
    plotting.plot_mean_std
    plotting.plot_data_distribution_stats
    plotting.plot_mean_std_partitions
    plotting.plot_tests


Dataloaders
~~~~~~~~~~~~~~~

.. module:: spared.dataloaders
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    dataloaders.get_pretrain_dataloaders
    dataloaders.get_graph_dataloaders


Models
~~~~~~~~~~~~~~~

.. module:: spared.models
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    models.ImageEncoder


Metrics
~~~~~~~~~~~~~~~

.. module:: spared.metrics
.. currentmodule:: spared

.. autosummary::
    :toctree: api

    metrics.get_pearsonr
    metrics.get_r2_score
    metrics.get_metrics