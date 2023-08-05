# Contains subject over 60% (per subject) accuracy
cfg = dict(
    data=dict(
        subjects =      [1,2,3,7,8],
        train_runs = {
                        1:[0, 1, 2, 3, 4],
                        2:[0, 1, 2, 3, 4],
                        3:[0, 1, 2, 3, 4],
                        4:[0, 1],
                        5:[0, 1, 2, 3, 4],
                        6:[0, 1, 2, 3, 4],
                        7:[0, 1, 2, 3, 4],
                        8:[0, 1, 2, 3, 4],
                        9:[0, 1, 2, 3, 4]
        },            
        test_runs = {
                        1:[5],
                        2:[5],
                        3:[5],
                        4:[2],
                        5:[5],
                        6:[5],
                        7:[5],
                        8:[5],
                        9:[5]
        }
    ),
    preprocessing=dict(
        target_freq=100,
        low_freq=8,
        high_freq=25,
        average_ref=True,
    ),
    epochs=dict(
        baseline=(-0.1, 1.9),
        tmin=-0.1,
        tmax=5.9,
    ),
    model=dict(
        num_subjects= 5,            # Number of subjects
        in_chans= 22,               # Number of input channels
        n_samples= 401,             # Number of samples
        n_classes= 4,               # Number of classes
        n_filters_time= 25,         # Number of filters in the temporal convolution
        norm_rate = 0.25,           # Normalization rate
        filter_time_length= 10,     # Length of the temporal convolution filters
        n_filters_spat = 25,        # Number of filters in the spatial convolution
        dropout_rate= 0.5,          # Dropout rate
        depth_multiplier= 2,        # Depth multiplier for depthwise convolution
        embedding_dim= 8,           # Dimension of the embedding for the attention mechanism
        n_filters3= 64,             # Number of filters in the final convolution layer
        weight_init_std= 0.2        # Standard deviation for weight initialization
        ),
    train=dict(
        batch_size=64,
        normalize=True,
        n_epochs=100,
        learning_rate=1e-3,
        weight_decay=0.02,
    )


)