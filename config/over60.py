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
        # General params
        n_classes= 4,               # Number of classes
        n_subjects= 5,            # Number of subjects
        in_chans= 22,               # Number of input channels
        n_samples= 401,             # Number of samples

        # EEGNet params
        kernel_length = 64,
        n_filters1 = 16,
        depth_multiplier = 2,
        n_filters2 = 32,
        dropout_rate= 0.2,


        # Conditioned EEGNet params
        embed_dim = 16,
        weight_init_std= None        # Standard deviation for weight initialization

        ),
    train=dict(
        batch_size=64,
        normalize=True,
        n_epochs=100,
        learning_rate=1e-3,
        weight_decay=0.02,
    )


)