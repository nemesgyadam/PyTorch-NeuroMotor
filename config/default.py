cfg = dict(
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
        in_chans=22,
        n_classes=4,
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
    ),
    train=dict(
        batch_size=48,
        normalize=True,
        n_epochs=200,
        learning_rate=1e-3,
        weight_decay=0.02,
    )


)