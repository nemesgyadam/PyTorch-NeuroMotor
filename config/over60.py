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
        n_subjects= 5,              # Number of subjects

        # EEGNet parameters
        n_classes= 4,               # Number of classes
        in_channels= 22,            # Number of input channels
        in_timesteps= 401,          # Number of input time steps
       
        n_time_filters= 25,         # Number of filters in the temporal convolution
        time_filter_length= 10,     # Length of the temporal convolution filters
        
        n_depth_filters = 25,       # Number of filters in the depthwise convolution
        
        n_sep_filters= 64,          # Number of filters in the separable convolution layer
        
        dropout_rate= 0.5,          # Dropout rate
        weight_init_std= 0.2,       # Standard deviation for weight initialization
        # Subject Encoder params
        n_time_filters_subject = 2, #(for attn_cond_eegnet_subjectFeatures) Number of filters in the temporal convolution
        n_depth_filters_subject = 2,#(for attn_cond_eegnet_subjectFeatures) Number of filters in the depthwise convolution
        n_sep_filters_subject = 2,  #(for attn_cond_eegnet_subjectFeatures) Number of filters in the separable convolution layer
        subject_dim = 12,           #(for attn_cond_eegnet_subjectFeatures only) Number of features in the subject encoder
        # Concat EEGnet parameters
        subject_filters = 16,       # Number of filters in the subject encoder
        final_features = 4,         # Number of features after concatenation
        
        # Other parameters
        embed_dim= 6,               # Dimension of the embedding for the attention mechanism
        ),
    train=dict(
        batch_size=64,
        normalize=True,
        n_epochs=100,
        learning_rate=5e-3,
        weight_decay=0.02,
    )


)