def set_to_debug_mode(cfg):
    for key in ['train', 'val', 'test']:
        cfg[key].data_loader.kwargs.num_workers = 0
        # cfg[key].data_loader.kwargs.batch_size = 16
    # cfg['val'].data_loader.kwargs.batch_size = 1
        # cfg[key].data_loader.kwargs.prefetch_factor = 0
    return cfg
