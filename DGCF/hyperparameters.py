hparams = {
    "ml-1m": {
        "emb_size": 32,
        "num_layers": 1,
        "lr": 1e-2,
        "weight_decay": 6e-6,
        "dropout": 0.3,
        # carry_alpha controls the influence of the previous snapshot's propagated
        # node states. 0.0 == no carry-over, 1.0 == full carry-over.
        "carry_alpha": 0.5,
        # use_agate enables the learnable AGate fusion module instead of simple
        # additive carry-over. When True, the model will learn a per-dimension
        # gate to merge previous and current embeddings.
        "use_agate": True
    },
    "ml-100k": {
        "emb_size": 32,
        "num_layers": 1,
        "lr": 8e-2,
        "weight_decay": 1e-4,
        "dropout": 0.,
        "carry_alpha": 0.5,
        "use_agate": False
    },
    "flixster": {
        "emb_size": 32,
        "num_layers": 1,
        "lr": 7e-3,
        "weight_decay": 6e-4,
        "dropout": 0.3,
        "carry_alpha": 0.5,
        "use_agate": False
    },
    "douban": {
        "emb_size": 16,
        "num_layers": 2,
        "lr": 1e-2,
        "weight_decay": 3e-5,
        "dropout": 0.1,
        "carry_alpha": 0.5,
        "use_agate": False
    },
    "yahoo_music": {
        "emb_size": 16,
        "num_layers": 1,
        "lr": 5e-2,
        "weight_decay": 9e-4,
        "dropout": 0.2,
        "carry_alpha": 0.5,
        "use_agate": False
    }
}
