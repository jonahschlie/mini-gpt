class GPTConfig:
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for key, value in kwargs.items():
            setattr(self, key, value)

class OwnConfig(GPTConfig):
    # architecture
    n_layer = 8
    n_head = 8
    n_embd = 64

    # dropouts
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    dropout = 0.1

    # training
    use_torch_compile = True
    device = "mps"
    dataloader_num_workers = 1
    max_epochs = 30
    batch_size = 32
    block_size = 128
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    weight_decay = 1e-1
    grad_clip = 1.0

    # Speed/Control
    max_steps_per_epoch = 3000
    eval_interval_epochs = 1
    eval_subset_batches = None

    # Early stopping
    early_stopping_patience = 3

    # Safe weights
    save_dir = 'weights'
    save_best_weights = True