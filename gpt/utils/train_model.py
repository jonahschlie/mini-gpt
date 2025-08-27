from torch.utils.data import DataLoader
from contextlib import nullcontext
from gpt.utils.dataset import ShakespeareDataset
import torch
import os

def ensure_dir(path: str):
    """Create directory if it doesn’t exist."""
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def train_model(model, train_data, valid_data, test_data, config, device):
    # --- read config values (with defaults) ---
    block_size = getattr(config, 'block_size', 64)
    batch_size = getattr(config, 'batch_size', 32)
    max_epochs = getattr(config, 'max_epochs', 10)
    max_steps_per_epoch = getattr(config, 'max_steps_per_epoch', None)
    eval_every_epochs = getattr(config, 'eval_interval_epochs', 2)
    eval_subset_batches = getattr(config, 'eval_subset_batches', 50)
    patience = getattr(config, 'early_stopping_patience', 3)
    learning_rate = getattr(config, 'learning_rate', 6e-4)
    betas = getattr(config, 'betas', (0.9, 0.95))
    weight_decay = getattr(config, 'weight_decay', 1e-1)

    # checkpoint / save settings
    save_dir = getattr(config, 'save_dir', 'checkpoints')
    save_best_weights = getattr(config, 'save_best_weights', True)
    ensure_dir(save_dir)

    # --- datasets & loaders ---
    train_ds = ShakespeareDataset(train_data, block_size)
    val_ds   = ShakespeareDataset(valid_data, block_size)
    test_ds  = ShakespeareDataset(test_data, block_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # --- optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )

    # use mixed precision (float16) on CUDA/MPS for speed/memory savings;
    # fall back to normal precision (nullcontext) on CPU
    if device.type in ('cuda', 'mps'):
        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16)
    else:
        amp_ctx = nullcontext()

    # track best val loss for early stopping
    best_val = float('inf')
    best_epoch = -1
    global_step = 0

    steps_per_epoch_est = len(train_loader) if max_steps_per_epoch is None else min(len(train_loader),
                                                                                    max_steps_per_epoch)
    print(f"Info - steps_per_epoch ≈ {steps_per_epoch_est}, eval every {eval_every_epochs} epochs")

    train_losses = []
    val_losses = []

    # --- training loop ---
    for epoch in range(1, max_epochs + 1):
        model.train()
        step_in_epoch = 0
        train_total = 0.0
        train_steps = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                _, loss = model(x, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, 'grad_clip', 1.0))
            optimizer.step()

            global_step += 1
            step_in_epoch += 1
            train_total += loss.item()
            train_steps += 1

            # occasional progress log
            if step_in_epoch % 200 == 0:
                print(f"epoch {epoch} step {step_in_epoch}/{steps_per_epoch_est} "
                      f"(global {global_step}): loss={loss.item():.4f}")

            # respect max_steps_per_epoch if set
            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                break

        # end of epoch
        train_epoch_loss = train_total / max(1, train_steps)
        train_losses.append(train_epoch_loss)
        print(f"TRAIN - epoch {epoch}/{max_epochs} — loss={train_epoch_loss:.4f}")

        # validation
        if epoch % eval_every_epochs == 0:
            model.eval()
            val_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for i, (xv, yv) in enumerate(val_loader):
                    if eval_subset_batches is not None and i >= eval_subset_batches:
                        break
                    xv = xv.to(device, non_blocking=True)
                    yv = yv.to(device, non_blocking=True)
                    with amp_ctx:
                        _, vloss = model(xv, yv)
                    val_total += vloss.item()
                    val_steps += 1

            val_loss = val_total / max(1, val_steps)
            val_losses.append(val_loss)
            print(f"VAL - epoch {epoch}/{max_epochs} — val_loss={val_loss:.4f}")

            # early stopping
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                if save_best_weights:
                    best_w_path = os.path.join(save_dir, "best_weights.pt")
                    model.save_weights(best_w_path)
                    print(f"Save - best weights -> {best_w_path}")
            elif epoch - best_epoch >= patience:
                print(f"EarlyStop - epoch {epoch}: No improvement for {patience} epochs "
                      f"(best={best_val:.4f} at epoch {best_epoch})")
                break

    # final test evaluation
    model.eval()
    test_total = 0.0
    test_steps = 0
    with torch.no_grad():
        for i, (xt, yt) in enumerate(test_loader):
            if eval_subset_batches is not None and i >= eval_subset_batches:
                break
            xt = xt.to(device, non_blocking=True)
            yt = yt.to(device, non_blocking=True)
            with amp_ctx:
                _, tloss = model(xt, yt)
            test_total += tloss.item()
            test_steps += 1
    print(f"TEST - loss={test_total / max(1, test_steps):.4f}")

    # store weights & return losses + loaders for downstream use
    result_paths = {
        "best_weights": os.path.join(save_dir, "best_weights.pt") if save_best_weights else None,
    }
    return train_losses, val_losses, result_paths, (val_loader, test_loader)