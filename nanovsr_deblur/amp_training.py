"""Bounded, same-batch AMP recovery. Never advance training on a failed attempt."""

import json
import math

import torch


AMP_POLICY = 'same_batch_retry_v1'
INIT_SCALE = 4096.0
GROWTH_INTERVAL = 10000
MAX_OVERFLOW_RETRIES = 8  # Initial attempt plus at most eight retries.


def make_scaler(enabled):
    return torch.cuda.amp.GradScaler(
        enabled=enabled, init_scale=INIT_SCALE, growth_interval=GROWTH_INTERVAL
    )


def restore_scaler(scaler, checkpoint_data):
    state = checkpoint_data.get('scaler')
    if state is None:
        return 'LEGACY_CHECKPOINT_SCALER_INITIALIZED'
    if bool(state) != scaler.is_enabled():
        raise RuntimeError('CHECKPOINT_AMP_MODE_MISMATCH')
    scaler.load_state_dict(state)
    return 'CHECKPOINT_SCALER_RESTORED'


def training_update(model, optimizer, scaler, loss_closure, *, context=None,
                    max_retries=MAX_OVERFLOW_RETRIES, emit=None):
    """Run exactly one update, or fail without updating on invalid gradients.

    loss_closure must recompute a scalar loss on the SAME tensors each time.
    The current WaveShift model has no stochastic layers or mutable BN buffers.
    The caller owns the scheduler and advances it only after this returns.
    """
    if max_retries < 0:
        raise ValueError('max_retries must be nonnegative')
    context = dict(context or {})
    named_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if emit is None:
        emit = lambda record: print(json.dumps(record, allow_nan=False), flush=True)

    def event(kind, **fields):
        emit(dict(context, event=kind, **fields))

    for attempt in range(max_retries + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_closure()
        if loss.numel() != 1 or not torch.isfinite(loss).all().item():
            event('NON_FINITE_LOSS', attempt=attempt + 1, loss=str(loss.detach()))
            raise RuntimeError('NON_FINITE_LOSS: no optimizer update; recovery forbidden')
        loss_value = float(loss.detach())
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        del loss  # Release the old graph before any retry (critical at native resolution).
        scaler.unscale_(optimizer)

        grads = [(name, p) for name, p in named_parameters if p.grad is not None]
        if not grads:
            raise RuntimeError('NO_GRADIENTS')
        finite = torch.stack([torch.isfinite(p.grad).all() for _, p in grads])
        if not finite.all().item():
            bad_names = [n for (n, _), ok in zip(grads, finite.tolist()) if not ok]
            if not scaler.is_enabled():
                event('NON_FINITE_GRADIENT_WITHOUT_AMP', parameters=bad_names)
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError('NON_FINITE_GRADIENT_WITHOUT_AMP')
            # unscale_ has recorded nonfinite elements. GradScaler.step SKIPS
            # optimizer.step, and update backs off scale. Never clip these grads.
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)
            event('AMP_OVERFLOW', attempt=attempt + 1, retry_limit=max_retries,
                  loss=loss_value, scale_before=scale_before, scale_after=scale_after,
                  parameters=bad_names, optimizer_updated=False)
            if (attempt == max_retries or not math.isfinite(scale_after)
                    or scale_after <= 0 or scale_after >= scale_before):
                raise RuntimeError('AMP_OVERFLOW_RETRY_EXHAUSTED_OR_INVALID_SCALE')
            continue

        # An overflowing aggregate norm with finite elements is NOT the same
        # as GradScaler's found_inf. Do not call scaler.step in that case.
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for _, p in grads], max_norm=0.5, error_if_nonfinite=True
            )
        except RuntimeError:
            event('GRADIENT_NORM_FAILURE', loss=loss_value, scale=scale_before)
            optimizer.zero_grad(set_to_none=True)
            raise
        scaler.step(optimizer)
        scaler.update()
        if attempt:
            event('AMP_RECOVERED', retries=attempt, loss=loss_value,
                  grad_norm=float(grad_norm), scale=scaler.get_scale())
        return {'loss': loss_value, 'grad_norm': float(grad_norm),
                'overflow_retries': attempt, 'scale': scaler.get_scale()}
