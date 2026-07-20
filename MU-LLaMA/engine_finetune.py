import math
import sys
import time
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from llama import LLaMA_adapter


def _grad_norm(parameters):
    grads = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    return torch.stack(grads).norm(2).item()

def train_one_epoch(model: LLaMA_adapter,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    # model.module.set_default_trainability()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    epoch_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if getattr(args, "max_train_batches", None) is not None and data_iter_step >= args.max_train_batches:
            break
        if len(batch) == 5:
            examples, labels, example_mask, imgs, extras = batch
        else:
            examples, labels, example_mask, imgs = batch
            extras = {}
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        examples = examples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)
        dissonance = extras.get("dissonance")
        dissonance_mask = extras.get("dissonance_mask")
        audio_lengths = extras.get("audio_lengths")
        if dissonance is not None:
            dissonance = dissonance.to(device, non_blocking=True)
            dissonance_mask = dissonance_mask.to(device, non_blocking=True)
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(device, non_blocking=True)
        autocast_dtype = torch.bfloat16 if getattr(args, "precision", "fp16") == "bf16" else torch.float16
        with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=autocast_dtype):
             c_loss, m_loss = model(
                 examples, labels, imgs, dissonance=dissonance,
                 dissonance_mask=dissonance_mask, audio_lengths=audio_lengths,
             )
        loss = c_loss  + m_loss * 0
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        m_loss_value = m_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        update_grad = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(loss, optimizer, clip_grad=getattr(args, "gradient_clip_norm", None),
                    parameters=(p for p in model.parameters() if p.requires_grad),
                    update_grad=update_grad)
        if update_grad:
            base_model = model.module if hasattr(model, "module") else model
            if getattr(base_model, "ds_encoder", None) is not None:
                metric_logger.update(ds_encoder_grad_norm=_grad_norm(base_model.ds_encoder.parameters()))
                if getattr(base_model, "ds_temporal", None) is not None:
                    metric_logger.update(ds_temporal_grad_norm=_grad_norm(base_model.ds_temporal.parameters()))
                metric_logger.update(ds_fusion_grad_norm=_grad_norm(base_model.ds_fusion.parameters()))
            llama_peft = [parameter for name, parameter in base_model.named_parameters()
                          if name.startswith("llama.") and parameter.requires_grad]
            metric_logger.update(llama_peft_grad_norm=_grad_norm(llama_peft))
            stage2_extra = [parameter for name, parameter in base_model.named_parameters()
                            if parameter.requires_grad and name.startswith(("prefix_query.", "mu_mert_norm_"))]
            metric_logger.update(stage2_extra_grad_norm=_grad_norm(stage2_extra))
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(mloss=m_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        base_model = model.module if hasattr(model, "module") else model
        for key, value in getattr(base_model, "last_dissonance_stats", {}).items():
            metric_logger.update(**{key: value})

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        m_loss_value_reduce = misc.all_reduce_mean(m_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('m_train_loss', m_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["epoch_time"] = time.time() - epoch_start
    stats["peak_gpu_memory"] = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    )
    return stats
