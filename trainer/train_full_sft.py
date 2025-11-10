import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
import matplotlib
import json

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

warnings.filterwarnings("ignore")

LOSS_RECORD_INTERVAL = 100
LOSS_PLOT_INTERVAL = 10000
loss_history = []
last_plot_step = 0
loss_plot_dir = None
loss_log_path = None
last_logged_step = -1


def record_loss(step, loss_value):
    global loss_history, last_logged_step
    if loss_history and step == loss_history[-1][0]:
        loss_history[-1] = (step, loss_value)
    else:
        loss_history.append((step, loss_value))
    if loss_log_path and step > last_logged_step:
        entry = {"step": step, "loss": loss_value}
        with open(loss_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        last_logged_step = step


def save_loss_plot(step):
    global last_plot_step
    if loss_plot_dir is None or not loss_history:
        return
    if step <= last_plot_step:
        return
    history = [item for item in loss_history if item[0] <= step]
    if not history:
        return
    steps, losses = zip(*history)
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, marker="o", markersize=3, linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (0 - {step})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0, step)
    plot_path = os.path.join(loss_plot_dir, f"loss_step_{step}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    last_plot_step = step
    Logger(f"Loss plot saved to {plot_path}")


def load_loss_history(max_step=None, reset=False):
    global loss_history, last_logged_step
    if reset:
        loss_history = []
        last_logged_step = -1
    if not loss_log_path or not os.path.exists(loss_log_path):
        return
    loaded = []
    with open(loss_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = data.get("step")
            loss = data.get("loss")
            if step is None or loss is None:
                continue
            if max_step is not None and step > max_step:
                continue
            loaded.append((step, loss))
    loaded.sort(key=lambda x: x[0])
    if loaded:
        loss_history = loaded
        last_logged_step = loaded[-1][0]


def train_epoch(epoch, loader, epoch_total_iters, start_step=0, global_step_offset=0, wandb=None):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    pending_micro_batches = 0
    processed_steps = 0
    deferred_save = None

    def perform_save(step_idx, abs_step_val):
        if not is_main_process():
            return
        model.eval()
        moe_suffix = "_moe" if lm_config.use_moe else ""
        ckp_path = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        module = model.module if isinstance(model, DistributedDataParallel) else model
        torch.save(module.state_dict(), ckp_path)
        lm_checkpoint(
            lm_config,
            weight=args.save_weight,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step_idx,
            wandb=wandb,
            save_dir="../checkpoints",
            scaler=scaler,
            global_step=abs_step_val,
        )
        model.train()

    for local_idx, (X, Y, loss_mask) in enumerate(loader):
        step = start_step + local_idx + 1
        abs_step = global_step_offset + processed_steps + 1

        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * epoch_total_iters + step, args.epochs * epoch_total_iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1),
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
        pending_micro_batches += 1

        if is_main_process():
            loss_scalar = loss.detach().float().item() * args.accumulation_steps
            if abs_step % LOSS_RECORD_INTERVAL == 0 or step == epoch_total_iters:
                record_loss(abs_step, loss_scalar)
            if abs_step % LOSS_PLOT_INTERVAL == 0 or (step == epoch_total_iters and abs_step > last_plot_step):
                save_loss_plot(abs_step)

        if pending_micro_batches == args.accumulation_steps or step == epoch_total_iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            pending_micro_batches = 0

        if pending_micro_batches == 0 and deferred_save is not None:
            perform_save(*deferred_save)
            deferred_save = None

        should_save = (step % args.save_interval == 0 or step == epoch_total_iters)
        if should_save:
            if pending_micro_batches == 0:
                perform_save(step, abs_step)
            else:
                deferred_save = (step, abs_step)

        if step % args.log_interval == 0 or step == epoch_total_iters:
            spend_time = time.time() - start_time
            processed_in_epoch = step - start_step
            avg_step_time = spend_time / max(processed_in_epoch, 1)
            remaining_steps = max(epoch_total_iters - step, 0)
            eta_min = (avg_step_time * remaining_steps) / 60.0
            current_loss = loss.detach().float().item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{epoch_total_iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min:.1f}min"
            )

            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        processed_steps += 1

    if pending_micro_batches > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        pending_micro_batches = 0

    if deferred_save is not None:
        perform_save(*deferred_save)

    return processed_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", default="full_sft", type=str, help="保存权重的前缀")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=16, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=512, type=int, help="训练的最大截断长度")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_512.jsonl", help="训练数据路径")
    parser.add_argument("--from_weight", default="pretrain", type=str, help="基于哪个权重训练，为none则不加载权重")
    parser.add_argument("--from_resume", default=1, type=int, choices=[0, 1], help="是否自动检查续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="gaochao0609-Full-SFT", help="wandb项目名称")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    loss_plot_dir = os.path.join(args.save_dir, "loss_plots")
    os.makedirs(loss_plot_dir, exist_ok=True)
    loss_log_path = os.path.join(loss_plot_dir, "loss_log.jsonl")
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    requested_dtype = args.dtype.lower()
    amp_dtype = None
    scaler_enabled = False
    if device_type == "cuda":
        supports_bf16 = (
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        if requested_dtype == "bfloat16":
            if supports_bf16:
                amp_dtype = torch.bfloat16
            else:
                Logger("当前设备不支持 bfloat16，将退回到 float16。")
                amp_dtype = torch.float16
                scaler_enabled = True
        elif requested_dtype == "float16":
            amp_dtype = torch.float16
            scaler_enabled = True
    if amp_dtype is not None:
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    steps_completed = 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        if "optimizer" in ckp_data:
            optimizer.load_state_dict(ckp_data["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(args.device)
        if "scaler" in ckp_data:
            scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        steps_completed = ckp_data.get("global_step", start_step)

    if ckp_data:
        load_loss_history(max_step=steps_completed, reset=True)
    else:
        if loss_log_path and os.path.exists(loss_log_path):
            os.remove(loss_log_path)
        load_loss_history(reset=True)
        steps_completed = 0

    if loss_history:
        last_plot_step = (loss_history[-1][0] // LOSS_PLOT_INTERVAL) * LOSS_PLOT_INTERVAL

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            base_sampler = train_sampler or range(len(train_ds))
            batch_sampler = SkipBatchSampler(base_sampler, args.batch_size, start_step)
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=(device_type == "cuda"),
            )
            remaining_iters = len(loader)
            if remaining_iters == 0:
                Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 已处理完全部 batch，跳过。")
                start_step = 0
                continue

            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前 {start_step} 个 step，从 step {start_step + 1} 开始继续训练。")
            epoch_total_iters = start_step + remaining_iters
            processed = train_epoch(
                epoch,
                loader,
                epoch_total_iters,
                start_step=start_step,
                global_step_offset=steps_completed,
                wandb=wandb,
            )
            steps_completed += processed
            start_step = 0
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=(device_type == "cuda"),
            )
            epoch_total_iters = len(loader)
            if epoch_total_iters == 0:
                Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 无可用 batch，跳过。")
                continue

            processed = train_epoch(
                epoch,
                loader,
                epoch_total_iters,
                start_step=0,
                global_step_offset=steps_completed,
                wandb=wandb,
            )
            steps_completed += processed
