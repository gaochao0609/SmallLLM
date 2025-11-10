import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
import matplotlib
import json

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

LOSS_RECORD_INTERVAL = 20
LOSS_PLOT_INTERVAL = 2000
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
        entry = {'step': step, 'loss': loss_value}
        with open(loss_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
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
    plt.plot(steps, losses, marker='o', markersize=3, linewidth=1)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (0 - {step})')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, step)
    plot_path = os.path.join(loss_plot_dir, f'loss_step_{step}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    last_plot_step = step
    Logger(f'Loss plot saved to {plot_path}')


def load_loss_history(max_step=None, reset=False):
    global loss_history, last_logged_step
    if reset:
        loss_history = []
        last_logged_step = -1
    if not loss_log_path or not os.path.exists(loss_log_path):
        return
    loaded = []
    with open(loss_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = data.get('step')
            loss = data.get('loss')
            if step is None or loss is None:
                continue
            if max_step is not None and step > max_step:
                continue
            loaded.append((step, loss))
    loaded.sort(key=lambda x: x[0])
    if loaded:
        loss_history = loaded
        last_logged_step = loaded[-1][0]


def train_epoch(epoch, loader, iters, start_step=0, global_step_offset=0, wandb=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    pending_micro_batches = 0
    deferred_save = None

    def perform_save(step_idx, abs_step_val):
        if not is_main_process():
            return
        model.eval()
        moe_suffix = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        torch.save(module.state_dict(), ckp)
        lm_checkpoint(
            lm_config,
            weight=args.save_weight,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            step=step_idx,
            global_step=abs_step_val,
            wandb=wandb,
            save_dir='../checkpoints'
        )
        model.train()

    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
        pending_micro_batches += 1
        abs_step = global_step_offset + (step - start_step)

        if is_main_process():
            loss_scalar = loss.detach().float().item() * args.accumulation_steps
            if abs_step % LOSS_RECORD_INTERVAL == 0 or step == iters:
                record_loss(abs_step, loss_scalar)
            if abs_step % LOSS_PLOT_INTERVAL == 0 or (step == iters and abs_step > last_plot_step):
                save_loss_plot(abs_step)

        if pending_micro_batches == args.accumulation_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            pending_micro_batches = 0

        if pending_micro_batches == 0 and deferred_save is not None:
            perform_save(*deferred_save)
            deferred_save = None

        should_save = (step % args.save_interval == 0 or step == iters)
        if should_save:
            if pending_micro_batches == 0:
                perform_save(step, abs_step)
            else:
                deferred_save = (step, abs_step)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            local_steps = step - start_step
            avg_step_time = spend_time / max(local_steps, 1)
            remaining_steps = max(iters - step, 0)
            eta_min = (avg_step_time * remaining_steps) / 60.0

            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min:.1f}min:')
            
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

    if pending_micro_batches > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        pending_micro_batches = 0
    if deferred_save is not None:
        perform_save(*deferred_save)
    return iters - start_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=8, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=96, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=1, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="gaochao0609-Pretrain", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    loss_plot_dir = os.path.join(args.save_dir, 'loss_plots')
    os.makedirs(loss_plot_dir, exist_ok=True)
    loss_log_path = os.path.join(loss_plot_dir, 'loss_log.jsonl')
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    steps_completed = 0
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    requested_dtype = args.dtype.lower()
    amp_dtype = None
    scaler_enabled = False
    if device_type == "cuda" and torch.cuda.is_available():
        supports_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
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
    elif device_type == "cuda":
        Logger("检测到 CUDA 不可用，将在 CPU 上以 FP32 运行。")
        device_type = "cpu"
    if amp_dtype is not None:
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        steps_completed = ckp_data.get('global_step', start_step)
        load_loss_history(max_step=steps_completed, reset=True)
    else:
        if os.path.exists(loss_log_path):
            os.remove(loss_log_path)
        load_loss_history(reset=True)
        steps_completed = 0

    if loss_history:
        last_plot_step = (loss_history[-1][0] // LOSS_PLOT_INTERVAL) * LOSS_PLOT_INTERVAL

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            total_iters = start_step + len(loader)
            processed_steps = train_epoch(epoch, loader, total_iters, start_step=start_step, global_step_offset=steps_completed, wandb=wandb)
            steps_completed += processed_steps
        else: # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            total_iters = len(loader)
            processed_steps = train_epoch(epoch, loader, total_iters, start_step=0, global_step_offset=steps_completed, wandb=wandb)
            steps_completed += processed_steps
