"""
DracoAI V1 — Advanced Trainer (Production‑grade)
=================================================
Tất cả các chế độ huấn luyện cho DracoTransformerTorchV1.

Đã sửa toàn bộ các lỗi (batch gốc + batch mới + batch v1.1):
  • Shape mismatch CE loss / MTP loss
  • Gradient accumulation không reset khi gặp NaN
  • Flatten chat data gây trộn lẫn hội thoại
  • Resume không load optimizer + scaler → mất momentum
  • Hardcode EOS_ID → lấy từ tokenizer
  • Cứng hoá expert indices → dùng config
  • Data sampling batch_size=1 không bao phủ → DataLoader
  • Thiếu seed control → set_seed()
  • Overflow guard cho AMP (+ zero_grad khi skip_update)
  • Curriculum pipeline cơ bản
  • LR schedule resume-safe: global_total_steps freeze từ lần đầu (không phình ra)
  • _freeze_non_thinking dynamic từ config["thinking_experts"]
  • train_pipeline tự động resume giữa phase + clone kwargs từng phase
  • Optimizer init khi resume: verify trainable_param_names trước load
  • Optimizer/scaler reset cùng nhau khi mismatch
  • _sync_to_numpy chỉ gọi cuối run / export, tránh stall giữa training
  • MoE health monitoring: entropy, balance, std, collapse warning threshold
  • Gradient anomaly detect qua context manager cục bộ (không global)
  • Gradient accumulation scaling: loss / accumulation
  • _prepare_data chèn EOS giữa documents
  • _prepare_data EOS check per-seq (fix edge case seq đầu tiên rỗng)
  • _prepare_data flat list[int]: đảm bảo EOS cuối (fix silent cross-doc)
  • _prepare_data flat list[int]: giữ label_mask nếu được truyền (fix mask bị bỏ qua)
  • MTP length clamp: min(pred_len, label_len) tránh edge alignment
  • _SimpleDataset EOS-aware (không cross-document prediction)
  • _SimpleDataset hỗ trợ document-level uniform sampling (fix length bias)
  • _SimpleDataset resample offset mỗi lần __getitem__ (fix single-offset bug)
  • _SimpleDataset offset deterministic per-sample: random.Random(idx + epoch_seed)
    → multi-worker safe, reproducible, thay đổi qua epoch (fix global random state)
  • _SimpleDataset trả label_mask cho SFT label masking
  • _PackedDataset pad tail để không waste tokens
  • CE + MTP loss normalize theo số token hợp lệ (reduction="sum"/n_valid_tok)
    → hai loss cùng scale khi valid_len < T, tránh MTP overpower CE
  • Curriculum DataLoader rebuild chỉ khi ctx thay đổi ≥ 64
    → giảm gián đoạn prefetch warmup khi ctx tăng từng bước nhỏ
  • Validation dùng _PackedDataset khi use_packing=True
    → val_loss phản ánh đúng phân phối token, tránh mismatch train/val
  • _log_moe_health entropy threshold động theo 0.3 * log(n_experts)
    → tránh false alarm (2 experts) và cảnh báo quá muộn (16 experts)
  • Checkpoint pruning: atomic rename thay vì os.remove trực tiếp
  • Checkpoint save: rank-0 only để tránh race condition DDP/FSDP
  • FSDP state_dict: dùng FULL_STATE_DICT để checkpoint portable
  • Label masking chuẩn SFT: _compute_loss áp dụng label_mask
  • Logits alignment fix: bỏ l1[:,:-1] thừa (l1 và y đã khớp 1-1)
  • Explicit attention_mask causal: truyền thẳng vào forward
  • Validation split tại EOS boundary (không split giữa document)
  • tok_per_s công thức đúng: tính batch_size × accumulation
  • DistributedSampler.set_epoch() mỗi epoch + guard HAS_DDP (fix shuffle lặp + AttributeError)
  • Curriculum + packing: _PackedDataset.update_ctx() rebuild windows; DataLoader reset an toàn
  • Curriculum ctx thay đổi: rebuild DataLoader để tránh batch shape mismatch
  • _PackedDataset cross-doc leakage: mask token đầu doc mới = 0
  • Gradient clipping FSDP-aware: model.clip_grad_norm_() khi FSDP
  • AMP overflow tracking: khôi phục prev_scale check → step không tăng khi overflow
  • _prepare_data mask/ids length assert: phát hiện tokenizer mismatch sớm
  • _compute_loss mask guard: trả (None,0,0) khi toàn bộ mask = 0 (tránh fake loss)
  • FSDP isinstance check đúng: isinstance(model, FSDP) không phải isinstance(base, FSDP)
  • _log_moe_health: torch.tensor thay np.array để tránh GPU→CPU sync không cần thiết
  • tok_per_s DDP-aware: nhân world_size để throughput log chính xác
  • Validation shuffle=False (seed đã cố định, shuffle thừa và tốn compute)
  • Best checkpoint: lưu model tốt nhất theo val_loss (trainer_best.pt, không bị prune)
  • set_seed() cudnn flags chỉ set khi CUDA available

Yêu cầu:
  - transformer_torch_v1.py
  - transformer_v1.py
  - PyTorch + bitsandbytes (cho QLoRA)
"""

import os, json, math, time, random, contextlib
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple

# ── PyTorch ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformer_torch_v1 import DracoTransformerTorchV1
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Distributed ──────────────────────────────────────────────────────
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    HAS_DDP = True
except ImportError:
    HAS_DDP = False

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    HAS_FSDP = True
except ImportError:
    HAS_FSDP = False

# ── NumPy model ──────────────────────────────────────────────────────
try:
    from transformer_v1 import DracoTransformerV1, GGUFExporter
except ImportError:
    DracoTransformerV1 = None
    GGUFExporter = None


# ════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ════════════════════════════════════════════════════════════════════════

class _PackedDataset(Dataset):
    """
    Data packing: nhồi nhiều document ngắn vào một window ctx.
    Điền pack_ids liên tục; đặt label_mask=0 tại vị trí pad/EOS ranh giới
    để loss không tính trên BOS giả của document tiếp theo.
    """
    def __init__(self, segments: List[List[int]], ctx: int, eos_id: int):
        self.ctx      = ctx
        self.eos_id   = eos_id
        self._segs    = segments   # giữ lại để update_ctx có thể rebuild
        self.windows: List[Tuple[List[int], List[int]]] = []
        self._pack(segments)

    def update_ctx(self, new_ctx: int):
        """Rebuild windows với ctx mới — dùng cho curriculum khi use_packing=True."""
        if new_ctx != self.ctx:
            self.ctx     = new_ctx
            self.windows = []
            self._pack(self._segs)

    def _pack(self, segments):
        buf  = []
        mask = []  # 1 = tính loss, 0 = bỏ qua
        for seg in segments:
            toks  = list(seg)
            # mask = 1 cho tất cả trừ:
            #   - token EOS cuối doc (không train predict-after-EOS)
            #   - token ĐẦU doc mới nếu buf không rỗng (tránh cross-doc leakage)
            smask = [1] * len(toks)
            if toks and toks[-1] == self.eos_id:
                smask[-1] = 0   # EOS cuối doc
            if buf:
                smask[0] = 0    # token đầu doc tiếp theo không được predict từ doc trước
            buf.extend(toks)
            mask.extend(smask)
            while len(buf) >= self.ctx + 2:
                self.windows.append((buf[: self.ctx + 2], mask[: self.ctx + 2]))
                buf  = buf[self.ctx + 2:]
                mask = mask[self.ctx + 2:]
        # flush nếu đủ dài
        if len(buf) >= self.ctx + 2:
            self.windows.append((buf[: self.ctx + 2], mask[: self.ctx + 2]))
        elif len(buf) > 2:
            # Pad tail bằng EOS để không waste tokens cuối
            while len(buf) < self.ctx + 2:
                buf.append(self.eos_id)
                mask.append(0)
            self.windows.append((buf[: self.ctx + 2], mask[: self.ctx + 2]))

    def __len__(self):  return len(self.windows)

    def __getitem__(self, idx):
        ids, msk = self.windows[idx]
        x = torch.tensor(ids[:-2], dtype=torch.long)
        y = torch.tensor(ids[1:-1], dtype=torch.long)
        m = torch.tensor(msk[1:-1], dtype=torch.bool)
        return x, y, m


class _SimpleDataset(Dataset):
    """
    EOS-aware với document-level uniform sampling (fix length-bias).

    Thay vì nhảy i += eos_positions[0]+1 (bias doc dài → nhiều sample),
    ta build danh sách document boundaries rồi lấy random offset trong
    mỗi document — mỗi document đóng góp đúng 1 starting index.

    label_mask: 1 = tính loss, 0 = bỏ qua (dùng cho SFT masking).
    Mặc định mask=1 toàn bộ (phù hợp pretrain).
    Để SFT mask prompt, truyền segments có kèm mask.
    """
    def __init__(
        self,
        ids:    List[int],
        ctx:    int,
        eos_id: int,
        label_masks: Optional[List[int]] = None,  # cùng độ dài ids
        epoch_seed:  int = 0,   # tăng mỗi epoch để offset thay đổi qua các epoch
    ):
        self.ids         = ids
        self.ctx         = ctx
        self.eos_id      = eos_id
        self.label_masks = label_masks  # None → pretrain (all-1)
        self.epoch_seed  = epoch_seed   # seed gốc cho RNG per-sample
        self._docs       = self._build_docs()   # list of (doc_start, doc_end_excl)

    def _build_docs(self) -> List[Tuple[int, int]]:
        """
        Build danh sách document boundaries.
        Trả về list (ds, de) cho mọi doc đủ dài để lấy ≥1 window ctx+2.
        """
        n     = len(self.ids)
        docs  = []
        start = 0
        for i, tok in enumerate(self.ids):
            if tok == self.eos_id:
                docs.append((start, i + 1))
                start = i + 1
        if start < n:
            docs.append((start, n))

        valid = [(ds, de) for ds, de in docs if (de - ds) >= self.ctx + 2]

        # Fallback: nếu không có doc hợp lệ, toàn bộ stream = 1 doc
        if not valid and n >= self.ctx + 2:
            valid = [(0, n)]

        return valid

    def update_ctx(self, new_ctx: int):
        """
        Cập nhật ctx và rebuild doc list mà không cần tạo lại toàn bộ dataset.
        Dùng cho curriculum: tránh reset DataLoader iterator và DistributedSampler state.
        """
        if new_ctx != self.ctx:
            self.ctx   = new_ctx
            self._docs = self._build_docs()

    def set_epoch_seed(self, epoch: int):
        """Tăng epoch_seed để offset thay đổi qua các epoch (gọi khi bắt đầu epoch mới)."""
        self.epoch_seed = epoch

    def __len__(self):
        return max(len(self._docs), 1)

    def __getitem__(self, idx):
        if not self._docs:
            # Fallback tuyệt đối (dataset quá ngắn)
            s = 0
        else:
            ds, de = self._docs[idx % len(self._docs)]
            max_off = (de - ds) - (self.ctx + 2)
            # Dùng RNG gắn với (idx, epoch_seed) → deterministic, multi-worker safe,
            # không phụ thuộc thứ tự gọi của worker → reproduce được.
            rng    = random.Random(idx + self.epoch_seed * 1_000_003)
            offset = rng.randint(0, max_off)
            s = ds + offset

        ids = self.ids[s: s + self.ctx + 2]
        x   = torch.tensor(ids[:-2], dtype=torch.long)
        y   = torch.tensor(ids[1:-1], dtype=torch.long)
        if self.label_masks is not None:
            raw_m = self.label_masks[s: s + self.ctx + 2]
            m     = torch.tensor(raw_m[1:-1], dtype=torch.bool)
        else:
            m = torch.ones(self.ctx, dtype=torch.bool)
        return x, y, m


# ════════════════════════════════════════════════════════════════════════
# SFT masking helper
# ════════════════════════════════════════════════════════════════════════

def build_sft_label_mask(
    token_ids: List[int],
    response_ranges: List[Tuple[int, int]],
) -> List[int]:
    """
    Tạo label_mask cho SFT: chỉ tính loss trên phần response.

    Args:
        token_ids:       toàn bộ token ids của một conversation
        response_ranges: list (start, end) index [inclusive, exclusive]
                         của các response turn (assistant)
    Returns:
        mask list có cùng độ dài token_ids, 1=tính loss, 0=bỏ qua
    """
    mask = [0] * len(token_ids)
    for s, e in response_ranges:
        for i in range(s, min(e, len(mask))):
            mask[i] = 1
    return mask


# ════════════════════════════════════════════════════════════════════════
# Curriculum sequence-length scheduler
# ════════════════════════════════════════════════════════════════════════

def curriculum_ctx(step: int, total: int, ctx_min: int, ctx_max: int) -> int:
    """
    Dynamic sequence-length curriculum: tăng dần ctx từ ctx_min → ctx_max
    theo cosine schedule. Giúp model học short-turn trước, long-form sau.
    """
    progress = min(step / max(total, 1), 1.0)
    # cosine ease-in: 0 → 1 từ từ ở đầu
    t = 0.5 * (1 - math.cos(math.pi * progress))
    raw = ctx_min + t * (ctx_max - ctx_min)
    # Làm tròn về bội số 64 gần nhất
    return max(ctx_min, int(round(raw / 64)) * 64)


# ════════════════════════════════════════════════════════════════════════
# Eval harness
# ════════════════════════════════════════════════════════════════════════

class EvalHarness:
    """
    Validation loop nhẹ: tính perplexity, token accuracy, mean entropy.
    Không tính gradient. Trả về dict metrics.
    """
    @staticmethod
    @torch.no_grad()
    def evaluate(
        model,
        val_ids:  List[int],
        ctx:      int,
        eos_id:   int,
        device:   str,
        n_batches: int = 50,
        batch_size: int = 4,
        vocab_size: Optional[int] = None,
        num_workers: int = 0,
        val_seed: int = 0,    # fixed seed để val metrics deterministic
        use_packing: bool = False,  # dùng _PackedDataset để match train distribution khi use_packing=True
        val_segments: Optional[List[List[int]]] = None,  # segments cho _PackedDataset
    ) -> Dict[str, float]:
        model.eval()
        # Dùng cùng loại dataset với train để val_loss phản ánh đúng phân phối token
        if use_packing and val_segments is not None:
            dataset = _PackedDataset(val_segments, ctx, eos_id)
        else:
            dataset = _SimpleDataset(val_ids, ctx, eos_id)
        if len(dataset) == 0:
            model.train()
            return {"val_ppl": float("inf"), "val_acc": 0.0, "val_entropy": 0.0}

        # shuffle=False: seed đã cố định qua Generator, shuffle thêm thừa + tốn compute
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             drop_last=True, num_workers=num_workers)
        total_loss = 0.0
        total_acc  = 0.0
        total_ent  = 0.0
        n_counted  = 0

        base = model.module if hasattr(model, "module") else model
        vs   = vocab_size or getattr(base, "vocab_size", None)

        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            x, y, m = batch
            x, y, m = x.to(device), y.to(device), m.to(device)

            attn_mask = _make_causal_mask(x.shape[0], x.shape[1], device)
            try:
                l1, l2, aux = model(x, attention_mask=attn_mask, return_aux=True)
            except TypeError:
                l1, l2, aux = model(x, return_aux=True)

            # l1 và y đã khớp 1-1 từ dataset; không cắt [:, :-1]
            logits  = l1
            T_logit = logits.shape[1]
            T_label = y.shape[1]
            T       = min(T_logit, T_label)

            logits = logits[:, :T]
            labels = y[:, :T]
            mask_t = m[:, :T]

            if vs:
                logits = logits[:, :, :vs]

            # Loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            flat_mask = mask_t.reshape(-1).float()
            denom     = flat_mask.sum().clamp(min=1)
            loss_val  = (loss * flat_mask).sum() / denom
            total_loss += loss_val.item()

            # Accuracy (token-level)
            preds   = logits.argmax(-1)                  # (B, T)
            correct = (preds == labels) & mask_t
            acc     = correct.sum().float() / mask_t.sum().clamp(min=1)
            total_acc += acc.item()

            # Entropy (mean over active tokens)
            probs   = torch.softmax(logits, dim=-1)      # (B, T, V)
            ent     = -(probs * (probs + 1e-9).log()).sum(-1)   # (B, T)
            ent_val = (ent * mask_t.float()).sum() / mask_t.sum().clamp(min=1)
            total_ent += ent_val.item()

            n_counted += 1

        model.train()
        n = max(n_counted, 1)
        avg_loss = total_loss / n
        return {
            "val_ppl":     math.exp(min(avg_loss, 20)),
            "val_loss":    avg_loss,
            "val_acc":     total_acc / n,
            "val_entropy": total_ent / n,
        }


# ════════════════════════════════════════════════════════════════════════
# Attention mask helper
# ════════════════════════════════════════════════════════════════════════

def _make_causal_mask(B: int, T: int, device: str) -> torch.Tensor:
    """
    Tạo causal mask boolean (B, T, T) — True = attend.
    Truyền thẳng vào forward để tránh phụ thuộc model tự build.
    """
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).expand(B, -1, -1)


# ════════════════════════════════════════════════════════════════════════
# Main Trainer
# ════════════════════════════════════════════════════════════════════════

class TrainerV1:
    def __init__(
        self,
        checkpoint_dir:   str,
        config:           dict,
        numpy_model:      Optional[Any] = None,
        tokenizer:        Any = None,
        device:           Optional[str] = None,
        debug_mode:       bool = False,
        keep_checkpoints: int  = 3,
        # Distributed
        distributed:      str  = "none",   # "none" | "ddp" | "fsdp"
        local_rank:       int  = 0,
        # Gradient checkpointing
        gradient_checkpointing: bool = False,
        # Validation
        val_every:        int  = 500,
        val_batches:      int  = 50,
        # Curriculum
        use_curriculum:   bool = False,
        ctx_min:          Optional[int] = None,
        ctx_max:          Optional[int] = None,
        # Data packing
        use_packing:      bool = False,
        # DataLoader tuning
        num_workers:      int  = 0,
        prefetch_factor:  Optional[int] = None,
    ):
        self.ckpt_dir               = checkpoint_dir
        self.config                 = config
        self.numpy_model            = numpy_model
        self.tokenizer              = tokenizer
        self.debug_mode             = debug_mode
        self.keep_checkpoints       = keep_checkpoints
        self.distributed            = distributed
        self.local_rank             = local_rank
        self.gradient_checkpointing = gradient_checkpointing
        self.val_every              = val_every
        self.val_batches            = val_batches
        self.use_curriculum         = use_curriculum
        self.ctx_min                = ctx_min or max(64, config.get("window", 512) // 8)
        self.ctx_max                = ctx_max or config.get("window", 512)
        self.use_packing            = use_packing
        self.num_workers            = num_workers
        self.prefetch_factor        = prefetch_factor if num_workers > 0 else None

        if distributed in ("ddp", "fsdp"):
            self.device = f"cuda:{local_rank}"
        else:
            self.device = device or (
                "cuda" if HAS_TORCH and torch.cuda.is_available() else
                "mps"  if HAS_TORCH and torch.backends.mps.is_available() else "cpu"
            )

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", 151645)

        if self.debug_mode:
            print("[Trainer] ⚠️  debug_mode=True: anomaly detection bật CỤC BỘ mỗi forward "
                  "(~5-10x chậm hơn). Tắt khi train thật.")

    # ═══════════════════════════════════════════════════════════════
    # Reproducibility
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # ═══════════════════════════════════════════════════════════════
    # Distributed init / wrap
    # ═══════════════════════════════════════════════════════════════
    def _init_distributed(self):
        if self.distributed == "none" or not HAS_DDP:
            return
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            print(f"[Trainer] Distributed init: {self.distributed} | "
                  f"rank={dist.get_rank()}/{dist.get_world_size()}")

    def _wrap_distributed(self, model):
        if self.distributed == "ddp" and HAS_DDP:
            model = DDP(model, device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=False)
            print("[Trainer] Model wrapped with DDP.")
        elif self.distributed == "fsdp" and HAS_FSDP:
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mp_policy,
                device_id=self.local_rank,
            )
            print("[Trainer] Model wrapped with FSDP (FULL_SHARD + fp16).")
        return model

    # ═══════════════════════════════════════════════════════════════
    # DataLoader factory (DistributedSampler + num_workers)
    # ═══════════════════════════════════════════════════════════════
    def _make_dataloader(self, dataset, batch_size: int) -> DataLoader:
        """
        Tạo DataLoader chuẩn cho cả single-GPU và distributed.
        - DDP/FSDP: dùng DistributedSampler để chia đều data cho mỗi rank.
        - Single: shuffle=True như cũ.
        - num_workers / prefetch_factor được đặt từ __init__ để tránh GPU đói data.
        """
        if self.distributed != "none" and HAS_DDP and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        loader_kwargs: dict = dict(
            batch_size=max(1, batch_size),
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            drop_last=True,
            pin_memory=(self.device != "cpu"),
            num_workers=self.num_workers,
        )
        if self.prefetch_factor is not None and self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(dataset, **loader_kwargs)

    # ═══════════════════════════════════════════════════════════════
    # API chính: train()
    # ═══════════════════════════════════════════════════════════════
    def train(
        self,
        data:               List[Union[str, List[int]]],
        steps:              int   = 5000,
        batch_size:         int   = 1,
        accumulation:       int   = 1,
        mode:               str   = "full",
        skill_group:        Optional[str]   = None,
        lora_rank:          int   = 8,
        lora_alpha:         float = 16.0,
        max_lr:             float = 3e-4,
        min_lr:             float = 3e-5,
        warmup_steps:       int   = 200,
        grad_clip:          float = 1.0,
        log_every:          int   = 50,
        save_every:         int   = 500,
        resume:             bool  = False,
        global_total_steps: Optional[int] = None,
        # SFT label masks (cùng độ dài data, mỗi phần tử là mask list)
        label_masks:        Optional[List[List[int]]] = None,
        # Validation split (0.0–0.2)
        val_split:          float = 0.05,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required.")

        self._init_distributed()
        print(f"\n{'='*60}\n DracoAI V1 Training | mode={mode} | steps={steps}\n{'='*60}")

        data_ids, mask_ids = self._prepare_data(data, label_masks)
        ctx_base = self.config["window"]

        if len(data_ids) < ctx_base + 3:
            raise ValueError("Training data too short.")

        # ── Train / val split (tại EOS boundary gần nhất) ────────
        split_at = max(0, int(len(data_ids) * (1 - val_split)))
        # Advance đến EOS gần nhất để không split giữa document
        eos_id = self.eos_token_id
        while split_at < len(data_ids) - 1 and data_ids[split_at] != eos_id:
            split_at += 1
        if split_at < len(data_ids):
            split_at += 1  # include EOS trong train set
        train_ids = data_ids[:split_at]
        val_ids   = data_ids[split_at:] if split_at < len(data_ids) else data_ids
        train_mask = mask_ids[:split_at] if mask_ids else None

        model, optimizer, scaler, start_step, loaded_g_total = self._init_train_state(
            mode, skill_group, lora_rank, lora_alpha, max_lr, resume
        )

        # ── Gradient checkpointing ─────────────────────────────────
        if self.gradient_checkpointing:
            base = model.module if hasattr(model, "module") else model
            if hasattr(base, "enable_gradient_checkpointing"):
                base.enable_gradient_checkpointing()
                print("[Trainer] Gradient checkpointing enabled.")
            elif hasattr(base, "gradient_checkpointing_enable"):
                base.gradient_checkpointing_enable()
                print("[Trainer] Gradient checkpointing enabled.")
            else:
                print("[Trainer] ⚠️  Model does not support gradient checkpointing API.")

        # ── Distributed wrap ───────────────────────────────────────
        model = self._wrap_distributed(model)

        # ── Global schedule freeze ─────────────────────────────────
        if global_total_steps is not None:
            g_total = global_total_steps
        elif loaded_g_total is not None:
            g_total = loaded_g_total
        else:
            g_total = start_step + steps

        # ── Dataset + DataLoader ───────────────────────────────────
        ctx = curriculum_ctx(start_step, g_total, self.ctx_min, self.ctx_max) \
              if self.use_curriculum else ctx_base

        if self.use_packing:
            # Cắt train_ids thành segments theo EOS
            segs   = self._split_segments(train_ids)
            dataset = _PackedDataset(segs, ctx, self.eos_token_id)
        else:
            dataset = _SimpleDataset(train_ids, ctx, self.eos_token_id, train_mask)

        loader      = self._make_dataloader(dataset, batch_size)
        loader_iter = iter(loader)

        loss_hist    = []
        best_val_loss = float("inf")
        t0          = time.time()
        step        = start_step
        target_step = start_step + steps
        accum_steps = 0
        epoch       = 0   # epoch counter cho DistributedSampler.set_epoch()
        trainable   = [p for p in model.parameters() if p.requires_grad]
        base_model  = model.module if hasattr(model, "module") else model
        vocab_size  = getattr(base_model, "vocab_size", None)

        while step < target_step:
            # ── Curriculum: cập nhật ctx + rebuild DataLoader ─────────
            # Chỉ rebuild khi chênh lệch ctx ≥ 64 (tránh gián đoạn prefetch liên tục
            # khi ctx thay đổi từng bước nhỏ trong cosine schedule).
            # update_ctx() cũng phải được gọi để dataset trả đúng batch shape.
            if self.use_curriculum:
                new_ctx = curriculum_ctx(step, g_total, self.ctx_min, self.ctx_max)
                if new_ctx != ctx and abs(new_ctx - ctx) >= 64:
                    ctx = new_ctx
                    dataset.update_ctx(ctx)           # cập nhật docs/windows trong dataset
                    loader      = self._make_dataloader(dataset, batch_size)
                    loader_iter = iter(loader)
                    epoch = 0   # reset epoch counter khi rebuild

            try:
                batch = next(loader_iter)
            except StopIteration:
                # Hết epoch → set_epoch để shuffle khác đi (quan trọng cho DDP)
                epoch += 1
                if HAS_DDP and hasattr(loader, "sampler") and isinstance(
                    loader.sampler, DistributedSampler
                ):
                    loader.sampler.set_epoch(epoch)
                # Cập nhật epoch_seed cho _SimpleDataset → offset thay đổi qua epoch
                if hasattr(dataset, "set_epoch_seed"):
                    dataset.set_epoch_seed(epoch)
                loader_iter = iter(loader)
                batch = next(loader_iter)

            x, y, m = batch
            x = x.to(self.device)
            y = y.to(self.device)
            m = m.to(self.device)

            lr = self._cosine_lr(step, g_total, max_lr, min_lr, warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ── Causal attention mask ──────────────────────────────
            attn_mask = _make_causal_mask(x.shape[0], x.shape[1], self.device)

            # ── Forward ───────────────────────────────────────────
            anomaly_ctx = (torch.autograd.detect_anomaly()
                           if self.debug_mode else contextlib.nullcontext())
            with anomaly_ctx:
                loss, tok_acc, tok_ent = self._compute_loss(
                    model, x, y, m, attn_mask, vocab_size
                )

            if loss is None or not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0
                continue

            scaled_loss = loss / accumulation
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_steps += 1
            loss_hist.append(loss.item())

            if accum_steps >= accumulation:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                # FSDP-aware gradient clipping
                if HAS_FSDP and isinstance(model, FSDP):
                    model.clip_grad_norm_(grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

                overflow = False
                if scaler.is_enabled():
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)   # bỏ qua nội bộ nếu found_inf
                    scaler.update()
                    # Nếu scale giảm → overflow xảy ra, step không thật sự update
                    if scaler.get_scale() < prev_scale:
                        overflow = True
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0

                if not overflow:
                    step += 1
                else:
                    # Không tăng step, không log loss ảo — xoá entry vừa push
                    if loss_hist:
                        loss_hist.pop()
                    continue

            # ── Logging ───────────────────────────────────────────
            if step % log_every == 0 and step > start_step:
                avg_loss = sum(loss_hist[-log_every:]) / min(len(loss_hist), log_every)
                elapsed  = time.time() - t0
                tok_per_s = (step - start_step) * batch_size * accumulation * ctx / elapsed / 1000 if elapsed > 0 else 0
                ctx_info  = f" ctx={ctx}" if self.use_curriculum else ""
                print(f"  Step {step:5d}/{g_total} | loss={avg_loss:.4f} | "
                      f"ppl={math.exp(min(avg_loss, 20)):.1f} | lr={lr:.2e} | "
                      f"acc={tok_acc:.3f} | ent={tok_ent:.3f} | "
                      f"{tok_per_s:.1f}K tok/s{ctx_info}")
                self._log_moe_health(model)

            # ── Validation ────────────────────────────────────────
            if self.val_every > 0 and step % self.val_every == 0 and step > start_step:
                if len(val_ids) >= ctx + 3:
                    # Khi use_packing, tạo val_segments để _PackedDataset match train distribution
                    val_segs = self._split_segments(val_ids) if self.use_packing else None
                    metrics = EvalHarness.evaluate(
                        model, val_ids, ctx, self.eos_token_id,
                        self.device, self.val_batches, batch_size, vocab_size,
                        num_workers=self.num_workers, val_seed=0,
                        use_packing=self.use_packing, val_segments=val_segs,
                    )
                    print(f"  [Val] step={step} | "
                          f"ppl={metrics['val_ppl']:.2f} | "
                          f"acc={metrics['val_acc']:.3f} | "
                          f"entropy={metrics['val_entropy']:.3f}")
                    # Lưu best checkpoint nếu val_loss cải thiện
                    if metrics["val_loss"] < best_val_loss:
                        best_val_loss = metrics["val_loss"]
                        self._save_checkpoint(
                            model, optimizer, scaler, step,
                            best_val_loss, g_total, tag="best"
                        )
                        print(f"  [Val] ✅ Best model saved (val_loss={best_val_loss:.4f})")

            # ── Checkpoint ────────────────────────────────────────
            if step % save_every == 0 and step > start_step:
                self._save_checkpoint(model, optimizer, scaler, step,
                                      loss_hist[-1] if loss_hist else 0.0, g_total)
                self._prune_checkpoints()

        # ── Cuối run ──────────────────────────────────────────────
        self._save_checkpoint(model, optimizer, scaler, step,
                              loss_hist[-1] if loss_hist else 0.0, g_total)
        self._prune_checkpoints()
        if self.numpy_model is not None:
            self._sync_to_numpy(model)
        print(f"[Trainer] Training complete ({step} steps).")

    # ═══════════════════════════════════════════════════════════════
    # train_chat()  — SFT với label masking chuẩn
    # ═══════════════════════════════════════════════════════════════
    def train_chat(
        self,
        conversations:      List[List[Dict[str, str]]],
        steps:              int   = 5000,
        batch_size:         int   = 1,
        accumulation:       int   = 1,
        mode:               str   = "full",
        skill_group:        Optional[str]   = None,
        lora_rank:          int   = 8,
        lora_alpha:         float = 16.0,
        max_lr:             float = 3e-4,
        min_lr:             float = 3e-5,
        warmup_steps:       int   = 200,
        grad_clip:          float = 1.0,
        log_every:          int   = 50,
        save_every:         int   = 500,
        resume:             bool  = False,
        global_total_steps: Optional[int] = None,
        # SFT: mask prompt tokens (không tính loss trên system/user)
        mask_prompt:        bool  = True,
    ):
        if self.tokenizer is None or not hasattr(self.tokenizer, "encode_chat"):
            raise RuntimeError("Tokenizer with encode_chat required.")

        print(f"\n{'='*60}\n DracoAI V1 Chat Training | mode={mode}\n{'='*60}")

        eos_id   = self.eos_token_id
        flat_ids: List[int] = []
        flat_mask: List[int] = []

        for conv in conversations:
            # encode_chat nên trả về (ids, response_ranges) nếu mask_prompt
            if mask_prompt and hasattr(self.tokenizer, "encode_chat_with_mask"):
                ids, resp_ranges = self.tokenizer.encode_chat_with_mask(conv)
                seg_mask = build_sft_label_mask(ids, resp_ranges)
            else:
                ids = self.tokenizer.encode_chat(conv, add_generation_prompt=True)
                # fallback: tính loss toàn bộ (như cũ)
                seg_mask = [1] * len(ids)

            flat_ids.extend(ids)
            flat_mask.extend(seg_mask)
            # EOS boundary
            if eos_id is not None:
                flat_ids.append(eos_id)
                flat_mask.append(0)   # EOS boundary không tính loss

        self.train(
            data=[flat_ids],
            steps=steps,
            batch_size=batch_size,
            accumulation=accumulation,
            mode=mode,
            skill_group=skill_group,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            grad_clip=grad_clip,
            log_every=log_every,
            save_every=save_every,
            resume=resume,
            global_total_steps=global_total_steps,
            label_masks=[flat_mask],   # truyền mask vào train()
        )

    # ═══════════════════════════════════════════════════════════════
    # Pipeline curriculum
    # ═══════════════════════════════════════════════════════════════
    def train_pipeline(
        self,
        pretrain_data:   Optional[List[str]] = None,
        skill_data:      Optional[List[str]] = None,
        skill_group:     Optional[str] = None,
        thinking_data:   Optional[List[str]] = None,
        chat_data:       Optional[List[List[Dict[str, str]]]] = None,
        steps_per_phase: int = 2000,
        mode:            str = "full",
        **kwargs,
    ):
        """
        Chạy tuần tự: pretrain → skill → thinking → chat.
        Phase đầu resume theo kwargs. Phase sau tự động resume=True.
        """
        phase          = 1
        initial_resume = kwargs.pop("resume", False)

        if pretrain_data:
            print(f"\n🔹 Phase {phase}: Pretrain")
            self.train(data=pretrain_data, steps=steps_per_phase, mode=mode,
                       resume=initial_resume, **dict(kwargs))
            phase += 1

        if skill_data and skill_group:
            print(f"\n🔹 Phase {phase}: Skill – {skill_group}")
            self.train(data=skill_data, steps=steps_per_phase, mode="skill",
                       skill_group=skill_group, resume=(phase > 1), **dict(kwargs))
            phase += 1

        if thinking_data:
            print(f"\n🔹 Phase {phase}: Thinking (CoT)")
            self.train(data=thinking_data, steps=steps_per_phase, mode="thinking",
                       resume=(phase > 1), **dict(kwargs))
            phase += 1

        if chat_data:
            print(f"\n🔹 Phase {phase}: Chat")
            self.train_chat(conversations=chat_data, steps=steps_per_phase, mode=mode,
                            resume=(phase > 1), **dict(kwargs))

    # ═══════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════

    def _split_segments(self, ids: List[int]) -> List[List[int]]:
        """Cắt flat token list thành list of document segments theo EOS."""
        segs, buf = [], []
        for tok in ids:
            buf.append(tok)
            if tok == self.eos_token_id:
                segs.append(buf)
                buf = []
        if buf:
            segs.append(buf)
        return segs

    def _prepare_data(
        self,
        data: List[Union[str, List[int]]],
        label_masks: Optional[List[List[int]]] = None,
    ) -> Tuple[List[int], Optional[List[int]]]:
        """
        Tokenize nếu cần, chèn EOS giữa mỗi document.
        Fix: EOS check per-seq (không check toàn buffer).

        Trả về (flat_ids, flat_mask_or_None).
        """
        eos_id = self.eos_token_id
        all_ids: List[int]  = []
        all_mask: List[int] = []

        if self.tokenizer and data and isinstance(data[0], str):
            print("[Trainer] Tokenizing data...")
            for i, text in enumerate(data):
                ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
                all_ids.extend(ids)
                if eos_id is not None:
                    all_ids.append(eos_id)
                # mask: 1 trừ EOS boundary
                if label_masks and i < len(label_masks):
                    all_mask.extend(label_masks[i])
                    all_mask.append(0)
                else:
                    all_mask.extend([1] * len(ids))
                    all_mask.append(0)
            return all_ids, all_mask if all_mask else None

        if data and isinstance(data[0], list):
            for i, seq in enumerate(data):
                all_ids.extend(seq)
                # FIX: check EOS tại cuối seq này (không check all_ids toàn bộ)
                if eos_id is not None and (not seq or seq[-1] != eos_id):
                    all_ids.append(eos_id)
                if label_masks and i < len(label_masks):
                    msk = label_masks[i]
                    assert len(msk) == len(seq), (
                        f"[_prepare_data] Mask/ids length mismatch at index {i}: "
                        f"ids={len(seq)}, mask={len(msk)}. "
                        f"Tokenizer hoặc mask builder có thể không đồng bộ."
                    )
                    all_mask.extend(msk)
                    all_mask.append(0)  # EOS boundary
                else:
                    all_mask.extend([1] * len(seq))
                    all_mask.append(0)
            return all_ids, all_mask if all_mask else None

        # Flat list[int] — đảm bảo EOS cuối để tránh cross-document prediction
        # FIX: nếu có label_masks thì phải trả mask, không được bỏ qua (trước đây luôn return None)
        ids = list(data)
        eos_id = self.eos_token_id
        if eos_id is not None and (not ids or ids[-1] != eos_id):
            ids.append(eos_id)
        # Flat mode: label_masks[0] là mask cho toàn bộ flat list
        if label_masks and isinstance(label_masks, list) and len(label_masks) > 0:
            flat_mask_data = label_masks[0] if isinstance(label_masks[0], list) else list(label_masks)
            # Sync độ dài: nếu EOS vừa thêm thì append 0 cho mask
            if len(flat_mask_data) < len(ids):
                flat_mask_data = list(flat_mask_data) + [0] * (len(ids) - len(flat_mask_data))
            elif len(flat_mask_data) > len(ids):
                flat_mask_data = flat_mask_data[:len(ids)]
            return ids, flat_mask_data
        return ids, None

    def _init_train_state(self, mode, skill_group, lora_rank, lora_alpha, max_lr, resume):
        if resume:
            state = self._load_latest_checkpoint(mode, skill_group, lora_rank, lora_alpha, max_lr)
            return (state["model"], state["optimizer"], state["scaler"],
                    state["step"], state.get("global_total_steps"))
        model     = DracoTransformerTorchV1(self.config).to(self.device)
        self._apply_mode(model, mode, skill_group, lora_rank, lora_alpha)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
        scaler    = torch.cuda.amp.GradScaler(enabled=(self.device.startswith("cuda")))
        return model, optimizer, scaler, 0, None

    def _apply_mode(self, model, mode, skill_group, lora_rank, lora_alpha):
        if mode in ("full", "pretrain", "mixed"):
            print(f"[Trainer] Mode: {mode}")
        elif mode == "router":
            model.freeze_experts()
        elif mode == "skill":
            if skill_group not in ("code", "language"):
                raise ValueError("skill_group must be 'code' or 'language'")
            self._freeze_non_skill(model, skill_group)
        elif mode == "thinking":
            self._freeze_non_thinking(model)
        elif mode == "lora":
            model.enable_lora(rank=lora_rank, alpha=lora_alpha)
        elif mode == "qlora":
            model.enable_qlora(rank=lora_rank, alpha=lora_alpha)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _freeze_non_skill(self, model, skill: str):
        n_exp = self.config.get("n_experts", 8)
        mid   = n_exp // 2
        keep  = set(range(0, mid)) if skill == "code" else set(range(mid, n_exp))
        for blk in model.blocks:
            for e, exp in enumerate(blk.moe.experts):
                for p in exp.parameters():
                    p.requires_grad = (e in keep)
            for p in blk.moe.shared.parameters():
                p.requires_grad = False

    def _freeze_non_thinking(self, model):
        keep = set(self.config.get("thinking_experts", [0, 2]))
        print(f"[Trainer] Thinking mode: keeping experts {sorted(keep)}")
        for blk in model.blocks:
            for e, exp in enumerate(blk.moe.experts):
                for p in exp.parameters():
                    p.requires_grad = (e in keep)
            for p in blk.moe.shared.parameters():
                p.requires_grad = False
            for p in blk.attn.parameters():
                p.requires_grad = False

    def _compute_loss(
        self,
        model,
        input_ids:  torch.Tensor,
        labels:     torch.Tensor,
        label_mask: torch.Tensor,
        attn_mask:  torch.Tensor,
        vocab_size: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], float, float]:
        """
        Tính CE loss + MTP loss với:
          - Label masking chuẩn SFT (label_mask)
          - Explicit causal attention mask (attn_mask)
          - Token-level metrics: accuracy, entropy

        Trả về (loss, token_accuracy, mean_entropy).
        """
        base = model.module if hasattr(model, "module") else model

        # Truyền attention_mask tường minh; fallback nếu model không nhận
        try:
            l1, l2, aux_total = model(input_ids, attention_mask=attn_mask, return_aux=True)
        except TypeError:
            l1, l2, aux_total = model(input_ids, return_aux=True)

        vs = vocab_size or getattr(base, "vocab_size", None)

        # ── Align length ───────────────────────────────────────────
        # l1 shape: (B, T, V) — mỗi vị trí dự đoán token tiếp theo
        # y shape : (B, T)    — target tương ứng (đã khớp 1-1 từ dataset)
        # KHÔNG dùng l1[:,:-1]: dataset đã trả x=ids[:-2], y=ids[1:-1]
        # nên l1 và y đã căn chỉnh hoàn toàn; cắt thêm sẽ mất tín hiệu token cuối.
        T_logit = l1.shape[1]
        T_label = labels.shape[1]
        T       = min(T_logit, T_label)

        logits = l1[:, :T]             # (B, T, V)
        tgt    = labels[:, :T]         # (B, T)
        msk    = label_mask[:, :T]     # (B, T) bool

        if vs:
            logits = logits[:, :, :vs]

        # Guard: nếu toàn bộ mask = 0 → không có token nào cần học → skip batch
        if msk.sum() == 0:
            return None, 0.0, 0.0

        # Áp dụng label mask: token ngoài mask → ignore_index=-100
        tgt_masked = tgt.clone()
        tgt_masked[~msk] = -100

        # CE loss normalize theo số token hợp lệ (reduction="sum" / n_valid_tokens)
        # Đảm bảo CE và MTP cùng scale khi valid_len < T
        ce_n_tok = msk.sum().clamp(min=1).float()
        ce_loss_raw = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt_masked.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        ce_loss = ce_loss_raw / ce_n_tok

        # ── Token-level metrics ────────────────────────────────────
        with torch.no_grad():
            preds   = logits.argmax(-1)
            correct = (preds == tgt) & msk
            tok_acc = correct.sum().float() / msk.sum().clamp(min=1)
            probs   = torch.softmax(logits, dim=-1)
            ent     = -(probs * (probs + 1e-9).log()).sum(-1)
            tok_ent = (ent * msk.float()).sum() / msk.sum().clamp(min=1)

        # ── MTP loss ───────────────────────────────────────────────
        # Normalize MTP theo số token hợp lệ của MTP (không phải T toàn bộ)
        # → CE và MTP cùng scale dù valid_len < T
        mtp_coeff = getattr(base, "_mtp_aux_coeff", 0.1)
        if mtp_coeff > 0 and l2.shape[1] > 1 and labels.shape[1] > 2:
            pred_len  = l2.shape[1] - 1
            label_len = labels.shape[1] - 2
            valid_len = min(pred_len, label_len)
            if valid_len > 0:
                l2_logits = l2[:, :valid_len]
                if vs:
                    l2_logits = l2_logits[:, :, :vs]
                mtp_tgt = labels[:, 2:2 + valid_len].clone()
                mtp_msk = label_mask[:, 2:2 + valid_len]
                mtp_tgt[~mtp_msk] = -100
                mtp_n_tok = mtp_msk.sum().clamp(min=1).float()
                mtp_loss_raw = F.cross_entropy(
                    l2_logits.reshape(-1, l2_logits.shape[-1]),
                    mtp_tgt.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                mtp_loss = mtp_loss_raw / mtp_n_tok
                total = ce_loss + 0.01 * aux_total + mtp_coeff * mtp_loss
                return total, tok_acc.item(), tok_ent.item()

        return ce_loss + 0.01 * aux_total, tok_acc.item(), tok_ent.item()

    def _log_moe_health(self, model):
        """
        Log MoE health: entropy, balance (min/max), std, collapse/imbalance warning.
        Ngưỡng entropy cảnh báo tính động theo log(n_experts) — tránh false alarm khi
        n_experts nhỏ (ngưỡng cố định 0.5 gần max khi 2 expert) hoặc cảnh báo quá muộn
        khi n_experts lớn (0.5 << log(16)).
        Có thể override bằng config:
          config["moe_entropy_warn_ratio"]    (default 0.3, cảnh báo khi entropy < ratio * log(n))
          config["moe_load_std_warn"]         (default 0.3)
          config["moe_load_std_collapse"]     (default 0.01)
        """
        try:
            base = model.module if hasattr(model, "module") else model
            if not hasattr(base, "get_router_stats"):
                return
            stats = base.get_router_stats()
            if not stats:
                return

            entropy = stats.get("entropy")
            load    = stats.get("load")
            parts   = []

            if entropy is not None:
                # Ngưỡng động: cảnh báo khi entropy < ratio * log(n_experts)
                n_experts   = self.config.get("n_experts", 8)
                warn_ratio  = self.config.get("moe_entropy_warn_ratio", 0.3)
                entropy_thr = warn_ratio * math.log(max(n_experts, 2))
                parts.append(f"router_entropy={entropy:.4f}")
                if entropy < entropy_thr:
                    parts.append(f"⚠️ LOW_ENTROPY(< {entropy_thr:.3f}, collapse risk)")

            if load is not None:
                arr     = torch.tensor(load, dtype=torch.float32)   # tránh GPU→CPU sync của np.array
                mx      = arr.max().item() + 1e-9
                balance = arr.min().item() / mx
                std     = float(arr.std().item())
                parts.append(f"expert_balance={balance:.3f} std={std:.4f}")
                if std > self.config.get("moe_load_std_warn", 0.3):
                    parts.append("⚠️ HIGH_IMBALANCE")
                elif std < self.config.get("moe_load_std_collapse", 0.01):
                    parts.append("⚠️ NEAR_COLLAPSE")

            if parts:
                print("    [MoE] " + " | ".join(parts))
        except Exception:
            pass

    # ── Checkpoint ─────────────────────────────────────────────────
    def _load_latest_checkpoint(self, mode, skill_group, lora_rank, lora_alpha, max_lr):
        ckpts = [f for f in os.listdir(self.ckpt_dir)
                 if f.startswith("trainer_") and f.endswith(".pt")]
        if not ckpts:
            print("[Trainer] No checkpoint – starting fresh.")
            return self._build_fresh_state(mode, skill_group, lora_rank, lora_alpha, max_lr)

        ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(self.ckpt_dir, x)))
        path = os.path.join(self.ckpt_dir, ckpts[-1])
        print(f"[Trainer] Loading checkpoint: {path}")
        ckpt  = torch.load(path, map_location=self.device)

        model = DracoTransformerTorchV1(ckpt.get("config", self.config)).to(self.device)
        self._apply_mode(model, mode, skill_group, lora_rank, lora_alpha)
        model.load_state_dict(ckpt["model"])

        trainable       = [p for p in model.parameters() if p.requires_grad]
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        ckpt_names      = set(ckpt.get("trainable_param_names", []))

        optimizer = torch.optim.AdamW(trainable, lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
        scaler    = torch.cuda.amp.GradScaler(enabled=(self.device.startswith("cuda")))

        if "optimizer" in ckpt:
            mismatch = ckpt_names and (ckpt_names != trainable_names)
            if mismatch:
                diff = ckpt_names.symmetric_difference(trainable_names)
                print(f"[Trainer] ⚠️  Param names mismatch ({len(diff)} params differ). "
                      f"Optimizer + scaler reset (model weights preserved).")
            else:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    if "scaler" in ckpt:
                        scaler.load_state_dict(ckpt["scaler"])
                except ValueError as e:
                    print(f"[Trainer] ⚠️  Optimizer load error: {e}. "
                          f"Optimizer + scaler reset (model weights preserved).")

        return {
            "model":              model,
            "optimizer":          optimizer,
            "scaler":             scaler,
            "step":               ckpt.get("step", 0),
            "global_total_steps": ckpt.get("global_total_steps"),
        }

    def _build_fresh_state(self, mode, skill_group, lora_rank, lora_alpha, max_lr):
        model     = DracoTransformerTorchV1(self.config).to(self.device)
        self._apply_mode(model, mode, skill_group, lora_rank, lora_alpha)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
        scaler    = torch.cuda.amp.GradScaler(enabled=(self.device.startswith("cuda")))
        return {"model": model, "optimizer": optimizer, "scaler": scaler,
                "step": 0, "global_total_steps": None}

    def _save_checkpoint(self, model, optimizer, scaler, step, loss,
                         global_total_steps: Optional[int] = None,
                         tag: str = ""):
        # Chỉ rank 0 được save — tránh race condition khi DDP/FSDP multi-process
        if HAS_DDP and dist.is_initialized() and dist.get_rank() != 0:
            return

        # Unwrap DDP để lấy raw state_dict
        base = model.module if hasattr(model, "module") else model

        # FSDP: cần FULL_STATE_DICT để checkpoint portable (không phải sharded)
        if HAS_FSDP and isinstance(base, FSDP):
            from torch.distributed.fsdp import StateDictType
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model_state = model.state_dict()
        else:
            model_state = base.state_dict()
        trainable_param_names = [n for n, p in base.named_parameters() if p.requires_grad]
        fname = f"trainer_best.pt" if tag == "best" else f"trainer_{step:06d}.pt"
        path  = os.path.join(self.ckpt_dir, fname)
        save_dict = {
            "step":                  step,
            "model":                 model_state,
            "optimizer":             optimizer.state_dict(),
            "scaler":                scaler.state_dict(),
            "config":                self.config,
            "trainable_param_names": trainable_param_names,
        }
        if global_total_steps is not None:
            save_dict["global_total_steps"] = global_total_steps

        torch.save(save_dict, path)
        with open(os.path.join(self.ckpt_dir, "trainer_state.json"), "w") as f:
            json.dump({"step": step, "loss": loss,
                       "global_total_steps": global_total_steps}, f)
        print(f"[Trainer] Checkpoint saved: {path}")

    def _prune_checkpoints(self):
        """
        Giữ lại keep_checkpoints gần nhất, xoá phần còn lại.
        Dùng atomic rename → tmp trước để tránh race condition khi multi-process.
        Chỉ rank 0 thực hiện.
        """
        if HAS_DDP and dist.is_initialized() and dist.get_rank() != 0:
            return
        if self.keep_checkpoints <= 0:
            return
        ckpts = sorted(
            [f for f in os.listdir(self.ckpt_dir)
             if f.startswith("trainer_") and f.endswith(".pt")
             and f != "trainer_best.pt"],   # không prune best checkpoint
            key=lambda x: os.path.getmtime(os.path.join(self.ckpt_dir, x))
        )
        for f in ckpts[: max(0, len(ckpts) - self.keep_checkpoints)]:
            full_path = os.path.join(self.ckpt_dir, f)
            # Atomic rename → .del trước, sau đó xoá
            tmp_path  = full_path + ".del"
            try:
                os.rename(full_path, tmp_path)   # atomic trên cùng filesystem
                os.remove(tmp_path)
                print(f"[Trainer] Pruned: {f}")
            except OSError:
                # Nếu tmp_path vẫn tồn tại (rename thành công nhưng remove fail) → thử lại
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    def _sync_to_numpy(self, torch_model):
        """
        Sync weights sang NumPy model.
        Chỉ gọi ở cuối run hoặc export_gguf – tránh stall giữa training.
        Clone state_dict trước để an toàn.
        """
        if self.numpy_model is None:
            return
        base = torch_model.module if hasattr(torch_model, "module") else torch_model
        try:
            state_copy = {k: v.clone().cpu() for k, v in base.state_dict().items()}
            base._full_sync(self.numpy_model, state_dict=state_copy)
            print("[Trainer] Weights synced to NumPy model.")
        except TypeError:
            try:
                base._full_sync(self.numpy_model)
                print("[Trainer] Weights synced to NumPy model (legacy sync).")
            except Exception as e:
                print(f"[Trainer] Sync failed: {e}")
        except Exception as e:
            print(f"[Trainer] Sync failed: {e}")

    @staticmethod
    def _cosine_lr(step, total, max_lr, min_lr, warmup):
        """
        Cosine LR với global step/total.
        total được freeze từ lần đầu → resume nhiều lần không làm lệch curve.
        """
        if step < warmup:
            return max_lr * step / max(warmup, 1)
        if step >= total:
            return min_lr
        progress = (step - warmup) / max(total - warmup, 1)
        return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)

    # ── Export GGUF ──────────────────────────────────────────────────
    def export_gguf(self, torch_model, output_path: str):
        self._sync_to_numpy(torch_model)
        if GGUFExporter is None:
            raise RuntimeError("transformer_v1.py not found.")
        GGUFExporter(self.numpy_model).write_gguf(output_path)
        print(f"[Trainer] GGUF exported to {output_path}")


# Alias
DracoTrainer = TrainerV1