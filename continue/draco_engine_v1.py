# Copyright 2026 Draco Studio and DUCNGUYEN-creator
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
DracoAI V1 — Main Engine (Production‑ready)
===========================================
Tích hợp đầy đủ ThinkingEngineV1, Memory, Transformer thông qua TransformerBridge.
Tương thích với: engine_v1.py, transformer_v1.py, tokenizer.py, memory_v1.py

Changelog V1 (fixes):
  [A] _register_query_hash: FIFO eviction + threading.Lock chống race condition
  [B] intent_bias_arr: guard None; dùng np.clip() thay np.tanh() (nhanh hơn)
  [C] skill_calc: regex chỉ match expr hợp lệ (số-op-số); validate expr không rỗng;
      ZeroDivisionError trả thông báo rõ thay vì silent None
  [D] _auto_learn: filter stopwords + BAD_PATTERNS (adj/adv) + context words
  [E] context trimming MAX_HISTORY_TURNS; _trim_history dùng for-loop thay while
  [F] _safe_ltm_store: normalize text trước khi hash (bỏ punctuation)
  [G] _auto_learn: hallucination flags guard
  [H] skill_read_file, skill_create_file, skill_write_report
  [I] skill_remember: re.IGNORECASE giữ casing key/value
  [J] skill_recall: word-boundary regex chống false positive
  [K] skill sandbox ai_workspace/: os.makedirs chỉ gọi 1 lần trong __init__
  [L] bridge.generate: try/except ở cả chat() và stream_chat()
  [M] import hashlib + numpy + ast + operator chuyển lên đầu file
  [N] _trim_history: fallback khi trim ra rỗng
  [O] use_mirostat: chỉ bật khi len(prompt_ids) > 50
  [P] _try_skills: calc chỉ trigger khi có số + operator; log warning thay vì pass
  [Q] _is_safe_path: makedirs chuyển vào __init__, hàm chỉ check thuần
  [R] skill_create_file: path sandbox dùng os.sep chặt bypass edge-case "../"
  [S] _auto_learn: thêm filter context words ("trong", "khi", "nếu", "với"...)
  [T] request_elevated_permission: hỏi người dùng trước khi thao tác quyền cao

Changelog V1 (patch-2 – bug fixes):
  [C+] _safe_calc: replace ^ → ** trước ast.parse; ^ = power, không phải XOR
  [D+] _auto_learn: regex subject phải bắt đầu chữ hoa; guard tối thiểu 2 từ
        tránh học chuỗi mơ hồ như "Trời hôm nay AI là..."
  [F+] _safe_ltm_store: thêm unicodedata.normalize("NFKC") trước hash;
        é vs é (2 dạng unicode) được coi là giống nhau
  [P+] _try_skills calc: dùng regex r'\d+\s*[op]\s*\d+' thay has_number+has_op;
        tránh false positive như "version 1.2.3-alpha"
  [R+] skill_create_file: dùng os.path.relpath để hiển thị output path an toàn
  [T+] request_elevated_permission: callback abstraction (permission_handler);
        không block thread trong môi trường web/async/GUI

Changelog V1 (patch-3 – correctness & safety fixes):
  [A+] _register_query_hash: >= thay == cho eviction check (safe với deque cực đoan)
  [C2] _safe_calc: log debug exception thay vì silent fail; dễ debug hơn
  [D2] _auto_learn: sửa guard logic sai (subj[0].isupper() luôn True do regex);
        đơn giản thành len(subj.split()) < 2
  [G+] _auto_learn: thêm avg_conf threshold (< 0.7 → skip); chặn hallucination
        leak vào LTM khi model không tự tin
  [G+] stream_chat: tính stream_conf từ trung bình confidence các token,
        truyền vào _auto_learn
  [R2] skill_create_file: regex content dùng .+? + $ thay .+ greedy;
        tránh ăn hết text không liên quan

Changelog V1 (patch-4 – hardening & correctness):
  [A2] _register_query_hash: popleft() evict atomically thay [0]+discard;
        loại bỏ khả năng double-discard khi multi-thread
  [C3] _safe_calc: AST whitelist strict (Expression/BinOp/UnaryOp/Constant/ops);
        chặn ast.Call, ast.Name, ast.Attribute – biến thành math DSL thuần
  [S2] _auto_learn: context word check dùng token split() thay substring;
        'AI trong tương lai' không còn lọt qua filter
  [P2] _try_skills: multi-dispatch – collect TẤT CẢ skill triggered, join kết quả;
        tránh bỏ sót 'remember' khi 'calc' trigger trước
  [B2] intent_bias_arr: log warning khi clip fail thay vì silent degrade;
        áp dụng cả chat() và stream_chat()
  [O+] use_mirostat: heuristic thông minh hơn – bật thêm khi temperature > 0.9
        hoặc intent == 'creative'; áp dụng cả chat() và stream_chat()
  [F2] _safe_ltm_store: đổi MD5 → blake2b(digest_size=16);
        nhanh hơn + collision-resistant hơn ở scale lớn
  [stream] stream_chat: memory update SAU khi yield loop kết thúc;
        tránh lưu full_reply khi client cancel stream giữa chừng

Changelog V1 (patch-5 – robustness & security):
  [C4] _safe_calc: chặn bool (True→1.0), string, inf/nan khỏi float() Constant;
        loại bỏ Python silent coercion
  [H+] skill_read_file: error message dùng relpath thay raw_path;
        không leak system path (/etc/passwd...) trong output
  [Q2] _is_safe_path: realpath() thay abspath() – chặn symlink attack + unicode homoglyph
  [P3] _try_skills: hyphen đặt cuối character class [+*\/^-] tránh descending range;
        không còn FutureWarning trong Python re
  [D3] _auto_learn: filter desc chứa subjective word (rất/very/khá...);
        tránh học opinion như "Python là rất mạnh"
  [perf] _auto_learn: giới hạn tối đa 3 fact/response – tránh CPU spike với LLM dài
  [stream2] stream_chat: completed flag – chỉ lưu memory khi generate xong hoàn toàn,
        không lưu khi bị interrupt/exception giữa chừng
  [err] chat(): thêm error_code + retry_hint vào error response;
        client có thể xử lý lỗi có cấu trúc thay vì parse string

Changelog V1 (patch-6 – correctness, safety & new skills):
  [fix-1] _safe_calc: thêm ast.UAdd (unary +) vào _ops + _ALLOWED_NODES;
          giới hạn recursion depth <= 50 trong _eval(node, depth) – chặn stack overflow
  [fix-2] _try_skills calc: sync regex với skill_calc dùng pattern đầy đủ
          thay pattern cũ thiếu decimal support
  [fix-3] _auto_learn: guard isupper() cho chuỗi dài (> 5 ký tự) tránh "AI LÀ GÌ";
          thay hard-reject 1 từ bằng allow-list kỹ thuật (Python, AI, GPU, CPU, ...)
  [fix-4] _register_query_hash: rollback queue.pop() nếu set.add() fail – atomic
  [fix-5] _trim_history: fallback cải tiến – tìm user gần nhất từ cuối list
  [fix-6] stream_chat: docstring WARNING rõ ràng về buffered stream limitation
  [new]   skill_save_code: gom khung ```lang code``` → lưu file sandbox-safe
  [new]   skill_write_code: detect yêu cầu code + inject markdown format hint

Changelog V1 (patch-7 – correctness, safety & architecture):
  [fix-p7-1] _safe_calc: thay silent-None bằng exception hierarchy (CalcInvalidError /
          CalcRuntimeError); caller phân biệt "biểu thức sai" vs "chia cho 0" rõ ràng;
          np.isfinite() thay manual nan/inf check
  [fix-p7-2] _try_skills: priority system (calc=100, save_code=70, ...) + sort trước khi chạy;
          calc short-circuit ngay sau khi trigger – không "dính combo" với remember;
          skill_write_code XÓA khỏi _skills, thay bằng _detect_code_request() trả
          (bool, lang) để chat()/stream_chat() inject system hint vào prompt – không
          trả "[CODE_FORMAT_HINT]..." cho user nữa
  [fix-p7-3] _auto_learn: thêm proper-noun check (≥1 từ Hoa-đầu-thường);
          reject subject chứa động từ nối (Là/Is/Means); reject desc bắt đầu bằng adj/adv
  [fix-p7-4] _safe_ltm_store: thêm semantic dedup (ltm.search score > 0.9 → skip)
          trước hash check – giảm duplicate "AI là gì" vs "AI là cái gì"
  [fix-p7-5] stream_chat: thêm skill shortcut (nhất quán với chat());
          yield_done flag – chỉ lưu memory khi yield loop thực sự hoàn thành
  [fix-p7-6] intent_bias_arr: degrade về zeros_like thay None – giữ shape/routing signal
          nhẹ thay vì mất hoàn toàn; áp dụng cả chat() và stream_chat()
  [fix-p7-7] _trim_history: fallback trả history[max(0, i-1):] (giữ 1 msg trước user
          gần nhất để context); nếu không có user → history[-3:] thay -2
  [fix-base] _SKILL_BASE_DIR: dùng __file__ thay getcwd() – stable dù chạy từ bất kỳ cwd;
          dead-code trong skill_read_file (else branch) đã được đơn giản hóa

Changelog V1 (patch-8 – integration & cleanup):
  [fix-p8-1] _auto_learn: gộp 2 set _ALLOW_SINGLE/_ALLOW_SINGLE_2 giống hệt nhau thành
          1 module-level frozenset _LEARN_ALLOW_SINGLE – tránh lặp code, dễ bảo trì
  [fix-p8-2] chat() + stream_chat(): tích hợp _detect_code_request() – inject SYS_HINT
          vào user_input_for_engine/user_msg_for_engine trước khi gọi thinking.process();
          user_input gốc được giữ nguyên cho memory/ltm để không leak hint cho user
  [fix-p8-3] skill_create_file: thêm auto-backup (.bak) trước khi ghi đè file đã tồn tại;
          dùng os.replace() (atomic trên POSIX) + log warning nếu backup fail
  [fix-p8-4] stream_chat: xóa yield_done flag dư thừa (sau vòng for pieces chạy hết,
          GeneratorExit không thể xảy ra ở code sau yield); _generate_completed[0]
          là guard duy nhất và đủ

Changelog V1 (patch-9 – architecture & correctness):
  [fix-p9-1] _try_skills: thay thế priority-only bằng DETERMINISTIC/SIDE_EFFECT hierarchy;
          DETERMINISTIC (calc, recall, read_file, write_report) → ưu tiên output, short-circuit
          calc ngay; SIDE_EFFECT (remember, create_file, save_code) → luôn chạy nhưng output
          phụ, append sau deterministic nếu cùng trigger; tránh "tính 2+2 và nhớ X" ghi
          memory không mong muốn khi calc đã thành công.
  [fix-p9-2] _auto_learn: thêm Layer 3 grammar heuristic – subject phải match
          r"\b[A-ZÀ-Ỹ][a-zà-ỹ]+\b" (noun phrase có từ Hoa-đầu-thường); acronym kỹ thuật
          trong _LEARN_ALLOW_SINGLE được bypass; tránh LTM poisoning với "Python Là rất mạnh".
  [fix-p9-3] stream_chat: khôi phục yield_done flag – chỉ lưu memory khi cả 2 điều kiện
          thỏa mãn: generate hoàn thành (_generate_completed[0]) VÀ toàn bộ token đã
          deliver cho client (yield_done); tránh lưu khi generator bị drop giữa stream.

Changelog V1 (patch-11 – hybrid tool-calling implementation & hardening):
  [hybrid-1] _parse_tool_call(text): trích (action, input) từ [ACTION: input] bằng
          _TOOL_CALL_PATTERN; chỉ chấp nhận action trong ALLOWED_TOOLS_FOR_LLM
          (calc, recall, read_file) – an toàn, không để LLM gọi side-effect skill.
  [hybrid-2] _TOOL_INPUT_PREFIX + _execute_tool: skill read_file/recall dùng regex
          bắt câu lệnh đầy đủ; khi LLM truyền input thuần ("report.txt", "Python")
          cần ghép tiền tố tương ứng ("đọc file ", "bạn có nhớ ") để regex khớp –
          fix bug trả None thay vì kết quả thực sự.
  [hybrid-3] tool_hint: hướng dẫn LLM chỉ xuất đúng một cú pháp tool trên một dòng
          riêng, không kèm văn bản; ngăn _parse_tool_call bỏ sót tool call thứ hai
          khi LLM viết text + [CALC:...] + text trong cùng một lượt.
  [hybrid-4] Loop dùng base_conversation + tool_results_this_turn tách biệt;
          trim chỉ áp dụng lên base – tool results không bao giờ bị cắt giữa chừng;
          loại bỏ context overflow khi tool loop kéo dài nhiều rounds.
  [hybrid-note] stream_chat() giữ nguyên pipeline cũ (buffered stream) –
          tool loop không áp dụng cho stream; documented rõ trong docstring.

Changelog V1 (patch-10 – bug fixes & hardening):
  [fix-p10-1] _LEARN_BAD_OPINION_WORDS: khai báo frozenset module-level (tiếng Việt + tiếng Anh);
          fix NameError crash trong _auto_learn khi gặp desc chứa từ tiêu cực.
  [fix-p10-2b] _safe_calc: thêm post-compute bomb check – abs(result) > 1e12 → raise _Invalid;
          chặn 999999999**2 (operand hợp lệ nhưng kết quả cực lớn → CPU/RAM spike).
  [fix-p10-3] _try_skills: join TẤT CẢ deterministic_results thay vì chỉ [0];
          recall + read_file cùng trigger → cả 2 output đều hiển thị; xóa calc_triggered dead-code.
  [fix-p10-4] skill_create_file: bọc os.replace() trong try/except FileNotFoundError;
          chịu race condition multi-thread: file bị xóa giữa check và replace → bỏ qua, tiếp tục.
  [fix-p10-7c] _has_symlink_in_path(): helper mới – duyệt toàn bộ thành phần path,
          phát hiện symlink nằm trong parent dirs (không chỉ tại target);
          _is_safe_path() gọi helper này trước realpath() để chặn bypass qua dir chain.
"""

# ── [M] Imports tập trung ─────────────────────────────────────────────────────
import ast
import hashlib
import logging
import operator
import os
import re
import time
import unicodedata
from collections import deque
from threading import Lock
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from tokenizer import BPETokenizer
from memory_v1 import LongTermMemoryV1, WorkingMemoryV1, KVCacheBuffer
from engine_v1 import ThinkingEngineV1
from transformer_v1 import (
    DracoTransformerV1,
    TransformerBridge,
    GGUFExporter,
)

_log = logging.getLogger(__name__)

# ── [G] Hallucination guard ────────────────────────────────────────────────────
HALLUCINATION_FLAGS = [
    "tôi không chắc", "có thể là", "tôi đoán", "theo như tôi biết thì",
    "i'm not sure", "i think", "maybe", "i believe", "possibly",
    "tôi nghĩ rằng có thể", "chưa rõ",
]

# ── [D] Stopwords cho _auto_learn (chỉ từ đơn – cụm bị bỏ vì startswith từ đơn
#       đã bắt trước rồi, giữ cụm không có tác dụng thực tế)
_LEARN_STOPWORDS = {
    "tôi", "bạn", "nó", "họ", "chúng", "mình", "này", "đây", "kia",
    "the", "a", "an", "it", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "they", "my", "your", "our", "their",
}

# [D] Tính từ / trạng từ phổ biến – subject chứa → bỏ qua
_LEARN_BAD_PATTERNS = [
    "rất", "khá", "nhiều", "ít", "đang", "đã", "sẽ",
    "really", "very", "quite", "just", "often", "always", "never",
]

# [fix-p10-4a] Từ mang cảm xúc/ý kiến tiêu cực trong desc → bỏ (tránh học biased statements)
# Ví dụ: "Python là một ngôn ngữ tệ hại và lỗi thời"
_LEARN_BAD_OPINION_WORDS: frozenset = frozenset({
    # Tiếng Việt
    "tệ", "tệ hại", "kém", "dở", "lỗi thời", "lỗi", "xấu", "chán",
    "yếu", "cũ", "lạc hậu", "vô dụng", "phức tạp", "khó chịu",
    # English
    "bad", "terrible", "awful", "outdated", "poor", "weak", "ugly",
    "useless", "complex", "horrible", "worst", "inferior", "broken",
})

# [S] Từ ngữ cảnh/điều kiện – subject chứa → bỏ qua (tránh học câu điều kiện)
_LEARN_CONTEXT_WORDS = [
    "trong", "khi", "nếu", "với", "bởi", "vì", "để",
    "while", "if", "when", "because", "since", "for",
]

# [fix-p8-1] Gộp 2 set _ALLOW_SINGLE/_ALLOW_SINGLE_2 thành 1 module-level constant
# Tránh lặp code và dễ bảo trì
_LEARN_ALLOW_SINGLE = frozenset({
    "Python", "AI", "GPU", "CPU", "LLM", "API", "RAM",
    "Java", "Rust", "Go", "Swift", "Kotlin", "C++",
})

# ── [E] Context window limit ───────────────────────────────────────────────────
MAX_HISTORY_TURNS = 20  # 1 lượt = user + assistant

# ── [Hybrid Tool Calling] ──────────────────────────────────────────────────────
# Số lần LLM được phép gọi tool trong một lượt chat() (tránh vòng lặp vô tận)
MAX_TOOL_ROUNDS = 3

# Chỉ những skill DETERMINISTIC, không side-effect mới được LLM tự động gọi.
# Không bao giờ để LLM tự gọi remember / create_file / save_code.
ALLOWED_TOOLS_FOR_LLM: frozenset = frozenset({"calc", "recall", "read_file"})

# Regex bắt cú pháp [ACTION: input] mà LLM sinh ra
# Ví dụ: [CALC: 2+3*4]  [RECALL: Python]  [READ_FILE: notes.txt]
_TOOL_CALL_PATTERN = re.compile(r"\[(\w+):\s*(.*?)\]", re.IGNORECASE)

# ── [K][Q] Skill file sandbox – dùng __file__ thay getcwd() để stable dù chạy từ bất kỳ cwd
# [fix-base] os.getcwd() thay đổi theo working dir → tạo folder lung tung
_SKILL_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_workspace")


def _has_symlink_in_path(path: str) -> bool:
    """[fix-p10-7c] Kiểm tra có symlink nào trong TẤT CẢ các thành phần của path không.
    Ngăn tấn công qua: ai_workspace/safe/evil_link/file.txt
    trong đó evil_link là symlink trỏ ra ngoài sandbox.
    """
    parts = os.path.abspath(path).replace("\\", "/").split("/")
    cur = ""
    for p in parts:
        cur = cur + os.sep + p if cur else p
        if cur and os.path.islink(cur):
            return True
    return False


def _is_safe_path(path: str, base: str = _SKILL_BASE_DIR) -> bool:
    """[Q][R] Kiểm tra path nằm trong sandbox.
    Dùng os.sep để chặt bypass: 'ai_workspace_evil' không pass được.
    [Q2] realpath() thay abspath() – chặn thêm symlink attack + unicode homoglyph.
    [fix-p10-7a] Chặn symlink tại target trước khi realpath() resolve ra ngoài.
    [fix-p10-7b] Case-insensitive comparison trên Windows (NTFS không phân biệt case).
    [fix-p10-7c] Chặn symlink trong toàn bộ directory chain (không chỉ tại target).
    Không gọi makedirs ở đây – đã làm trong __init__.
    """
    # Chặn symlink tại target – tránh tạo symlink trỏ ra ngoài sandbox rồi ghi đè
    if os.path.islink(path):
        return False
    # [fix-p10-7c] Chặn symlink trong parent directory chain
    if _has_symlink_in_path(path):
        return False
    real_path = os.path.realpath(path)
    real_base = os.path.realpath(base)
    # Windows: NTFS case-insensitive → normalize cả hai về lower để so sánh đúng
    if os.name == "nt":
        real_path = real_path.lower()
        real_base = real_base.lower()
    return real_path == real_base or real_path.startswith(real_base + os.sep)


# ══════════════════════════════════════════════════════════════════════════════
# DracoEngineV1
# ══════════════════════════════════════════════════════════════════════════════
class DracoEngineV1:
    """Điều phối viên chính – kết nối tất cả các tầng."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.ckpt = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # [K][Q] Tạo sandbox một lần duy nhất ở đây
        os.makedirs(_SKILL_BASE_DIR, exist_ok=True)

        # ── Tokenizer ─────────────────────────────────────────────────
        tok_dir = os.path.join(checkpoint_dir, "tokenizer")
        if os.path.exists(os.path.join(tok_dir, "tokenizer_draco.json")):
            self.tokenizer = BPETokenizer.load(tok_dir)
        else:
            print("[Engine] Không tìm thấy tokenizer – khởi tạo rỗng")
            self.tokenizer = BPETokenizer()

        # ── Mô hình ────────────────────────────────────────────────────
        if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
            self.model = DracoTransformerV1.load_weights(checkpoint_dir)
        else:
            config = {
                "d_model": 128, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2,
                "head_dim": 32, "d_ff": 512, "n_experts": 8,
                "vocab_size": max(151936, self.tokenizer.vocab_size),
                "window": 1024,
            }
            self.model = DracoTransformerV1(config)

        # ── Bridge ─────────────────────────────────────────────────────
        self.bridge = TransformerBridge(numpy_model=self.model)

        # ── Memory ────────────────────────────────────────────────────
        mem_dir = os.path.join(
            os.path.dirname(os.path.abspath(checkpoint_dir)), "memory"
        )
        self.ltm = LongTermMemoryV1(mem_dir)
        kv_buf = KVCacheBuffer()
        self.working = WorkingMemoryV1(
            self.ltm, kv_buf,
            max_tokens=self.model.window,
            tokenizer=self.tokenizer,
        )

        # ── Thinking Engine ────────────────────────────────────────────
        self.thinking = ThinkingEngineV1(
            max_experts=4,
            bridge=self.bridge,
            tokenizer=self.tokenizer,
        )

        # ── Skills ─────────────────────────────────────────────────────
        self._register_skills()

        # ── [A] Dedup hash queue + lock chống race condition ───────────
        self._hash_queue: deque = deque(maxlen=512)
        self._hash_set:   set   = set()
        self._hash_lock:  Lock  = Lock()

        print(
            f"[DracoAI] Sẵn sàng! "
            f"(vocab={self.tokenizer.vocab_size}, "
            f"bridge={'đã kết nối' if self.bridge.is_connected() else 'stub'})\n"
        )

    # ══════════════════════════════════════════════════════════════════
    # [A] Dedup helper – thread-safe FIFO eviction
    # ══════════════════════════════════════════════════════════════════
    def _register_query_hash(self, h: str) -> bool:
        """Trả về True nếu hash mới, False nếu trùng. Thread-safe."""
        with self._hash_lock:
            if h in self._hash_set:
                return False
            # [A+] popleft() atomically evict FIFO – tránh double-discard multi-thread
            if len(self._hash_queue) >= self._hash_queue.maxlen:
                oldest = self._hash_queue.popleft()
                self._hash_set.discard(oldest)
            # [fix-4][fix-p10-1] atomic append+add – rollback nếu add fail.
            # Dùng remove(h) thay pop() để tránh pop nhầm phần tử của thread khác
            # trong trường hợp multi-thread cực đoan (thread B append sau thread A).
            self._hash_queue.append(h)
            try:
                self._hash_set.add(h)
            except Exception:
                try:
                    self._hash_queue.remove(h)  # xóa đúng phần tử h, không pop() mù
                except ValueError:
                    pass  # đã bị evict bởi thread khác, bỏ qua
                raise
            return True

    # ══════════════════════════════════════════════════════════════════
    # Safe calculator (AST – không exec)
    # ══════════════════════════════════════════════════════════════════
    # [fix-p7-1] Exception hierarchy để phân biệt lỗi rõ ràng thay vì silent None
    class _CalcError(Exception):
        """Base cho tất cả lỗi _safe_calc."""
    class _CalcInvalidError(_CalcError):
        """Biểu thức không hợp lệ (syntax, type, unsupported node)."""
    class _CalcRuntimeError(_CalcError):
        """Lỗi runtime hợp lệ (ví dụ: chia cho 0)."""

    @staticmethod
    def _safe_calc(expr: str) -> Optional[float]:
        """Tính biểu thức toán học an toàn qua AST.
        Trả về float nếu thành công.
        Raise _CalcInvalidError nếu biểu thức không hợp lệ.
        Raise _CalcRuntimeError nếu lỗi runtime (ZeroDivision).
        [fix-p7-1] Phân biệt loại lỗi thay vì silent None – caller quyết định UX.
        """
        # Tham chiếu class qua closure (staticmethod không có self)
        _Invalid = DracoEngineV1._CalcInvalidError
        _Runtime = DracoEngineV1._CalcRuntimeError

        _ops = {
            ast.Add:  operator.add,
            ast.Sub:  operator.sub,
            ast.Mult: operator.mul,
            ast.Div:  operator.truediv,
            ast.Pow:  operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,   # [fix-1] unary plus cho DSL nhất quán
        }

        def _eval(node, depth: int = 0):
            # [fix-1] giới hạn recursion – chặn stack overflow với biểu thức lồng sâu
            if depth > 50:
                raise _Invalid("Expression nested too deep (>50)")
            if isinstance(node, ast.Constant):
                # [C4] Chặn bool (True→1.0), string, None
                if isinstance(node.value, bool):
                    raise _Invalid("Bool literals not allowed")
                if isinstance(node.value, (int, float)):
                    v = float(node.value)
                    # [C4] Chặn inf/nan do "1e999" v.v.
                    if not np.isfinite(v):
                        raise _Invalid("NaN/Inf not allowed")
                    return v
                raise _Invalid(f"Unsupported constant type: {type(node.value).__name__}")
            if isinstance(node, ast.BinOp):
                op = _ops.get(type(node.op))
                if op is None:
                    raise _Invalid(f"Unsupported binary operator: {type(node.op).__name__}")
                lv = _eval(node.left,  depth + 1)
                rv = _eval(node.right, depth + 1)
                # [fix-p10-2] Compute bomb prevention: 2**2**2**... và số quá lớn
                # Giới hạn magnitude operand tránh freeze CPU dù AST hợp lệ
                if abs(lv) > 1e9 or abs(rv) > 1e9:
                    raise _Invalid("Operand magnitude too large (>1e9)")
                if isinstance(node.op, ast.Pow) and abs(rv) > 100:
                    raise _Invalid("Exponent too large (>100)")
                try:
                    result = op(lv, rv)
                except ZeroDivisionError:
                    raise _Runtime("Division by zero")
                # [fix-p10-2b] Post-compute bomb: operand hợp lệ nhưng result cực lớn
                # Ví dụ: 999999999**2 → 9.99e17 → CPU/RAM spike
                if isinstance(result, float) and not np.isfinite(result):
                    raise _Invalid("Result is NaN/Inf")
                if abs(result) > 1e12:
                    raise _Invalid("Result magnitude too large (>1e12)")
                return result
            if isinstance(node, ast.UnaryOp):
                op = _ops.get(type(node.op))
                if op is None:
                    raise _Invalid(f"Unsupported unary operator: {type(node.op).__name__}")
                return op(_eval(node.operand, depth + 1))
            raise _Invalid(f"Unsupported AST node: {type(node).__name__}")

        try:
            # [C+] ^ → ** vì trong Python AST, ^ là XOR, không phải luỹ thừa
            expr_parsed = expr.strip().replace("^", "**")
            tree = ast.parse(expr_parsed, mode="eval")

            # [C3] Strict math DSL whitelist – chặn ast.Call, ast.Name, ast.Attribute, v.v.
            _ALLOWED_NODES = (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                ast.USub, ast.UAdd,
            )
            for node in ast.walk(tree):
                if not isinstance(node, _ALLOWED_NODES):
                    raise DracoEngineV1._CalcInvalidError(
                        f"Unsafe AST node blocked: {type(node).__name__}"
                    )

            return _eval(tree.body)

        except DracoEngineV1._CalcError:
            raise  # re-raise phân loại đã rõ → caller xử lý
        except Exception as e:
            raise DracoEngineV1._CalcInvalidError(str(e)) from e

    # ══════════════════════════════════════════════════════════════════
    # ╔══════════════════════════════════════════════════════════════╗
    # ║              SKILL SANDBOX  –  KHU VỰC SKILLS               ║
    # ║  • Mỗi skill là hàm độc lập, dễ thêm / xóa / sửa           ║
    # ║  • Đăng ký trigger + tên skill tại cuối _register_skills()  ║
    # ╚══════════════════════════════════════════════════════════════╝
    # ══════════════════════════════════════════════════════════════════
    def _register_skills(self):

        # ── Skill: tính toán ──────────────────────────────────────────
        def skill_calc(text, *_):
            """[C] Regex chỉ match chuỗi số-operator-số thực sự hợp lệ.
            [fix-p7-1] Dùng exception phân loại từ _safe_calc thay vì check None.
            """
            m = re.search(
                r"(\d+(?:\.\d+)?(?:\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?)+)",
                text,
            )
            if not m:
                return None
            expr = m.group(1).strip()
            if not expr or not re.search(r"\d", expr):
                return None
            try:
                r = self._safe_calc(expr)
                return f"Kết quả: {r:g}"
            except DracoEngineV1._CalcRuntimeError:
                return "❌ Lỗi tính toán: chia cho 0."
            except DracoEngineV1._CalcInvalidError:
                return "❌ Biểu thức không hợp lệ."

        # ── Skill: ghi nhớ ────────────────────────────────────────────
        def skill_remember(text, *_):
            """[I] re.IGNORECASE – giữ casing gốc của key và value."""
            m = re.search(
                r"nhớ\s+(?:rằng\s+)?(.+?)\s+(?:là|=|is)\s+(.+)",
                text,
                re.IGNORECASE,
            )
            if m:
                k = m.group(1).strip()
                v = m.group(2).strip()
                self.ltm.remember_fact(k, v)
                return f"✅ Đã ghi nhớ: **{k}** = {v}"
            return None

        # ── Skill: nhớ lại ────────────────────────────────────────────
        def skill_recall(text, *_):
            """[J] Word-boundary regex – tránh 'AI' match trong 'chair'."""
            facts = self.ltm.get_facts()
            for f in facts:
                pattern = r"\b" + re.escape(f["key"].lower()) + r"\b"
                if re.search(pattern, text.lower()):
                    return f"💾 **{f['key']}**: {f['value']}"
            results = self.ltm.search(text, top_k=2)
            if results and results[0]["score"] > 0.5:
                return f"💾 [Memory] {results[0]['text']}"
            return None

        # ── Skill: đọc file ───────────────────────────────────────────
        def skill_read_file(text, *_):
            """[H][K][Q] Đọc file, giới hạn chặt trong ai_workspace/."""
            m = re.search(
                r"(?:đọc|mở|xem|read|open|load|hiện|show)\s+(?:file\s+)?[\"']?([^\s\"']+\.\w+)[\"']?",
                text, re.IGNORECASE,
            )
            if not m:
                return None

            raw_path = m.group(1).strip()
            target = (
                os.path.join(_SKILL_BASE_DIR, raw_path)
                if not os.path.isabs(raw_path)
                else raw_path
            )

            if not _is_safe_path(target):
                return "⛔ Không có quyền đọc file ngoài thư mục `ai_workspace/`."
            if not os.path.exists(target):
                # [H+][fix-base] target đã qua _is_safe_path → luôn trong sandbox, relpath an toàn
                safe_display = os.path.relpath(os.path.abspath(target), _SKILL_BASE_DIR)
                return f"❌ Không tìm thấy file: `{safe_display}`"

            try:
                ext  = os.path.splitext(target)[1].lower()
                size = os.path.getsize(target)

                text_exts = (
                    ".txt", ".md", ".log", ".py", ".json", ".csv",
                    ".yaml", ".yml", ".ini", ".toml", ".xml", ".html",
                    ".js", ".ts", ".java", ".c", ".cpp", ".h", ".rs",
                )
                if ext in text_exts:
                    with open(target, "r", encoding="utf-8", errors="replace") as fh:
                        content = fh.read(8192)  # Tối đa 8 KB – tránh OOM với log lớn
                    lines   = content.splitlines()
                    preview = "\n".join(lines[:50])
                    note    = f"\n... (còn {len(lines) - 50} dòng)" if len(lines) > 50 else ""
                    return (
                        f"📄 **File:** `{raw_path}` ({size:,} bytes)\n"
                        f"```{ext.lstrip('.')}\n{preview}{note}\n```"
                    )

                return (
                    f"📦 **File:** `{raw_path}` ({size:,} bytes)\n"
                    f"Định dạng `{ext}` — cần xử lý chuyên biệt "
                    f"(pdf/docx/xlsx/pptx). Bạn muốn tôi phân tích không?"
                )
            except Exception as e:
                return f"❌ Lỗi đọc file: {e}"

        # ── Skill: tạo file ───────────────────────────────────────────
        def skill_create_file(text, *_):
            """[H][K][R] Tạo/ghi file trong ai_workspace/. os.sep chặt bypass "../"."""
            m = re.search(
                r"(?:tạo|viết|ghi|lưu|create|write|save|xuất)\s+"
                r"(?:file\s+|ra\s+)?[\"']?([^\s\"']+\.\w+)[\"']?"
                r"(?:\s+với\s+nội\s+dung\s+|\s+content[:\s]+|\s+chứa\s+)(.+?)\s*$",
                text, re.IGNORECASE | re.DOTALL,
            )
            if not m:
                return None

            raw_path    = m.group(1).strip()
            content_raw = m.group(2).strip()

            # [R] Luôn resolve về sandbox rồi mới check – không trust raw_path
            target = os.path.abspath(os.path.join(_SKILL_BASE_DIR, raw_path))
            if not _is_safe_path(target):
                return "⛔ Không thể ghi file ngoài thư mục `ai_workspace/`."

            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                # [fix-p8-4][fix-race] Auto-backup: nếu file đã tồn tại, đổi tên thành .bak trước khi ghi đè
                # Dùng try/except FileNotFoundError để chịu race condition:
                # nếu thread khác xóa file giữa check và replace → bỏ qua, tiếp tục ghi
                if os.path.exists(target):
                    bak_path = target + ".bak"
                    try:
                        os.replace(target, bak_path)
                        _log.info("skill_create_file: backed up existing file to %s", bak_path)
                    except FileNotFoundError:
                        pass  # file đã bị xóa/di chuyển bởi thread khác – không cần backup
                    except Exception as bak_err:
                        _log.warning("skill_create_file: backup failed (proceeding): %s", bak_err)
                with open(target, "w", encoding="utf-8") as fh:
                    fh.write(content_raw)
                safe_rel = os.path.relpath(target, _SKILL_BASE_DIR)
                return (
                    f"✅ Đã tạo file: `ai_workspace/{safe_rel}` "
                    f"({len(content_raw.encode())} bytes, "
                    f"{len(content_raw.splitlines())} dòng)"
                )
            except Exception as e:
                return f"❌ Lỗi tạo file: {e}"

        # ── Skill: viết báo cáo ───────────────────────────────────────
        def skill_write_report(text, *_):
            """[H] Tạo template báo cáo Markdown."""
            keywords_vi = ["viết báo cáo", "tạo báo cáo", "soạn báo cáo",
                           "báo cáo về", "lập báo cáo"]
            keywords_en = ["write report", "create report", "generate report",
                           "draft report", "report on"]
            tl = text.lower()
            if not any(k in tl for k in keywords_vi + keywords_en):
                return None

            topic = re.sub(
                r"(viết|tạo|soạn|lập|write|create|generate|draft)\s+"
                r"(báo cáo|report)\s*(về|on|about)?\s*",
                "", tl, flags=re.IGNORECASE,
            ).strip().title() or "Chủ đề chưa xác định"

            template = (
                f"# Báo cáo: {topic}\n\n"
                "## 1. Tóm tắt điều hành (Executive Summary)\n"
                "> Mô tả ngắn gọn mục tiêu, phương pháp và kết quả chính.\n\n"
                "## 2. Giới thiệu\n"
                "- Bối cảnh và lý do thực hiện\n"
                "- Phạm vi và giới hạn của báo cáo\n\n"
                "## 3. Phương pháp\n"
                "- Công cụ / nguồn dữ liệu sử dụng\n"
                "- Quy trình thu thập và xử lý\n\n"
                "## 4. Kết quả & Phân tích\n"
                "| Chỉ số | Giá trị | Ghi chú |\n"
                "|--------|---------|----------|\n"
                "| ...    | ...     | ...      |\n\n"
                "## 5. Đánh giá & Thảo luận\n"
                "- Điểm mạnh\n- Hạn chế\n- So sánh với kỳ vọng / baseline\n\n"
                "## 6. Kết luận & Kiến nghị\n"
                "- Kết luận chính\n- Đề xuất hành động tiếp theo\n\n"
                "## 7. Tài liệu tham khảo\n- [1] ...\n\n"
                f"---\n*Báo cáo được tạo bởi DracoAI V1 — {time.strftime('%Y-%m-%d %H:%M')}*\n"
            )
            return f"📝 **Template báo cáo về: {topic}**\n\n{template}"

        # ── Skill: lưu code từ khung markdown vào file ───────────────
        def skill_save_code(text, *_):
            """[new] Phát hiện khung code markdown trong text và lưu vào file.

            Trigger khi user nói "lưu code này", "save code", "xuất code vào file", v.v.
            Tìm khối ```<lang>\\n<code>\\n``` và ghi vào ai_workspace/<tên_auto>.<ext>.

            Map ngôn ngữ → extension:
                python/py → .py | javascript/js/ts → .js/.ts | c++ → .cpp
                java → .java | rust/rs → .rs | go → .go | html → .html
                css → .css | json → .json | yaml/yml → .yaml | sh/bash → .sh
                text/txt/unknown → .txt
            """
            _LANG_EXT = {
                "python": ".py", "py": ".py",
                "javascript": ".js", "js": ".js",
                "typescript": ".ts", "ts": ".ts",
                "java": ".java",
                "c++": ".cpp", "cpp": ".cpp", "c": ".c",
                "rust": ".rs", "rs": ".rs",
                "go": ".go",
                "html": ".html",
                "css": ".css",
                "json": ".json",
                "yaml": ".yaml", "yml": ".yaml",
                "bash": ".sh", "sh": ".sh",
                "sql": ".sql",
                "markdown": ".md", "md": ".md",
            }

            # Tìm tất cả khung code markdown
            blocks = re.findall(
                r"```([A-Za-z+#]*)\s*\n([\s\S]+?)```",
                text,
                re.MULTILINE,
            )
            if not blocks:
                return "⚠️ Không tìm thấy khung code markdown (```lang\\n...```) nào trong text."

            saved = []
            timestamp = time.strftime("%H%M%S")
            for idx, (lang, code) in enumerate(blocks):
                lang_clean = lang.strip().lower()
                ext        = _LANG_EXT.get(lang_clean, ".txt")
                fname      = f"code_{timestamp}_{idx + 1}{ext}"
                target     = os.path.abspath(os.path.join(_SKILL_BASE_DIR, fname))

                if not _is_safe_path(target):
                    saved.append(f"⛔ Bị chặn: `{fname}`")
                    continue
                try:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with open(target, "w", encoding="utf-8") as fh:
                        fh.write(code.rstrip() + "\n")
                    size = len(code.encode())
                    saved.append(
                        f"✅ `ai_workspace/{fname}` ({size:,} bytes, "
                        f"{len(code.splitlines())} dòng)"
                    )
                except Exception as e:
                    saved.append(f"❌ Lỗi lưu `{fname}`: {e}")

            header = f"💾 Đã lưu {len(saved)} file:\n"
            return header + "\n".join(saved)

        # (skill_write_code đã được thay bằng _detect_code_request() – xem bên dưới)

        # ══════════════════════════════════════════════════════════════
        # Đăng ký skills + triggers + priority
        # [fix-p7-2] Priority system: calc (100) short-circuit ngay sau khi chạy.
        #            write_code (0) không đăng ký vào _skills – xử lý bằng flag
        #            _code_format_hint trong chat()/stream_chat() để không phá hội thoại.
        # ══════════════════════════════════════════════════════════════
        # Format: name → (fn, triggers, priority)
        #   priority cao → chạy trước; calc short-circuit sau khi chạy xong
        #   write_code KHÔNG vào đây – xem _detect_code_request()
        self._skills: Dict[str, Tuple[Any, List[str], int]] = {
            "calc": (skill_calc, [
                "tính", "calculate", "compute",
            ], 100),
            "remember": (skill_remember, [
                "nhớ rằng", "ghi nhớ", "lưu lại", "remember that",
            ], 50),
            "recall": (skill_recall, [
                "bạn có nhớ", "nhớ không", "bạn biết", "what is", "recall",
            ], 40),
            "read_file": (skill_read_file, [
                "đọc file", "mở file", "xem file",
                "read file", "open file", "load file",
            ], 60),
            "create_file": (skill_create_file, [
                "tạo file", "ghi file", "lưu file",
                "create file", "write file", "save file", "xuất file",
            ], 60),
            "write_report": (skill_write_report, [
                "viết báo cáo", "tạo báo cáo", "soạn báo cáo",
                "lập báo cáo", "write report", "create report",
                "draft report", "báo cáo về",
            ], 55),
            "save_code": (skill_save_code, [
                "lưu code", "save code", "xuất code", "lưu đoạn code",
                "lưu code này", "save this code", "export code",
                "ghi code vào file", "write code to file",
            ], 70),
        }

        # [fix-p7-2] write_code: detect riêng bằng hàm này, KHÔNG phải skill thông thường
        # Trả về (detected: bool, lang: str) để chat()/stream_chat() inject vào system prompt
        self._write_code_triggers = [
            "viết code", "viết hàm", "viết class", "viết script",
            "viết chương trình", "code cho tôi", "tạo hàm", "tạo class",
            "implement", "lập trình", "ví dụ code",
            "write code", "write a function", "write a class", "write a script",
            "write a program", "code for me", "create a function",
            "create a class", "give me code", "show me code",
            "code example", "example code",
        ]
        self._write_code_lang_map = {
            "python": "python", "py": "python",
            "javascript": "javascript", "js": "javascript",
            "typescript": "typescript", "ts": "typescript",
            "java": "java", "c++": "cpp", "cpp": "cpp",
            "rust": "rust", "go": "go", "html": "html",
            "css": "css", "sql": "sql", "bash": "bash", "shell": "bash",
        }

    # ══════════════════════════════════════════════════════════════════
    # ╚══════════════ KẾT THÚC SKILL SANDBOX ═══════════════╝
    # ══════════════════════════════════════════════════════════════════

    # [fix-p7-2 + doc-fix] Phân loại skill theo tính chất:
    # DETERMINISTIC: trả kết quả xác định, ưu tiên output, không side-effect
    # SIDE_EFFECT: ghi nhớ, ghi file, có tác dụng phụ → vẫn chạy nhưng output optional
    _DETERMINISTIC_SKILLS = frozenset({"calc", "recall", "read_file", "write_report"})
    _SIDE_EFFECT_SKILLS   = frozenset({"remember", "create_file", "save_code"})

    def _try_skills(self, text: str) -> Optional[str]:
        """[P][fix-p7-2] Priority-based multi-dispatch với DETERMINISTIC/SIDE_EFFECT hierarchy.

        Quy tắc:
        - DETERMINISTIC skill (calc, recall, ...): ưu tiên output; calc short-circuit ngay.
        - SIDE_EFFECT skill (remember, create_file, ...): luôn chạy nhưng output phụ.
        - Nếu có deterministic → output nó; side-effect append phía sau nếu có kết quả.
        - Nếu chỉ có side-effect → join và trả về.
        """
        tl = text.lower()
        triggered: List[Tuple[int, str, Any]] = []  # (priority, name, fn)

        for name, (fn, triggers, priority) in self._skills.items():
            if name == "calc":
                has_keyword = any(t in tl for t in triggers)
                has_expr    = bool(re.search(
                    r"\d+(?:\.\d+)?(?:\s*[+\-*/^]\s*\d+(?:\.\d+)?)+", text
                ))
                if not (has_keyword or has_expr):
                    continue
            else:
                if not any(t in tl for t in triggers):
                    continue
            triggered.append((priority, name, fn))

        if not triggered:
            return None

        # Sort priority cao → thấp
        triggered.sort(key=lambda x: x[0], reverse=True)

        deterministic_results: List[str] = []
        side_effect_results:   List[str] = []

        for _, name, fn in triggered:
            try:
                r = fn(text, {}, {})
                if not r:
                    continue
                if name in self._DETERMINISTIC_SKILLS:
                    deterministic_results.append(r)
                else:
                    side_effect_results.append(r)
            except Exception as e:
                _log.warning("Skill '%s' failed: %s", name, e)

        # [fix-p2] Join TẤT CẢ deterministic results – không bỏ sót kết quả từ skill thứ 2 trở đi
        # Ví dụ: recall + read_file cùng trigger → cả 2 output đều hiển thị cho user.
        # [fix-p10-3] calc short-circuit output nhưng KHÔNG ngăn side-effect chạy:
        # Loop đã chạy hết → side-effect đã được collect đầy đủ.
        if deterministic_results:
            base = "\n".join(deterministic_results)
            if side_effect_results:
                return base + "\n" + "\n".join(side_effect_results)
            return base

        if side_effect_results:
            return "\n".join(side_effect_results)

        return None

    def _detect_code_request(self, text: str) -> Tuple[bool, str]:
        """[fix-p7-2] Detect yêu cầu viết code và ngôn ngữ lập trình.
        Trả về (is_code_request, detected_lang).
        Dùng bởi chat() và stream_chat() để inject system hint vào prompt
        mà KHÔNG phá hội thoại (không return trực tiếp cho user).
        """
        tl = text.lower()
        if not any(k in tl for k in self._write_code_triggers):
            return False, "python"

        detected_lang = "python"
        for kw, lang in self._write_code_lang_map.items():
            if kw in tl:
                detected_lang = lang
                break
        return True, detected_lang

    # ══════════════════════════════════════════════════════════════════
    # [Hybrid] Tool-call helpers
    # ══════════════════════════════════════════════════════════════════
    def _parse_tool_call(self, text: str) -> Optional[Tuple[str, str]]:
        """Trích xuất (action, input) từ cú pháp [ACTION: input] mà LLM sinh ra.
        Chỉ chấp nhận action nằm trong ALLOWED_TOOLS_FOR_LLM – không để LLM
        tự gọi remember/create_file/save_code.
        """
        m = _TOOL_CALL_PATTERN.search(text)
        if m:
            action = m.group(1).strip().lower()
            inp    = m.group(2).strip()
            if action in ALLOWED_TOOLS_FOR_LLM:
                return action, inp
        return None

    # Map action → tiền tố cần ghép để regex của skill khớp khi LLM truyền tên file/từ khoá thuần
    _TOOL_INPUT_PREFIX: Dict[str, str] = {
        "read_file": "đọc file ",   # skill_read_file regex cần "đọc|mở|xem|read|open|load ..."
        "recall":    "bạn có nhớ ", # skill_recall trigger: "bạn có nhớ", "recall", v.v.
    }

    def _execute_tool(self, action: str, inp: str) -> str:
        """Chạy skill được LLM yêu cầu và trả kết quả dạng chuỗi.

        Một số skill (read_file, recall) dùng regex bắt câu lệnh đầy đủ.
        Khi LLM gọi [READ_FILE: report.txt] chỉ truyền tên file thuần →
        cần ghép tiền tố tương ứng từ _TOOL_INPUT_PREFIX để regex khớp.

        Chỉ gọi skill trong ALLOWED_TOOLS_FOR_LLM – không trigger side-effect.
        Nếu skill trả None hoặc raise Exception → trả thông báo rõ ràng.
        """
        entry = self._skills.get(action)
        if entry is None:
            return f"[{action}] không tìm thấy skill."
        fn, _, _ = entry
        # Ghép tiền tố nếu cần để regex của skill khớp
        prefix = self._TOOL_INPUT_PREFIX.get(action, "")
        effective_inp = f"{prefix}{inp}" if prefix else inp
        try:
            r = fn(effective_inp, {}, {})
            return str(r) if r is not None else f"[{action}] không trả về kết quả."
        except Exception as e:
            return f"Lỗi khi chạy {action}: {e}"

    # ══════════════════════════════════════════════════════════════════
    # [E][N] Context trimming helper
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _trim_history(
        history: List[dict],
        max_turns: int = MAX_HISTORY_TURNS,
    ) -> List[dict]:
        """Giữ tối đa `max_turns` lượt cuối (1 lượt = user + assistant).
        [E] for-loop thay while – tránh xoá mù khi nhiều assistant liên tiếp.
        [N] Fallback an toàn nếu sau trim không còn gì.
        """
        if len(history) <= max_turns * 2:
            return history

        trimmed = history[-(max_turns * 2):]

        # [E] Tìm index user đầu tiên bằng for-loop
        for i, msg in enumerate(trimmed):
            if msg.get("role") == "user":
                trimmed = trimmed[i:]
                break
        else:
            # [N][fix-5][fix-p7-7] fallback cải tiến:
            # tìm user gần nhất từ cuối + giữ thêm 1 message trước đó để context
            for i in range(len(history) - 1, -1, -1):
                if history[i].get("role") == "user":
                    return history[max(0, i - 1):]
            # Không có user nào → giữ 3 message cuối (nhiều hơn 2 để context phong phú hơn)
            return history[-3:] if len(history) >= 3 else history

        return trimmed

    # ══════════════════════════════════════════════════════════════════
    # [F] LTM store với duplicate check
    # ══════════════════════════════════════════════════════════════════
    def _safe_ltm_store(self, text: str, meta: dict):
        """[F] Lưu vào LTM chỉ khi chưa có nội dung trùng.
        Normalize text: NFKC unicode + bỏ punctuation + chuẩn hóa whitespace.
        [fix-p10-5] Thứ tự đúng: hash check TRƯỚC (rẻ) → semantic dedup SAU (tốn IO).
        Tránh gọi ltm.search tốn kém khi hash đã trùng (câu hỏi lặp y hệt).
        """
        # [F+] NFKC normalize trước – chuẩn hóa các dạng unicode khác nhau của cùng ký tự
        normalized = unicodedata.normalize("NFKC", text.strip().lower())
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        # [F2] blake2b thay MD5 – nhanh hơn và collision-resistant hơn ở scale lớn
        h = hashlib.blake2b(normalized.encode(), digest_size=16).hexdigest()
        if not self._register_query_hash(h):
            return  # hash trùng → chắc chắn đã lưu, skip luôn không cần search

        # [fix-p7-4] Semantic dedup nhẹ – chặn "AI là gì" vs "AI là cái gì"
        # Chỉ chạy khi hash mới (câu hỏi khác chữ nhưng có thể giống nghĩa)
        try:
            results = self.ltm.search(text, top_k=1)
            if results and results[0]["score"] > 0.92:
                return
        except Exception:
            pass  # ltm chưa có gì hoặc search fail → tiếp tục lưu
        self.ltm.store(text, meta)

    # ══════════════════════════════════════════════════════════════════
    # Chat
    # ══════════════════════════════════════════════════════════════════
    def chat(
        self,
        user_input: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.05,
        max_tokens: int = 256,
        think_mode: bool = False,
    ) -> dict:
        t0 = time.time()

        # 1. Skill shortcut
        skill_reply = self._try_skills(user_input)
        if skill_reply:
            self.working.add("user", user_input)
            self.working.add("assistant", skill_reply)
            return {
                "reply":          skill_reply,
                "intent":         {"intent": "skill"},
                "thought_plan":   {},
                "skill":          "builtin",
                "time_ms":        int((time.time() - t0) * 1000),
                "critique":       {},
                "confidence_avg": 1.0,
            }

        # [fix-p8-2] Inject code format hint vào prompt khi user yêu cầu viết code.
        # Augment user_input_for_engine (bản nội bộ) chứ KHÔNG thay đổi user_input gốc
        # để memory/ltm vẫn lưu câu hỏi gốc của người dùng.
        is_code_req, code_lang = self._detect_code_request(user_input)
        _hint_block = f"```{code_lang}\n...\n```"
        user_input_for_engine = (
            f"[SYS_HINT: Reply with code inside a {_hint_block} block. "
            f"Add comments explaining each step.]\n{user_input}"
            if is_code_req else user_input
        )

        # 2. Prepare memory
        intent    = self.thinking.detector.detect(user_input)
        mem_input = self.ltm.prepare_engine_input(user_input, intent, top_k=3)

        # [E] Trim history
        raw_history = self.working.get_messages()
        history     = self._trim_history(raw_history, max_turns=MAX_HISTORY_TURNS)

        # 3. Engine processing
        engine_out = self.thinking.process(
            user_input_for_engine,          # [fix-p8-2] dùng bản augmented cho Transformer
            history=history,
            memory_summary=mem_input["memory_summary"],
            ltm_facts=mem_input["ltm_facts"],
            memory_candidates=mem_input["memory_candidates"],
            think_mode=think_mode,
        )

        # 4. Tokenize
        prompt_ids = self.thinking.tokenize_prompt(engine_out["messages"])

        # 5. Build generation kwargs
        expert_boost    = engine_out.get("expert_boost", {})
        intent_bias_arr = self.bridge.expert_boost_to_array(expert_boost)

        # [B] Guard None + np.clip (nhanh hơn tanh, vẫn bound giá trị)
        if intent_bias_arr is not None:
            try:
                intent_bias_arr = np.clip(intent_bias_arr, -3.0, 3.0)
            except Exception as e:
                _log.warning("intent_bias clip failed, degrading to zeros: %s", e)
                # [fix-p7-6] zeros_like: giữ shape nhưng neutral bias thay vì mất signal
                try:
                    intent_bias_arr = np.zeros_like(intent_bias_arr)
                except Exception:
                    intent_bias_arr = None

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temp":           temperature,
            "top_p":          top_p,
            "min_p":          min_p,
            # [O+] Mirostat: prompt dài HOẶC temperature cao HOẶC intent creative
            "use_mirostat":   (
                len(prompt_ids) > 50
                or temperature > 0.9
                or intent.get("intent") == "creative"
            ),
            "intent_bias":    intent_bias_arr,
        }

        # 6. Hybrid reasoning loop – [L] LLM được phép gọi tool tối đa MAX_TOOL_ROUNDS lần
        # Cú pháp: [CALC: ...] / [RECALL: ...] / [READ_FILE: ...]
        # Kết quả tool được inject lại vào conversation (role "tool") trước khi generate tiếp.
        # Nếu không có tool call → lấy ngay làm final reply.
        # stream_chat() vẫn dùng pipeline cũ (buffered stream, không hỗ trợ tool loop).
        conversation = list(engine_out["messages"])

        # Gắn system hint hướng dẫn tool nếu chưa có trong conversation
        _tool_syntax_marker = "công cụ bằng cú pháp"
        if not any(_tool_syntax_marker in m.get("content", "") for m in conversation):
            tool_hint = (
                "Bạn có thể gọi công cụ bằng cú pháp:\n"
                "  [CALC: biểu thức]       — tính toán số học\n"
                "  [RECALL: từ khóa]       — nhớ lại kiến thức đã lưu\n"
                "  [READ_FILE: tên_file]   — đọc file trong ai_workspace/\n"
                "Khi cần dùng tool, hãy chỉ xuất ra đúng một cú pháp đó trên một dòng riêng, "
                "không viết thêm văn bản nào khác trong cùng lượt đó. "
                "Sau khi nhận kết quả tool, mới tiếp tục trả lời bình thường."
            )
            conversation.append({"role": "system", "content": tool_hint})

        # Snapshot conversation gốc (trước khi tool loop bắt đầu).
        # Chiến lược trim an toàn: chỉ trim phần base (history + system hints),
        # KHÔNG bao giờ cắt tool_results_this_turn (LLM cần thấy kết quả tool
        # của vòng hiện tại để tiếp tục suy nghĩ; cắt mất → logic sai hoàn toàn).
        base_conversation = list(conversation)

        rounds = 0
        tool_results_this_turn: List[dict] = []  # tool results tích lũy trong lượt này
        final_reply = ""

        while rounds < MAX_TOOL_ROUNDS:
            # Trim history gốc để tránh context overflow qua nhiều tool rounds.
            # Tool results luôn được ghép NGUYÊN VẸN sau base đã trim.
            trimmed_base = self._trim_history(base_conversation, max_turns=MAX_HISTORY_TURNS)
            current_conversation = trimmed_base + tool_results_this_turn

            current_prompt_ids = self.thinking.tokenize_prompt(current_conversation)
            try:
                new_tokens = self.bridge.generate(current_prompt_ids, **gen_kwargs)
            except Exception as e:
                _log.error("Generate failed (round %d): %s", rounds, e)
                return {
                    "reply":          f"❌ Lỗi sinh văn bản: {e}",
                    "intent":         intent,
                    "thought_plan":   {},
                    "skill":          "transformer",
                    "time_ms":        int((time.time() - t0) * 1000),
                    "critique":       {},
                    "confidence_avg": 0.0,
                    "error_code":     "GENERATE_FAILED",
                    "retry_hint":     "Thử lại sau vài giây hoặc giảm max_tokens.",
                }

            raw_output = self.tokenizer.decode(new_tokens).strip()
            tool_call  = self._parse_tool_call(raw_output)

            if tool_call:
                action, inp = tool_call
                tool_result = self._execute_tool(action, inp)
                _log.debug("Tool call [%s: %s] → %s", action, inp, tool_result[:80])
                tool_results_this_turn.append({
                    "role":    "tool",
                    "content": f"[RESULT:{action}] {tool_result}",
                })
                rounds += 1
                continue  # LLM suy nghĩ tiếp với kết quả tool
            else:
                final_reply = raw_output
                break

        # Nếu vẫn chưa có reply sau MAX_TOOL_ROUNDS (tool gọi liên tục không kết thúc)
        if not final_reply:
            final_reply = "Tôi cần thêm thời gian suy nghĩ về câu hỏi này..."

        reply    = final_reply
        avg_conf = engine_out.get("calibrated_confidence", 0.8)

        # 8. Update memory
        self.working.add("user", user_input)
        self.working.add("assistant", reply)
        self._safe_ltm_store(user_input, {"type": "user_query", "intent": intent["intent"]})
        self._auto_learn(reply, avg_conf=avg_conf)

        return {
            "reply":           reply,
            "intent":          intent,
            "thought_plan":    engine_out["thought_plan"] if think_mode else {},
            "skill":           "transformer",
            "time_ms":         int((time.time() - t0) * 1000),
            "critique":        {},
            "confidence_avg":  round(avg_conf, 3),
            "rewritten_query": engine_out.get("rewritten_query", user_input),
            "process_mode":    engine_out.get("process_mode", "fast"),
            "debate":          engine_out["thought_plan"].get("debate_synthesis", ""),
        }

    # ══════════════════════════════════════════════════════════════════
    # Streaming Chat
    # ══════════════════════════════════════════════════════════════════
    def stream_chat(
        self,
        messages: List[dict],
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_tokens: int = 256,
        think_mode: bool = False,
    ) -> Generator[Tuple[str, float], None, None]:
        """Sinh văn bản và yield từng token sau khi generation hoàn thành.

        ⚠️  Lưu ý kiến trúc (fix-6):
            Phương thức này KHÔNG stream từng token theo thời gian thực.
            bridge.generate() chạy đồng bộ; stream_cb chỉ gom token vào buffer.
            Toàn bộ token được yield SAU khi generation kết thúc.
            Để streaming thực sự: cần bridge.generate() hỗ trợ yield-per-token
            (queue + thread) hoặc dùng async generator ở tầng bridge.
            Hiện tại đây là "buffered stream" – phù hợp cho CLI/testing.

        [fix-p7-5] Thêm skill shortcut: nếu _try_skills trả kết quả,
            yield ngay và return – nhất quán với chat().
        """
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # [fix-p7-5] Skill shortcut trong stream_chat – nhất quán với chat()
        skill_reply = self._try_skills(user_msg)
        if skill_reply:
            self.working.add("user", user_msg)
            self.working.add("assistant", skill_reply)
            yield skill_reply, 1.0
            return

        # [fix-p8-2] Inject code format hint – giống chat(), không thay user_msg gốc
        is_code_req, code_lang = self._detect_code_request(user_msg)
        _hint_block_s = f"```{code_lang}\n...\n```"
        user_msg_for_engine = (
            f"[SYS_HINT: Reply with code inside a {_hint_block_s} block. "
            f"Add comments explaining each step.]\n{user_msg}"
            if is_code_req else user_msg
        )

        intent    = self.thinking.detector.detect(user_msg)
        mem_input = self.ltm.prepare_engine_input(user_msg, intent, top_k=3)

        # [E] Trim history
        history_raw = [m for m in messages[:-1]]
        history     = self._trim_history(history_raw, max_turns=MAX_HISTORY_TURNS)

        engine_out = self.thinking.process(
            user_msg_for_engine,            # [fix-p8-2] dùng bản augmented cho Transformer
            history=history,
            memory_summary=mem_input["memory_summary"],
            ltm_facts=mem_input["ltm_facts"],
            memory_candidates=mem_input["memory_candidates"],
            think_mode=think_mode,
        )

        prompt_ids = self.thinking.tokenize_prompt(engine_out["messages"])

        expert_boost    = engine_out.get("expert_boost", {})
        intent_bias_arr = self.bridge.expert_boost_to_array(expert_boost)

        # [B] Guard None + clip
        if intent_bias_arr is not None:
            try:
                intent_bias_arr = np.clip(intent_bias_arr, -3.0, 3.0)
            except Exception as e:
                _log.warning("intent_bias clip failed (stream), degrading to zeros: %s", e)
                # [fix-p7-6] zeros_like thay None – giữ routing signal nhẹ thay vì mất hoàn toàn
                try:
                    intent_bias_arr = np.zeros_like(intent_bias_arr)
                except Exception:
                    intent_bias_arr = None

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temp":           temperature,
            "top_p":          top_p,
            "min_p":          0.05,
            # [O+] Mirostat: prompt dài HOẶC temperature cao HOẶC intent creative
            "use_mirostat":   (
                len(prompt_ids) > 50
                or temperature > 0.9
                or intent.get("intent") == "creative"
            ),
            "intent_bias":    intent_bias_arr,
        }

        pieces:           List[Tuple[str, float]] = []
        full_reply_parts: List[str]               = []
        _generate_completed: List[bool]           = [False]  # flag dùng closure

        def stream_cb(token_id: int, confidence: float):
            token_text = self.tokenizer.decode_token(token_id)
            if token_text:
                pieces.append((token_text, confidence))
                full_reply_parts.append(token_text)

        gen_kwargs["stream_cb"] = stream_cb

        # [L] try/except cho stream generate
        try:
            self.bridge.generate(prompt_ids, **gen_kwargs)
            _generate_completed[0] = True  # [stream2] đánh dấu generate xong hoàn toàn
        except Exception as e:
            _log.error("Stream generate failed: %s", e)
            yield f"❌ Lỗi sinh văn bản: {e}", 0.0
            return

        # Yield tokens – client nhận output
        yield_done = False
        for token_text, conf in pieces:
            yield token_text, conf
        yield_done = True

        # [stream-fix][stream2][doc-fix] Update memory SAU khi yield loop kết thúc hoàn toàn.
        # Cần cả 2 điều kiện:
        #   _generate_completed[0] = True  → bridge.generate() không exception
        #   yield_done = True              → toàn bộ token đã được deliver cho client
        # Tránh lưu khi generator bị cancel giữa chừng (client disconnect, GC drop).
        full_reply = "".join(full_reply_parts).strip()
        if full_reply and _generate_completed[0] and yield_done:
            self.working.add("user", user_msg)
            self.working.add("assistant", full_reply)
            self._safe_ltm_store(user_msg, {"type": "user_query", "intent": intent["intent"]})
            stream_conf = (
                sum(c for _, c in pieces) / len(pieces) if pieces else 0.8
            )
            self._auto_learn(full_reply, avg_conf=stream_conf)

    # ══════════════════════════════════════════════════════════════════
    # [D][G][S] Auto-learn
    # ══════════════════════════════════════════════════════════════════
    def _auto_learn(self, response: str, avg_conf: float = 1.0):
        """Tự học fact từ response.
        Guard: confidence threshold + hallucination flags + stopwords + bad adj/adv + context words.
        """
        # [G+] Không học khi confidence thấp – chặn hallucination leak vào LTM
        if avg_conf < 0.7:
            return

        resp_lower = response.lower()

        # [G] Không học khi model không chắc
        if any(flag in resp_lower for flag in HALLUCINATION_FLAGS):
            return

        # [D+] Subject phải bắt đầu bằng chữ hoa (danh từ riêng / thuật ngữ)
        # tránh match "Trời hôm nay AI là..." → subject = "Trời hôm nay AI"
        pattern = r"\b([A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9\s]{1,40})\s+(?:là|is|means|có nghĩa là)\s+([^\.]{5,150})"
        learned = 0  # [perf] giới hạn tối đa 3 fact / response – tránh CPU spike
        for m in re.finditer(pattern, response):
            subj = m.group(1).strip()
            desc = m.group(2).strip()
            subj_lower = subj.lower()

            # [fix-3] ALL CAPS dài → spam/không phải proper noun
            if subj.isupper() and len(subj) > 5:
                continue

            # [fix-p7-3][fix-p9-2] Proper noun check (Layer 2+3 gộp lại – không dư thừa):
            # Subject phải có ít nhất 1 từ dạng Hoa-đầu-thường (noun phrase thực sự).
            # Acronym kỹ thuật trong _LEARN_ALLOW_SINGLE được bypass.
            if len(re.findall(r"\b[A-ZÀ-Ỹ][a-zà-ỹ]+\b", subj)) < 1:
                if subj not in _LEARN_ALLOW_SINGLE:
                    continue

            # [fix-p7-3] Subject chứa động từ nối (Là/Is/Means) → đây là câu, không phải subject
            if re.search(r"\b(Là|Is|Means|Có)\b", subj):
                continue

            # [D] Bắt đầu bằng stopword → bỏ
            if any(subj_lower.startswith(sw) for sw in _LEARN_STOPWORDS):
                continue

            # [D] Chứa adj/adv phổ biến → bỏ
            if any(bp in subj_lower for bp in _LEARN_BAD_PATTERNS):
                continue

            # [S] Chứa từ ngữ cảnh/điều kiện ở mức token → bỏ
            subj_tokens = subj_lower.split()
            if any(cw in subj_tokens for cw in _LEARN_CONTEXT_WORDS):
                continue

            # Subject không quá dài (tránh học cả câu)
            if len(subj.split()) > 5:
                continue

            # [D+] Subject 1 từ không trong allow-list → bỏ
            if len(subj.split()) < 2 and subj not in _LEARN_ALLOW_SINGLE:
                continue

            if len(subj) < 40 and len(desc.split()) > 5 and "?" not in desc:
                # [D3] Tránh học opinion/subjective statement ("Python là rất mạnh")
                desc_lower = desc.lower()
                desc_tokens = desc_lower.split()
                if any(w in desc_tokens for w in _LEARN_BAD_PATTERNS):
                    continue
                # [fix-p7-3] desc bắt đầu bằng adj/adv → opinion, bỏ
                if desc_tokens and desc_tokens[0] in _LEARN_BAD_PATTERNS:
                    continue
                # [fix-p10-4a] Sentiment filter: tránh học biased statements
                # ("Python là một ngôn ngữ tệ hại và lỗi thời")
                if any(w in desc_lower for w in _LEARN_BAD_OPINION_WORDS):
                    continue
                # [fix-p10-4b] Desc diversity check: quá ít từ unique → generic/noise
                if len(set(desc_tokens)) < 4:
                    continue
                self.ltm.learn(subj, desc, source="auto_learn")
                learned += 1
                if learned >= 3:  # [perf] đủ 3 fact, dừng
                    break

    # ══════════════════════════════════════════════════════════════════
    # [T] Permission guard
    # ══════════════════════════════════════════════════════════════════
    # [T+] Callback handler – tích hợp với UI/async thực tế
    # Gán lại trước khi dùng trong môi trường web/GUI để tránh block:
    #   engine.permission_handler = lambda desc: my_ui_confirm(desc)
    # Mặc định dùng input() cho CLI; trả về False nếu không có terminal.
    permission_handler = None  # type: Optional[Any]  # callable(str) -> bool

    @classmethod
    def request_elevated_permission(
        cls,
        action_description: str,
        handler=None,
    ) -> bool:
        """[T] Hỏi người dùng trước khi thực hiện hành động cần quyền cao.
        Trả về True nếu đồng ý, False nếu từ chối.

        Ưu tiên: handler tham số → cls.permission_handler → input() CLI fallback.
        Trong môi trường web/async, gán cls.permission_handler = <UI callback>
        thay vì dùng input() vì input() sẽ block toàn bộ thread.
        """
        fn = handler or cls.permission_handler
        if fn is not None:
            try:
                return bool(fn(action_description))
            except Exception:
                return False

        # CLI fallback – chỉ dùng khi có terminal thực sự
        prompt = (
            f"\n⚠️  DracoAI muốn thực hiện: {action_description}\n"
            f"   Hành động này có thể cần quyền hạn cao hơn.\n"
            f"   Bạn có cho phép không? (yes/no): "
        )
        try:
            answer = input(prompt).strip().lower()
            return answer in ("yes", "y", "có", "co", "ok")
        except (EOFError, KeyboardInterrupt):
            return False

    # ══════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════
    def load_external_weights(self, state_dict: dict) -> dict:
        self.model.load_external_weights(state_dict, from_checkpoint=True)
        return {"status": "ok"}

    def end_session(self):
        self.working.end_session()

    def export_gguf(self, output_path: str = "dracoai_fp16.gguf"):
        GGUFExporter(self.model).write_gguf(output_path)
        self.bridge = TransformerBridge(gguf_path=output_path, n_gpu_layers=32)
        self.thinking.bridge = self.bridge