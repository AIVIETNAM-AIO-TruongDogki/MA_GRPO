# Kiến Trúc Hệ Thống MA-GRPO (Multi-Agent Group Relative Policy Optimization)

Tài liệu này mô tả chi tiết toàn bộ kiến trúc phần mềm, luồng dữ liệu, các tác nhân (Agents), và động cơ hàm thưởng nội tại (Intrinsic Reward Engine) của thư mục mã nguồn `src/` trong dự án **MA-GRPO**.

---

## 1. Tổng Quan Kiến Trúc (Executive Architecture Overview)

MA-GRPO giải quyết bài toán sinh văn bản tự sự siêu dài (>8.000 đến 15.000 từ) bằng cách **tách rời hoàn toàn khâu quy hoạch vĩ mô (Macro Planning) và khâu viết văn vi mô (Micro-Prose Execution)**:

```mermaid
flowchart TD
    subgraph STAGE1 ["STAGE 1: MACRO-NARRATIVE PLANNING (Inference-Only)"]
        UserPremise["User Premise & Length Constraint"] --> OutlineAgent["OutlineAgent<br/>(Event Graph G = V, E)"]
        OutlineAgent --> SubEventDecomposer["SubEventDecomposer<br/>(Narrative Beats Arrival/Confrontation/Turn)"]
        SubEventDecomposer --> PlanCritic["PlanCriticAgent<br/>(Dependency & Cycle Gate)"]
        PlanCritic --> PlanningAgent["PlanningAgent<br/>(Rolling K-Chapter Partition & Budgeting)"]
    end

    subgraph STAGE2 ["STAGE 2: LOCAL WRITING & REINFORCEMENT LEARNING"]
        PlanningAgent --> ChapterBlueprint["Chapter Goal (E_i) & Target Words (L_target)"]
        History["Past Generated Text C_{1...i-1}"] --> Coordinator["CoordinatorAgent (ReIO)<br/>(Compress Context to s_i < 4096 tokens)"]
        Coordinator --> ContextState["Two-Tier Context State (s_i)"]
        
        ContextState --> WritingAgent["Writing Agent Policy (pi_theta)<br/>(Qwen2.5-7B + QLoRA + Test-Time CoT)"]
        ChapterBlueprint --> WritingAgent
        
        WritingAgent --> PostGate["0-Token Hygiene Gates<br/>(Artifact Cleaner + Paragraph Dedup Gate)"]
        PostGate --> Rollouts["Clean Chapter / G Candidate Rollouts"]
    end

    subgraph REWARDS ["3-COMPONENT INTRINSIC REWARD ENGINE"]
        Rollouts --> R_Len["R_length: Rule-Based Word Compliance"]
        Rollouts --> R_Coh["R_coherence: Cosine Embedding + Anti-Context 3-Gram"]
        Rollouts --> R_Qual["R_quality: Fluency (PPL) + Entropy + Burstiness"]
        
        R_Len --> R_Total["R_total Composite Weighted Reward"]
        R_Coh --> R_Total
        R_Qual --> R_Total
        
        R_Total --> GRPO_Update["TRL GRPOTrainer: Group Relative Advantage & QLoRA Update"]
        GRPO_Update -.-> WritingAgent
    end
```

---

## 2. Chi Tiết Từng Module Trong `src/`

```text
src/
├── __init__.py          # Export các Agent chính và Reward Engine
├── planning/            # Stage 1: Package Lập dàn ý (mỗi Agent một module riêng)
│   ├── story_arc.py     # Story Arc Alignment 5 giai đoạn & Event Graph parser
│   ├── outline_agent.py # OutlineAgent
│   ├── sub_event_agent.py # SubEventAgent & DAG sanitization
│   ├── plan_critic.py   # PlanCriticAgent (5 cổng kiểm duyệt)
│   └── planning_agent.py# PlanningAgent (chia chương & ngân sách từ)
├── coordinator.py       # Stage 2: Tóm tắt ngữ cảnh ReIO, Cổng thẩm định V_s & cơ chế Fallback
├── rewards.py           # Động cơ 3 hàm thưởng nội tại (Length, Coherence, Quality)
├── pipeline.py          # Bộ điều phối End-to-End Orchestrator Pipeline & vLLM/Local Engine
└── dataset_generator.py # Sinh tập dữ liệu Pilot / GRPO Training
```

---

### 2.1. Package `src/planning/` (Macro Planning)

Module chịu trách nhiệm xây dựng cốt truyện toàn cục theo chuẩn lý thuyết kịch bản văn học (Narrative Theory) và phân bổ ngân sách từ vựng.

#### Các lớp chính:
1. **`OutlineAgent`**:
   - **Nhiệm vụ**: Chuyển đổi đề bài (`premise`) thành Đồ thị sự kiện (**Event Graph** $G = (V, E)$).
   - **Cấu trúc sự kiện**: Mỗi sự kiện bao gồm bộ 6 trường ngữ nghĩa: `Setting`, `Character`, `Action`, `Conflict`, `Plot Twist`, `Raw Text`.
   - **Định dạng chuẩn**: Sử dụng regex parser tương thích với chuẩn cấu trúc StoryWriter.
2. **`SubEventDecomposer`**:
   - **Nhiệm vụ**: Phân rã mỗi sự kiện vĩ mô thành các nhịp truyện chi tiết (*Narrative Beats*):
     - `Arrival`: Thiết lập bối cảnh và tình huống ban đầu.
     - `Confrontation`: Xung đột chính giữa các nhân vật.
     - `Turn`: Điểm ngoặt kịch tính phát triển sự kiện.
   - Gán `depends_on` để quản lý đồ thị có hướng không chu trình (DAG).
3. **`PlanCriticAgent`**:
   - Cổng kiểm duyệt đóng băng: Phát hiện sự kiện trùng lặp, sự kiện nhảy cóc chưa thỏa mãn phụ thuộc (`unmet dependencies`), hoặc kế hoạch thiếu ràng buộc số từ.
4. **`PlanningAgent`**:
   - Phân chia $G$ thành $K$ chương độc lập.
   - Tính toán ngân sách từ mục tiêu cho từng chương:
     $$L_{\text{target}} = \frac{L_{\text{total}}}{K}$$
   - Chia tiếp mỗi chương thành 3 cảnh (*Scene Breakdown & Word Budgets*): Cảnh 1 (~25%), Cảnh 2 (~45%), Cảnh 3 (~30%).

---

### 2.2. Module `src/coordinator.py` (Two-Tier Context Coordination)

Module giải quyết bài toán suy giảm ngữ cảnh (*Lost-in-the-Middle*) khi viết truyện dài qua nhiều chương mà không tốn chi phí validator bên ngoài (tầng Validator cũ đã bị loại bỏ hoàn toàn sau thực nghiệm ablation).

```mermaid
sequenceDiagram
    autonumber
    participant Hist as Past Chapters C_{1...i-1}
    participant Coord as CoordinatorAgent (ReIO)
    participant Writer as WritingAgent (pi_theta)
    participant Hygiene as Artifact Cleaner & Dedup Gate

    Hist->>Coord: Gửi đuôi chương gần nhất & lịch sử kế hoạch
    Coord->>Coord: Nén ngữ cảnh thành Two-Tier State s_i (< 4096 tokens)
    Coord->>Writer: Chuyển giao s_i + 180 từ cuối nguyên văn
    Writer->>Hygiene: Sinh văn bản chương C_i (<think> + prose)
    Hygiene->>Hist: Lọc sạch token đặc biệt & cắt đoạn lặp (>30 từ) trước khi lưu
```

#### Lớp chính:
- **`CoordinatorAgent` (Cơ chế ReIO - Rewrite Input and Output)**:
  - Thu gọn lịch sử các chương trước thành trạng thái 2 tầng $s_i$ (<4.000 tokens): `long_term_memory` (`major_events`, `character_states`, `unresolved_plot_threads`, `revealed_secrets`) và `short_term_memory` (`recent_chapter_summary`, `recent_tail_excerpt` 180 từ cuối nguyên văn).
  - Chỉ tự động rơi về `_structured_state_extraction` khi chạy offline hoặc LLM lỗi định dạng JSON.

---

### 2.3. Module `src/rewards.py` (Unified Intrinsic Reward Engine)

MA-GRPO loại bỏ hoàn toàn LLM Evaluator bên ngoài (như mô hình 72B) để tiết kiệm tài nguyên và độ trễ, thay thế bằng **hệ thống 3 hàm thưởng toán học nội tại (Intrinsic Rewards)**:

$$\mathbf{R_{\text{total}}(o_i^{(k)}) = w_1 \cdot R_{\text{length}} + w_2 \cdot R_{\text{coherence}} + w_3 \cdot R_{\text{quality}}}$$

#### 1. $R_{\text{length}}$ — Kiểm Soát Chiều Dài Từng Chương (Rule-Based O(1))
Loại bỏ hoàn toàn thẻ suy nghĩ `<think>...</think>` trước khi đếm từ thực tế:
$$R_{\text{length}}(o_i^{(k)}) = \max \left( 0, 1 - \frac{|\text{actual\_len}(o_i^{(k)}) - L_{\text{target}}|}{L_{\text{target}}} \right)$$
*Đặc tính*: Bằng cách ép phạt cục bộ ở cấp độ chương, mô hình không thể "ăn gian độ dài" (Reward Hacking) bằng cách lặp từ dông dài.

#### 2. $R_{\text{coherence}}$ — Mạch Chuyển Tiếp & Chống Lặp Ngữ Cảnh (Anti-Context Penalty)
$$R_{\text{coherence}}(o_i^{(k)}) = \text{CosineSim}\left(\text{Embed}(\text{End}(C_{i-1})), \text{Embed}(\text{Start}(o_i^{(k)}))\right) \times \frac{|\text{Uniq3Grams}(o_i^{(k)}) \setminus \text{3Grams}(s_i)|}{|\text{Total3Grams}(o_i^{(k)})|}$$
* **Chuyển tiếp mượt mà**: Sử dụng mô hình `BAAI/bge-small-en-v1.5` đo độ tương đồng giữa đoạn kết chương trước và mở đầu chương hiện tại.
* **Chống lặp ngữ cảnh**: Phạt nặng việc mô hình "chép lại" các cụm 3 từ có sẵn trong bản tóm tắt $s_i$, buộc mô hình dùng văn phong độc lập.

#### 3. $R_{\text{quality}}$ — Đo Lường Chất Lượng Văn Phong Nội Tại
$$\mathbf{R_{\text{quality}} = \alpha \cdot R_{\text{fluency}} + \beta \cdot R_{\text{richness}} + \gamma \cdot R_{\text{burstiness}}}$$
* **$R_{\text{fluency}}$ (Độ trôi chảy qua Perplexity)**: Sử dụng mô hình ngôn ngữ nền siêu nhẹ `Qwen2.5-0.5B`:
  $$R_{\text{fluency}} = \exp \left( - \frac{\text{PPL}(o_i^{(k)})}{\sigma_{\text{ref}}} \right)$$
* **$R_{\text{richness}}$ (Độ giàu từ vựng qua Shannon Entropy)**:
  $$R_{\text{richness}} = \frac{-\sum_{v \in V} P(v) \log_2 P(v)}{H_{\text{max}}}$$
* **$R_{\text{burstiness}}$ (Nhịp điệu câu văn)**: Đo độ biến thiên độ dài các câu thông qua hàm Sigmoid trên độ lệch chuẩn (StdDev).

---

### 2.4. Module `src/pipeline.py` (End-to-End Orchestrator Pipeline)

Kết nối toàn bộ quy trình sinh văn bản và tạo nhóm rollout cho thuật toán GRPO:
1. **`extract_cot_and_prose`**: Phân tách chuỗi suy nghĩ `<think>...</think>` (Test-Time Reasoning) và văn bản tự sự sạch.
2. **`VLLMWritingPolicy`**:
   - Tối ưu hóa cho môi trường GPU thông qua vLLM PagedAttention / Transformers SDPA.
   - Hỗ trợ hàm `generate_rollouts` sinh song song $G$ ứng viên cho mỗi chương phục vụ huấn luyện GRPO.
3. **`construct_prompt`**: Cung cấp cấu trúc chỉ dẫn chuẩn hóa, truyền bối cảnh, mục tiêu số từ và ngân sách từng cảnh vào prompt của Writing Agent.
4. **`EndToEndStoryPipeline`**: Quản lý vòng lặp qua $K$ chương, cập nhật trạng thái Coordinator sau mỗi chương thành công.

---

## 3. Quy Trình Huấn Luyện QLoRA GRPO (`scripts/train.py`)

Module huấn luyện tích hợp trực tiếp với thư viện `trl.GRPOTrainer` và `peft`:

```mermaid
graph LR
    subgraph INPUT ["Input Batch"]
        Prompt["Chapter Prompt (with target_words)"]
    end

    subgraph SAMPLING ["Parallel Rollout Sampling"]
        Prompt --> Policy["Trainable Policy pi_theta (Qwen2.5-7B 4-bit)"]
        Policy --> G1["Rollout 1"]
        Policy --> G2["Rollout 2"]
        Policy --> G3["Rollout 3"]
        Policy --> G4["Rollout 4"]
    end

    subgraph EVALUATION ["Lazy Intrinsic Evaluation"]
        G1 & G2 & G3 & G4 --> Engine["CompositeRewardEngine"]
        Engine --> R1["Score R_1"]
        Engine --> R2["Score R_2"]
        Engine --> R3["Score R_3"]
        Engine --> R4["Score R_4"]
    end

    subgraph ADVANTAGE ["GRPO Advantage & Optimization"]
        R1 & R2 & R3 & R4 --> NormAdv["Group Normalization: A_i = (R_i - mean) / std"]
        NormAdv --> Loss["GRPO Clipped Loss + KL Divergence"]
        Loss --> Backprop["Backward Pass & LoRA Gradients"]
        Backprop --> Update["Optimizer Step (AdamW 4-bit)"]
        Update --> Checkpoint["Auto-Save / Auto-Resume Checkpoint"]
    end
```

### Điểm kỹ thuật cốt tử:
* **Dynamic Target Extraction**: Trích xuất độ dài mục tiêu trực tiếp từ Prompt nhằm đảm bảo tính chính xác ngay cả khi TRL không truyền metadata qua kwargs.
* **Token Budget (2.200 tokens)**: Đảm bảo độ dài sinh bao gồm cả reasoning `<think>` lẫn văn bản hoàn chỉnh 1.400–1.600 từ mà không bị cắt cụt.
* **Gradient Accumulation**: Căn chỉnh `generation_batch_size` luôn chia hết cho `num_generations` ($G=4$) theo đúng chuẩn của TRL.
* **Auto-Resume**: Quét và tự động nạp checkpoint cao nhất từ Google Drive (`checkpoint-50`, `checkpoint-100`, ...) để tiếp tục huấn luyện liền mạch.

---

## 4. Tóm Tắt Giao Diện Lớp & Dữ Liệu (Class Interface Summary)

| Class | Tệp | Vai trò chính | Trạng thái huấn luyện |
| :--- | :--- | :--- | :--- |
| `OutlineAgent` | `planning.py` | Tạo Event Graph 6 trường ngữ nghĩa | Đông cứng (Inference-Only) |
| `SubEventDecomposer` | `planning.py` | Chia sự kiện thành Arrival / Confrontation / Turn | Đông cứng (Inference-Only) |
| `PlanCriticAgent` | `planning.py` | Kiểm duyệt tính hợp lệ và phụ thuộc của dàn ý | Rule-based |
| `PlanningAgent` | `planning.py` | Chia chương và gán ngân sách từ cho từng cảnh | Rule-based / LLM |
| `CoordinatorAgent` | `coordinator.py` | Nén ngữ cảnh lịch sử ReIO ($s_i < 4096$ tokens) | Đông cứng (Inference-Only) |
| `Artifact Cleaner & Dedup Gate` | `pipeline.py` | Xóa token đặc biệt & cắt/retry đoạn lặp (>30 từ) | Python/Regex 0-token |
| `CompositeRewardEngine` | `rewards.py` | Động cơ tổng hợp 3 hàm thưởng nội tại | Không tham số (Frozen Evaluators) |
| `VLLMWritingPolicy` | `pipeline.py` | Điều phối sinh rollout song song | Base Engine |
| `IntrinsicRewardFunctions` | `scripts/train.py` | Cầu nối giữa CompositeRewardEngine và TRL | Adapter Pattern |
| **`WritingAgent (pi_theta)`** | `scripts/train.py` | **Mô hình sinh văn bản chính (Qwen2.5-7B)** | **Huấn luyện duy nhất (QLoRA + GRPO)** |

