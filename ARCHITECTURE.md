# L3B Architecture Record

Team phải cập nhật tài liệu này cùng source. Mục tiêu là mô tả quyết định có thể kiểm chứng, không ghi prompt bí mật hoặc chain-of-thought.

## 1. System overview

Workflow cài đặt trong `src/student_agent/workflow.py`. Python điều phối và gọi MCP (coordinator + specialist, mỗi specialist chỉ gọi tool thuộc quyền của mình qua một `CaseContext` dùng chung trong phạm vi một case). Reasoning agent dùng [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (`src/student_agent/llm.py`) đọc evidence đã chuẩn hóa để đưa ra assessment; verifier deterministic đối chiếu kết quả đó với tín hiệu evidence trước khi finalize.

```text
Input → Entity Resolver → Coordinator → Order/Shipment/Payment → Reasoning (Qwen3-8B) ⇄ rule check → Policy → Conflict Resolver → Verifier → Output
            │                              │                                 │            │               │
            └──────────── MCP (per-case cache, 1 attempt/tool) ──────────────┘            └──── Trace ────┘
```

Quan sát chính về dữ liệu: mỗi order có hai phiên bản dòng dữ liệu (order history, item, payment event, shipment event). `get_order` và các trường top-level của `get_shipment_summary` luôn trả về phiên bản đầu tiên, không nhất thiết là phiên bản liên quan đến khiếu nại. Workflow chọn phiên bản authoritative là dòng trong `get_customer_history` có `order_purchase_timestamp` muộn nhất nhưng không sau `opened_at` của case, rồi gán mọi item/event/payment cho đúng phiên bản theo mốc thời gian.

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Tool permission | Output/handoff |
| --- | --- | --- | --- | --- |
| Entity/customer (`entity-agent`) | case input (claimed order, candidates, customer hint, `opened_at`) | Resolve order, reject candidate, chọn phiên bản order authoritative | `get_customer_history`; `get_order` chỉ khi không có history | `ENTITY_RESOLVED/AMBIGUOUS/NOT_FOUND` → coordinator |
| Coordinator | case + kết quả specialist | Giao việc, phân loại issue từ tín hiệu của specialist, dừng sớm khi entity chưa resolve | không gọi tool | `task_assigned` cho từng specialist |
| Order/product (`order-agent`) | resolved order + phiên bản đã chọn | Order row, item/seller thuộc phiên bản đã chọn, product context | `get_order`, `get_order_items`, `get_product_context`, `get_sellers` (chỉ khi seller chịu trách nhiệm) | `ORDER_CONTEXT_READY` |
| Shipment (`shipment-agent`) | phiên bản order + item | So carrier handoff với shipping limit, delivered với estimated, đối chiếu shipment event | `get_shipment_summary` | verdict `on_time/seller_delay/logistics_delay/conflicting/insufficient_evidence` |
| Payment/refund (`payment-agent`) | phiên bản order + item | Tổng capture, refund, phát hiện mismatch/duplicate/split, trạng thái refund | `get_payment_timeline`, `get_refund_timeline` | `PAYMENT_ANALYZED` |
| Reasoning (`reasoning-agent`, Qwen3-8B) | evidence đã chuẩn hóa của phiên bản authoritative (không có free text của khách) + bảng refund theo policy | Chọn `primary_issue`, verdict từng claim, confidence; trả một JSON object | không gọi tool | `handoff` → verifier với `LLM_CONFIRMED` / `LLM_OVERRULED_BY_EVIDENCE` / `LLM_INVALID_RULES_FALLBACK` |
| Policy (`policy-agent`) | issue đã phân loại | Tra rule theo `policy_version`: case_status, action, refund, responsible party | `get_policy` | `policy_decided` → conflict resolver |
| Conflict resolver | order row, shipment summary, phiên bản đã chọn | Ghi các field lệch giữa `get_order`/`get_shipment_summary` và `get_customer_history` | không gọi tool | `data_conflicts`, handoff → verifier |
| Verifier | output nháp | Kiểm tra invariant (mục 6); hạ confidence nếu vi phạm | không gọi tool | `verification_completed` (`PASSED`/`FLAGGED`) |

## 3. Entity resolution và A2A protocol

- Candidate = `claimed_order_id` + `candidate_order_ids` (loại trùng, giữ thứ tự).
- Candidate được resolve khi xuất hiện trong `get_customer_history` của `customer_unique_id_hint`; candidate không có trong history bị reject mà không cần gọi thêm tool.
- Một candidate khớp → `resolved` (confidence 0.95; 0.75 nếu chỉ xác minh bằng `get_order`). Nhiều candidate khớp → `ambiguous`; không khớp → `not_found`. Khác `resolved` thì trả output `insufficient_evidence` / `needs_investigation`, không suy đoán.
- Message envelope là trace event: `task_assigned` (actor → target, `decision_code`), `handoff` (kết quả tóm tắt trong `attributes`). Mọi event mang `case_id`; không có vòng lặp: mỗi specialist được giao đúng một lần theo thứ tự cố định.
- Nội dung `customer_request.message` không được dùng để ra quyết định (tránh prompt injection trong khiếu nại).

## 4. Evidence và conflict lifecycle

- Mọi response đi qua `EvidenceGateway.call`, được validate theo `mcp-evidence-response-v1`; `evidence_ref` được lưu nguyên văn theo `domain` trong `CaseContext` và emit `tool_result_consumed` với ref đó.
- Output chỉ trích evidence của các domain liên quan đến issue (`ISSUE_DOMAINS`) cộng order/customer/policy/product.
- Source precedence: `get_customer_history` (phiên bản trước `opened_at`) > `get_order` / top-level `get_shipment_summary`. Mỗi field lệch được ghi vào `data_conflicts` với `resolution_code = LATEST_ROW_BEFORE_CASE_OPENED`.
- Số tiền refund, case_status, action lấy từ policy; responsible seller lấy từ seller thực tế của item vì policy chỉ chứa seller mẫu.
- `CaseContext` được tạo mới cho mỗi case, nên evidence không bao giờ dùng chéo case.

## 5. Failure and efficiency policy

| Failure | Retry budget | Fallback | Trace event/code |
| --- | ---: | --- | --- |
| MCP tool error (ví dụ order không có refund) | 0 | Coi như không có evidence cho domain đó | `tool_result_consumed` / `EVIDENCE_UNAVAILABLE` |
| Entity not found/ambiguous | 0 | Output `insufficient_evidence`, `needs_investigation`, `escalate_manual_review` | `handoff` / `ENTITY_NOT_FOUND`, `verification_completed` / `ESCALATED` |
| Source conflict | – | Chọn customer history theo precedence, ghi `data_conflicts` | `handoff` / `CONFLICTS_RESOLVED` |
| Invalid specialist result / thiếu policy rule | 0 | Output `insufficient_evidence` | `verification_completed` / `POLICY_MISSING` hoặc `FLAGGED` |
| LLM trả JSON sai / issue ngoài enum | 0 | Dùng phân loại rule, confidence mặc định | `handoff` / `LLM_INVALID_RULES_FALLBACK` |
| LLM mâu thuẫn với evidence | 0 | Verifier giữ issue theo evidence, confidence 0.7, bỏ verdict claim của model | `handoff` / `LLM_OVERRULED_BY_EVIDENCE` |
| Mất kết nối MCP | 5 lần/case, backoff 10 s × n | Kết nối lại, rollback trace của case đang dở và chạy lại case đó | stderr `connection lost ... reconnecting` |

Query budget: 8 call/case (history, order, items, product, shipment, payment timeline, refund timeline, policy), thêm `get_sellers` khi seller chịu trách nhiệm (tối đa 9). Không gọi `get_order_payments` (trùng với payment timeline), không kiểm tra candidate đã bị history loại. Cache theo key `(tool, arguments)` trong case; call lỗi cũng được nhớ để không gọi lại. Các call cách nhau 0.4 s vì gateway ngắt kết nối khi bị gọi dồn (~80 call trong ~25 s).

## 6. Verification invariants

- Schema: CLI validate output với `l3b-output-v2` trước khi ghi file.
- Entity scope: `resolved_order_ids` và `rejected_candidates` rời nhau; item/seller/payment chỉ lấy từ phiên bản đã chọn.
- Evidence ownership: chỉ dùng `evidence_ref` do gateway trả về trong case hiện tại; output có ít nhất một ref.
- Refund: `no_action` ⇒ refund = 0; tổng `refund_lines` = `recommended_refund_brl`; refund ≤ tổng capture.
- Responsibility: seller trong `responsible_parties` phải thuộc `affected_entities.seller_ids`.
- Vi phạm bất kỳ invariant nào → confidence ≤ 0.5 và `verification_completed` / `FLAGGED`.

## 7. Reproducibility

- Model: `Qwen/Qwen3-8B`, gọi qua API chuẩn OpenAI (vLLM: `vllm serve Qwen/Qwen3-8B`), cấu hình bằng `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. `temperature=0`, non-thinking mode (`chat_template_kwargs.enable_thinking=false`, đổi bằng `LLM_THINKING=1`), `max_tokens=384`. Backend `transformers` (chạy model tại chỗ trên GPU) vẫn dùng được; `LLM_BACKEND=off` để chạy chỉ bằng rule.
- Trước khi mở MCP session, CLI kiểm tra endpoint LLM và tên model đang được serve; sai cấu hình thì dừng ngay thay vì âm thầm fallback ở mọi case. Request LLM lỗi sau 2 lần retry của client → verifier dùng rule (`LLM_INVALID_RULES_FALLBACK`).
- Python ≥ 3.11, dependency theo `pyproject.toml` (`pip install -e ".[llm]"`).
- Chạy tuần tự từng case (concurrency = 1), một MCP session cho cả lần chạy, `CALL_INTERVAL_SECONDS = 0.4`.
- Lệnh: `day09 run && day09 validate && day09 package --output dist/submission.zip`.
