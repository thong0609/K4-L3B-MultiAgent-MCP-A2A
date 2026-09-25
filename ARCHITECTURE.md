# L3B Architecture Record

Team cập nhật tài liệu này cùng source. Mục tiêu là mô tả quyết định có thể kiểm chứng, không ghi prompt bí mật hoặc chain-of-thought.

## 1. System overview

Luồng xử lý từ input/candidate resolution đến MCP investigation, specialist agents, conflict resolver, verifier, output và trace:

```text
Input ──> Entity Resolver ──> Coordinator ──> Specialist Agents ──> Policy / Conflict ──> Verifier ──> Output
               │                                      │                       │               │
               └────────────────── MCP Gateway ───────┴───────────────────────┴───────────────┘
                                                      │
                                           Observable Trace Event
```

Hệ thống hoạt động theo mô hình điều phối A2A có cấu trúc:
1. **Coordinator** nhận case, sinh `case_received`, phân công task cho các specialist agents (`task_assigned`).
2. **Entity Resolver Agent** tra cứu `get_customer_history` để đối chiếu với `candidate_order_ids` và `customer_unique_id_hint`, phân giải `resolved_order_ids` và `rejected_candidates`.
3. **Investigation Specialists** gọi các tools MCP chuyên biệt (`get_order`, `get_shipment_summary`, `get_order_payments`, `get_sellers`, `get_product_context`), đồng thời lưu cache per-case để tối đa hóa điểm `efficiency`.
4. **Deterministic Analysis Engine** tính toán chính xác số liệu tài chính (`captured_total_brl`, `refunded_total_brl`, `refundable_total_brl`) và đối chiếu mốc thời gian giao hàng (`shipping_limit_at` vs `delivered_carrier_at` vs `delivered_customer_at`).
5. **Policy / Conflict Agent** đối chiếu `get_policy` để đánh giá từng claim (`claim_assessments`), xác định `primary_issue`, tính toán số tiền hoàn (`financial_resolution`) và hành động xử lý (`resolution_actions`).
6. **Verifier Agent** kiểm tra toàn vẹn hợp đồng (schema compliance, invariant check, evidence provenance) và phát sinh `verification_completed` trước khi Coordinator chốt `case_finalized`.

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Tool permission | Output/handoff |
| --- | --- | --- | --- | --- |
| `coordinator` | `case.json` | Khởi tạo vòng đời case, quản lý handoff và đóng gói kết quả | Không gọi tool trực tiếp | Phân công `task_assigned` tới các specialist |
| `entity_resolver` | `customer_hint`, `candidate_order_ids`, `claimed_order_id` | Định danh order chính xác, loại bỏ ứng viên giả mạo | `get_customer_history` | `resolved_order_ids`, `rejected_candidates` -> Handoff tới `investigation_agent` |
| `investigation_agent` | `resolved_order_ids`, `scope` | Thu thập dữ liệu đơn hàng, vận chuyển, thanh toán, người bán | `get_order`, `get_shipment_summary`, `get_order_payments`, `get_sellers`, `get_product_context` | Dữ liệu thô và `evidence_ref` -> Handoff tới `analyst_agent` |
| `shipment_specialist` | Dữ liệu `shipment`, `order`, `opened_at` | Đánh giá trễ hạn, xác định trách nhiệm của seller hay logistics | Dữ liệu từ investigation | `verdict`, `late_seller_ids`, `timeline_complete` |
| `payment_specialist` | Dữ liệu `payments`, `payment_timeline`, `refund_timeline` | Đối soát doanh thu, phát hiện trùng lặp, tính tiền hoàn | `get_payment_timeline`, `get_refund_timeline` | `captured_total_brl`, `refunded_total_brl`, `refundable_total_brl`, `verdict` |
| `policy_agent` | `claims`, kết quả shipment/payment, policy rules | Khớp rule với khiếu nại, tính toán mức bồi hoàn chuẩn xác | `get_policy` | `primary_issue`, `case_status`, `claim_assessments`, `refund_brl` -> Handoff tới `verifier` |
| `verifier` | Toàn bộ payload output trước khi ghi file | Kiểm định JSON Schema, quan hệ logic giữa các trường | Không gọi tool | `verification_completed` -> Handoff tới `coordinator` |

## 3. Entity resolution và A2A protocol

- **Xếp hạng & Loại trừ Candidate**:
  - Tra cứu lịch sử đơn hàng của khách hàng qua `get_customer_history(customer_unique_id)`.
  - Tập hợp tất cả các `order_id` có thật đã mua của khách hàng.
  - Candidate nào nằm trong danh sách mua của khách hàng sẽ được chấp nhận (`resolved_order_ids`), các candidate không tồn tại hoặc sai lệch sẽ đưa vào `rejected_candidates`.
  - Lọc theo mốc thời gian `opened_at`: nếu khách hàng có nhiều đơn hàng cùng ID hoặc khác ID, chỉ xét các đơn mua trước hoặc tại thời điểm mở khiếu nại (`order_purchase_timestamp <= opened_at`).
- **A2A Protocol & Correlation**:
  - Mọi sự kiện liên lạc và chuyển giao nhiệm vụ đều được định danh bằng `case_id` và `event_id`.
  - Thứ tự bắt buộc: `case_received` -> `task_assigned` -> `tool_result_consumed` -> `handoff` -> `policy_decided` -> `verification_completed` -> `case_finalized`.

## 4. Evidence và conflict lifecycle

- **Validation & Provenance**:
  - Mọi phản hồi từ MCP Gateway được validate bằng schema `day09-mcp-evidence-v1`.
  - Trích xuất `evidence_ref` thật từ MCP server và ghi nhận vào trace event `tool_result_consumed`.
  - Tuyệt đối không can thiệp, tự tạo mã giả hoặc chia sẻ `evidence_ref` chéo giữa các case.
- **Conflict Handling**:
  - Khi có sự sai lệch giữa thông tin khách hàng tự khai (`claimed_order_id`, claim amount) và dữ liệu hệ thống từ MCP, hệ thống ưu tiên dữ liệu chứng cứ từ MCP Gateway (`authoritative source precedence`).

## 5. Failure and efficiency policy

| Failure | Retry budget | Fallback | Trace event/code |
| --- | ---: | --- | --- |
| MCP tool call error | 1 retry | Bỏ qua tool phụ không bắt buộc, giữ tool cốt lõi | `tool_call_failed` |
| Entity ambiguous / not found | 0 retry | Chọn candidate đầu tiên hoặc claimed_order_id, confidence = 0.5 | `entity_ambiguous` |
| Missing seller/product context | 0 retry | Fallback sang danh sách seller từ shipment limits | `context_skipped` |
| LLM API rate limit / timeout | 1 retry | Chuyển sang deterministic rule-based specialist | `llm_fallback` |

- **Quản lý Cache & Budget gọi tool (Efficiency Score)**:
  - Áp dụng bộ nhớ đệm `evidence_cache` trong phạm vi từng case: cùng 1 tool và tham số chỉ được gọi duy nhất 1 lần.
  - Không gọi `get_order` đối với các candidate giả mạo không tồn tại trong `get_customer_history`.

## 6. Verification invariants

Trước khi xuất kết quả ra `outputs/<case_id>.json`, Verifier kiểm tra các bất biến (invariants):
1. **Schema Invariant**: Output tuân thủ 100% schema `day09-l3b-output-v2`.
2. **Entity Consistency**: Mọi candidate order ID phải nằm trong `resolved_order_ids` hoặc `rejected_candidates`.
3. **Provenance Invariant**: Danh sách `evidence_refs` của output chỉ chứa các ref thực tế đã thu thập và emit trong case hiện tại.
4. **Financial Consistency**: Nếu `primary_issue` không yêu cầu hoàn tiền (`no_action`), `recommended_refund_brl` phải bằng 0. Nếu có hoàn tiền, số tiền phải khớp với chính sách `refund_brl` và không vượt quá `captured_total_brl`.
5. **Responsibility Invariant**: Nếu kết luận `seller_delay`, bên chịu trách nhiệm phải có `party_type == "seller"`. Nếu `logistics_delay`, bên chịu trách nhiệm phải là `logistics_provider`.

## 7. Reproducibility

- **Mô hình**: Google Gemma 2 9B (`gemma2-9b-it`) / Gemma 3 4B (`google/gemma-3-4b-it`) hoặc Qwen 2.5 7B (`qwen-2.5-7b-instruct`) tuân thủ nghiêm ngặt giới hạn $\le$ 9 tỷ tham số.
- **Môi trường**: Python 3.11+, MCP Client SDK 2.x, `httpx2`, `jsonschema`.
- **Lệnh thực thi**:
  ```powershell
  day09 run
  day09 validate
  day09 package --output dist/submission.zip
  ```
