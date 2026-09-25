from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI
from mcp.types import CallToolResult

from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter

async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Implement the L3B coordinator and specialist-agent workflow here."""
    
    # 1. Khởi tạo Client gọi đến OpenRouter
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Bạn chưa cấu hình OPENROUTER_API_KEY trong file .env")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    model_name = "qwen/qwen-2.5-7b-instruct"

    # Lấy danh sách tool từ MCP và convert sang định dạng Function Calling của OpenAI
    raw_tools = await gateway._session.list_tools()
    openai_tools = []
    for t in raw_tools.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema
            }
        })

    # Khởi tạo tin nhắn cho Assistant
    schema_instructions = """
    Trích xuất root cause, shipment, payment. Cuối cùng bạn PHẢI gọi tool `submit_final_result` để nộp bài JSON theo CẤU TRÚC SAU:
    {
      "schema_version": "day09-l3b-output-v2",
      "case_id": "<case_id_here>",
      "assessment": {
        "primary_issue": "insufficient_evidence",
        "secondary_issues": [],
        "case_status": "needs_investigation",
        "confidence": 0.0
      },
      "affected_entities": {
        "order_ids": [], "item_ids": [], "seller_ids": [], "payment_references": [], "shipment_ids": []
      },
      "entity_resolution": {
        "status": "not_found", "resolved_order_ids": [], "rejected_candidates": [], "confidence": 0.0
      },
      "customer_context": {
        "customer_unique_id": null, "related_order_ids": []
      },
      "shipment_analysis": {
        "verdict": "insufficient_evidence", "late_seller_ids": [], "timeline_complete": false
      },
      "payment_analysis": {
        "verdict": "insufficient_evidence", "captured_total_brl": null, "refunded_total_brl": null, "refundable_total_brl": null
      },
      "root_cause_analysis": {
        "ranked_causes": [{"cause_code": "UNKNOWN_ERROR", "rank": 1}],
        "responsible_parties": [{"party_type": "unknown", "party_id": null}]
      },
      "evidence_refs": [],
      "data_conflicts": [],
      "financial_resolution": {
        "currency": "BRL", "recommended_refund_brl": 0, "refund_lines": []
      },
      "resolution_actions": []
    }
    Lưu ý KHÔNG được thêm các trường như claim_status, customer_id, order_id vào object chính hoặc affected_entities.
    Chỉ dùng ĐÚNG CÁC TRƯỜNG ở trên.
    """
    messages = [
        {"role": "system", "content": f"Bạn là điều tra viên e-commerce. {schema_instructions}"},
        {"role": "user", "content": f"Case ID: {case['case_id']}\nData: {json.dumps(case)}"}
    ]

    # Thêm 1 tool đặc biệt để bắt model nộp bài cuối cùng
    openai_tools.append({
        "type": "function",
        "function": {
            "name": "submit_final_result",
            "description": "Gọi hàm này khi bạn đã thu thập đủ bằng chứng và muốn kết luận. Tham số truyền vào là JSON đúng schema L3B",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_json": {"type": "string", "description": "JSON string đúng schema day09-l3b-output-v2"}
                },
                "required": ["final_json"]
            }
        }
    })

    trace.emit(
        case_id=case["case_id"], 
        event_type="task_assigned", 
        actor="coordinator", 
        attributes={"message": "Bat dau phan tich case"}
    )

    print(f"[{case['case_id']}] Dang phan tich...")
    
    # 1. Thu thập TẤT CẢ dữ liệu trước bằng Python (chạy song song)
    import asyncio
    evidence_list = []
    
    def get_fallback(case_data):
        return {
            "schema_version": "day09-l3b-output-v2",
            "case_id": case_data["case_id"],
            "assessment": {
                "primary_issue": "insufficient_evidence",
                "secondary_issues": [],
                "case_status": "needs_investigation",
                "confidence": 0
            },
            "affected_entities": {
                "order_ids": [],
                "item_ids": [],
                "seller_ids": [],
                "payment_references": [],
                "shipment_ids": []
            },
            "entity_resolution": {
                "status": "not_found",
                "resolved_order_ids": [],
                "rejected_candidates": [],
                "confidence": 0
            },
            "customer_context": {
                "customer_unique_id": None,
                "related_order_ids": []
            },
            "shipment_analysis": {
                "verdict": "insufficient_evidence",
                "late_seller_ids": [],
                "timeline_complete": False
            },
            "payment_analysis": {
                "verdict": "insufficient_evidence",
                "captured_total_brl": None,
                "refunded_total_brl": None,
                "refundable_total_brl": None
            },
            "root_cause_analysis": {
                "ranked_causes": [],
                "responsible_parties": []
            },
            "evidence_refs": [],
            "data_conflicts": [],
            "financial_resolution": {
                "currency": "BRL",
                "recommended_refund_brl": 0,
                "refund_lines": []
            },
            "resolution_actions": []
        }

    async def fetch_data(tool_name, **kwargs):
        try:
            res = await gateway.call(tool_name, **kwargs)
            evidence_ref = res.get("evidence_ref")
            trace.emit(
                case_id=case["case_id"], 
                event_type="tool_result_consumed", 
                actor="coordinator", 
                tool_name=tool_name, 
                evidence_refs=[evidence_ref] if evidence_ref else []
            )
            return {"tool": tool_name, "data": res}
        except Exception as e:
            print(f"[{case['case_id']}] Tool {tool_name} failed: {e}")
            return {"tool": tool_name, "error": str(e)}

    tasks = []
    for order_id in case.get("candidate_order_ids", []):
        tasks.append(fetch_data("get_order", case_id=case["case_id"], order_id=order_id))
        tasks.append(fetch_data("get_shipment_summary", case_id=case["case_id"], order_id=order_id))
        tasks.append(fetch_data("get_order_payments", case_id=case["case_id"], order_id=order_id))
        tasks.append(fetch_data("get_order_items", case_id=case["case_id"], order_id=order_id))
        
    if case.get("customer_unique_id_hint"):
        tasks.append(fetch_data("get_customer_history", case_id=case["case_id"], customer_unique_id=case["customer_unique_id_hint"]))
        
    if case.get("policy_version"):
        tasks.append(fetch_data("get_policy", case_id=case["case_id"], policy_version=case["policy_version"]))
        
    results = await asyncio.gather(*tasks)
    evidence_list.extend(results)

    # 2. Đưa toàn bộ vào tin nhắn cho LLM
    messages = [
        {"role": "system", "content": f"Bạn là điều tra viên. {schema_instructions}"},
        {"role": "user", "content": f"Case ID: {case['case_id']}\nYêu cầu của khách: {json.dumps(case)}\n\nBẰNG CHỨNG ĐÃ THU THẬP:\n{json.dumps(evidence_list, ensure_ascii=False)}"}
    ]

    print(f"[{case['case_id']}] Cho Qwen doc du lieu...")
    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=openai_tools,
        tool_choice={"type": "function", "function": {"name": "submit_final_result"}}
    )
    
    if not getattr(msg, "tool_calls", None):
        print(f"[{case['case_id']}] Model không nhả ra tool call nào.")
        return get_fallback(case)
        
    for tool_call in msg.tool_calls:
        if tool_call.function.name == "submit_final_result":
            try:
                args = json.loads(tool_call.function.arguments)
                final_json_str = args.get("final_json", "")
                
                # Clean up markdown backticks if any
                final_json_str = final_json_str.strip()
                if final_json_str.startswith("```json"):
                    final_json_str = final_json_str[7:]
                elif final_json_str.startswith("```"):
                    final_json_str = final_json_str[3:]
                if final_json_str.endswith("```"):
                    final_json_str = final_json_str[:-3]
                final_json_str = final_json_str.strip()
                
                import re
                match = re.search(r'\{.*\}', final_json_str, re.DOTALL)
                if match:
                    final_json_str = match.group(0)
                    
                final_output = json.loads(final_json_str)
                final_output["case_id"] = case["case_id"]
                print(f"[{case['case_id']}] Hoan thanh!")
                return final_output
            except Exception as e:
                print(f"[{case['case_id']}] JSON parse error: {e}")
                
                return get_fallback(case)
                
    print(f"[{case['case_id']}] Không gọi đúng hàm submit_final_result")
    return get_fallback(case)
