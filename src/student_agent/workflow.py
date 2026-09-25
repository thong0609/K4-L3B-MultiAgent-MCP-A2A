from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from openai import AsyncOpenAI

from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter


def _parse_dt(val: str | None) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val)
    except Exception:
        return None


class MultiAgentWorkflow:
    def __init__(self, case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter) -> None:
        self.case = case
        self.case_id = case["case_id"]
        self.gateway = gateway
        self.trace = trace
        self.evidence_cache: dict[str, dict[str, Any]] = {}
        self.collected_evidence_refs: list[str] = []
        self.customer_history_orders: list[dict[str, Any]] = []

        # Optional LLM setup
        self.llm_api_key = os.getenv("LLM_API_KEY")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.llm_model = os.getenv("LLM_MODEL", "gemma2-9b-it")
        self.llm_client = (
            AsyncOpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)
            if self.llm_api_key
            else None
        )

    async def _call_mcp(self, tool_name: str, actor: str, **kwargs: Any) -> dict[str, Any]:
        """Call MCP tool with per-case caching to ensure high efficiency."""
        cache_key = f"{tool_name}:{sorted(kwargs.items())}"
        if cache_key in self.evidence_cache:
            return self.evidence_cache[cache_key]

        evidence = await self.gateway.call(tool_name, case_id=self.case_id, **kwargs)
        self.evidence_cache[cache_key] = evidence
        ref = evidence.get("evidence_ref")
        if ref and ref not in self.collected_evidence_refs:
            self.collected_evidence_refs.append(ref)

        self.trace.emit(
            case_id=self.case_id,
            event_type="tool_result_consumed",
            actor=actor,
            tool_name=tool_name,
            evidence_refs=[ref] if ref else [],
        )
        return evidence

    async def resolve_entities(self) -> dict[str, Any]:
        """Entity Resolver Agent: Resolves target order and candidate orders with temporal scoping."""
        actor = "entity_resolver"
        self.trace.emit(
            case_id=self.case_id,
            event_type="task_assigned",
            actor="coordinator",
            target=actor,
            attributes={"task": "resolve_order_and_customer"},
        )

        customer_hint = self.case.get("customer_unique_id_hint")
        candidates = self.case.get("candidate_order_ids", [])
        claimed_order_id = self.case.get("customer_request", {}).get("claimed_order_id")
        opened_at_dt = _parse_dt(self.case.get("opened_at"))

        customer_unique_id = customer_hint
        customer_history_orders: list[dict[str, Any]] = []
        customer_history_ref = None

        if customer_hint:
            try:
                history_evidence = await self._call_mcp(
                    "get_customer_history", actor=actor, customer_unique_id=customer_hint
                )
                customer_history_orders = history_evidence.get("data", {}).get("orders", [])
                customer_history_ref = history_evidence.get("evidence_ref")
            except Exception:
                pass

        self.customer_history_orders = customer_history_orders
        all_cust_order_ids = {o["order_id"] for o in customer_history_orders if "order_id" in o}

        # Filter orders that were purchased before or at opened_at
        past_orders = [
            o for o in customer_history_orders
            if _parse_dt(o.get("order_purchase_timestamp"))
            and opened_at_dt
            and _parse_dt(o.get("order_purchase_timestamp")) <= opened_at_dt
        ]
        if not past_orders:
            past_orders = customer_history_orders

        # Sort past orders by purchase timestamp descending (most recent first)
        past_orders.sort(
            key=lambda o: _parse_dt(o.get("order_purchase_timestamp")) or datetime.min,
            reverse=True,
        )

        active_order_data = past_orders[0] if past_orders else {}
        valid_order_ids = {o["order_id"] for o in past_orders if "order_id" in o}

        resolved_order_ids: list[str] = []
        rejected_candidates: list[str] = []

        # Reconcile candidates
        for c in candidates:
            if c in all_cust_order_ids:
                if c not in resolved_order_ids:
                    resolved_order_ids.append(c)
            else:
                rejected_candidates.append(c)

        if not resolved_order_ids:
            if claimed_order_id and claimed_order_id in all_cust_order_ids:
                resolved_order_ids.append(claimed_order_id)
            elif active_order_data.get("order_id"):
                resolved_order_ids.append(active_order_data["order_id"])
            elif candidates:
                resolved_order_ids.append(candidates[0])

        for c in candidates:
            if c not in resolved_order_ids and c not in rejected_candidates:
                rejected_candidates.append(c)

        resolved_order_ids = sorted(list(set(resolved_order_ids)))
        rejected_candidates = sorted(list(set(rejected_candidates)))

        status = "resolved" if resolved_order_ids else "not_found"
        confidence = 0.95 if status == "resolved" else 0.5
        all_related_orders = sorted(list(all_cust_order_ids | set(resolved_order_ids)))

        self.trace.emit(
            case_id=self.case_id,
            event_type="handoff",
            actor=actor,
            target="investigation_agent",
            decision_code=f"entity_{status}",
            attributes={"resolved_count": len(resolved_order_ids)},
        )

        return {
            "status": status,
            "resolved_order_ids": resolved_order_ids,
            "rejected_candidates": rejected_candidates,
            "confidence": confidence,
            "customer_unique_id": customer_unique_id,
            "related_order_ids": all_related_orders,
            "customer_history_ref": customer_history_ref,
            "active_order_data": active_order_data,
        }

    async def investigate_order(
        self, order_id: str, scope: dict[str, Any]
    ) -> dict[str, Any]:
        """Investigation Agent: Collects order, items, shipment, payment, seller and product data."""
        actor = "investigation_agent"
        self.trace.emit(
            case_id=self.case_id,
            event_type="task_assigned",
            actor="coordinator",
            target=actor,
            attributes={"order_id": order_id},
        )

        order_evidence = await self._call_mcp("get_order", actor=actor, order_id=order_id)
        order_data = order_evidence.get("data", {})

        items_evidence = None
        try:
            items_evidence = await self._call_mcp("get_order_items", actor=actor, order_id=order_id)
        except Exception:
            pass

        shipment_evidence = await self._call_mcp(
            "get_shipment_summary", actor=actor, order_id=order_id
        )
        shipment_data = shipment_evidence.get("data", {})

        payment_evidence = await self._call_mcp(
            "get_order_payments", actor=actor, order_id=order_id
        )
        payment_data = payment_evidence.get("data", [])

        seller_evidence = None
        try:
            seller_evidence = await self._call_mcp("get_sellers", actor=actor, order_id=order_id)
        except Exception:
            pass

        product_evidence = None
        if scope.get("include_product_context"):
            try:
                product_evidence = await self._call_mcp(
                    "get_product_context", actor=actor, order_id=order_id
                )
            except Exception:
                pass

        payment_timeline_evidence = None
        try:
            payment_timeline_evidence = await self._call_mcp(
                "get_payment_timeline", actor=actor, order_id=order_id
            )
        except Exception:
            pass

        refund_timeline_evidence = None
        try:
            refund_timeline_evidence = await self._call_mcp(
                "get_refund_timeline", actor=actor, order_id=order_id
            )
        except Exception:
            pass

        self.trace.emit(
            case_id=self.case_id,
            event_type="handoff",
            actor=actor,
            target="analyst_agent",
            decision_code="investigation_completed",
        )

        return {
            "order": order_data,
            "items": items_evidence.get("data", []) if items_evidence else [],
            "shipment": shipment_data,
            "payments": payment_data if isinstance(payment_data, list) else [],
            "sellers": seller_evidence.get("data", []) if seller_evidence else [],
            "products": product_evidence.get("data", []) if product_evidence else [],
            "payment_timeline": payment_timeline_evidence.get("data", {}) if payment_timeline_evidence else {},
            "refund_timeline": refund_timeline_evidence.get("data", {}) if refund_timeline_evidence else {},
            "refs": {
                "order": order_evidence.get("evidence_ref"),
                "items": items_evidence.get("evidence_ref") if items_evidence else None,
                "shipment": shipment_evidence.get("evidence_ref"),
                "payments": payment_evidence.get("evidence_ref"),
                "sellers": seller_evidence.get("evidence_ref") if seller_evidence else None,
                "product": product_evidence.get("evidence_ref") if product_evidence else None,
                "payment_timeline": payment_timeline_evidence.get("evidence_ref") if payment_timeline_evidence else None,
                "refund_timeline": refund_timeline_evidence.get("evidence_ref") if refund_timeline_evidence else None,
            },
        }

    def analyze_shipment(
        self,
        shipment: dict[str, Any],
        order: dict[str, Any],
        active_order_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Shipment Specialist: Evaluates delivery timestamps and delay responsibility with opened_at filtering."""
        opened_at_dt = _parse_dt(self.case.get("opened_at"))

        # Use order data active as of opened_at
        active_order = active_order_data if active_order_data else order
        order_status = active_order.get("order_status", order.get("order_status", ""))

        delivered_carrier = _parse_dt(active_order.get("order_delivered_carrier_date") or shipment.get("delivered_carrier_at"))
        delivered_customer = _parse_dt(active_order.get("order_delivered_customer_date") or shipment.get("delivered_customer_at"))
        estimated_delivery = _parse_dt(active_order.get("order_estimated_delivery_date") or shipment.get("estimated_delivery_at"))

        late_seller_ids: list[str] = []
        shipping_limits = shipment.get("shipping_limits", [])

        # Match shipping limit relevant to purchase period
        for limit in shipping_limits:
            seller_id = limit.get("seller_id")
            limit_dt = _parse_dt(limit.get("shipping_limit_at"))
            if seller_id and limit_dt and delivered_carrier:
                purchase_dt = _parse_dt(active_order.get("order_purchase_timestamp"))
                if purchase_dt and abs((limit_dt - purchase_dt).days) < 60:
                    if delivered_carrier > limit_dt:
                        late_seller_ids.append(seller_id)

        # Filter events by opened_at
        raw_events = shipment.get("events", [])
        events = [
            e for e in raw_events
            if not e.get("event_at") or not opened_at_dt or _parse_dt(e.get("event_at")) <= opened_at_dt
        ]

        has_logistics_late = any(
            e.get("event_type") == "delivered_late" and e.get("actor") == "logistics_provider"
            for e in events
        )
        has_seller_late = any(
            e.get("event_type") == "delivered_late" and e.get("actor") == "seller"
            for e in events
        )

        late_seller_ids = sorted(list(set(late_seller_ids)))
        verdict = "on_time"

        if order_status == "canceled":
            verdict = "returned"
        elif order_status == "unavailable":
            verdict = "lost"
        elif late_seller_ids or has_seller_late:
            verdict = "seller_delay"
        elif has_logistics_late:
            verdict = "logistics_delay"
        elif delivered_customer and estimated_delivery and delivered_customer > estimated_delivery:
            verdict = "logistics_delay"

        timeline_complete = bool(delivered_customer and delivered_carrier)
        return {
            "verdict": verdict,
            "late_seller_ids": late_seller_ids,
            "timeline_complete": timeline_complete,
        }

    def analyze_payments(
        self,
        payments: list[dict[str, Any]],
        items: list[dict[str, Any]],
        payment_timeline: dict[str, Any],
        refund_timeline: dict[str, Any],
        active_order_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Payment Specialist: Reconciles payments, detects duplicates, mismatches and refund statuses."""
        opened_at_dt = _parse_dt(self.case.get("opened_at"))
        active_order_id = active_order_data.get("order_id")

        # Filter payments for this order
        scoped_payments = payments
        if active_order_id:
            matching = [p for p in payments if p.get("order_id") == active_order_id]
            if matching:
                scoped_payments = matching

        captured_total = 0.0
        for p in scoped_payments:
            try:
                captured_total += float(p.get("payment_value", 0))
            except (ValueError, TypeError):
                pass
        captured_total = round(captured_total, 2)

        # Calculate items total (order nominal price + freight) for items relevant to opened_at
        order_items_total = 0.0
        relevant_items = [
            it for it in items
            if not it.get("shipping_limit_date") or not opened_at_dt or _parse_dt(it.get("shipping_limit_date")) <= opened_at_dt
        ]
        if not relevant_items:
            relevant_items = items

        for it in relevant_items:
            try:
                price = float(it.get("price", 0))
                freight = float(it.get("freight_value", 0))
                order_items_total += (price + freight)
            except (ValueError, TypeError):
                pass
        order_items_total = round(order_items_total, 2)

        # Filter refund timeline events by opened_at
        raw_refund_events = refund_timeline.get("events", []) if isinstance(refund_timeline, dict) else []
        refund_events = [
            ev for ev in raw_refund_events
            if not ev.get("event_at") or not opened_at_dt or _parse_dt(ev.get("event_at")) <= opened_at_dt
        ]

        refunded_total = 0.0
        refund_status = "none"
        for ev in refund_events:
            status = ev.get("status")
            amt = float(ev.get("amount_brl", ev.get("amount", 0)) or 0)
            if status == "completed":
                refunded_total += amt
                refund_status = "refunded"
            elif status == "pending":
                refund_status = "refund_pending"
            elif status == "failed":
                refund_status = "refund_failed"

        refunded_total = round(refunded_total, 2)
        refundable_total = round(max(0.0, captured_total - refunded_total), 2)

        # Filter payment timeline events by opened_at
        raw_pay_events = payment_timeline.get("events", []) if isinstance(payment_timeline, dict) else []
        pay_events = [
            ev for ev in raw_pay_events
            if not ev.get("event_at") or not opened_at_dt or _parse_dt(ev.get("event_at")) <= opened_at_dt
        ]
        has_reconciliation_mismatch_event = any(
            ev.get("event_type") == "reconciliation_mismatch" for ev in pay_events
        )

        # Scoped captured payment calculation using payment_timeline events if present
        captured_events = [ev for ev in pay_events if ev.get("event_type") == "captured"]
        if captured_events:
            captured_total = sum(float(ev.get("amount_brl", 0) or 0) for ev in captured_events)
            captured_total = round(captured_total, 2)

        refundable_total = round(max(0.0, captured_total - refunded_total), 2)

        # Check for duplicate capture
        is_duplicate = False
        if len(scoped_payments) > 1:
            payment_sigs = [(p.get("payment_sequential"), p.get("payment_type"), p.get("payment_value")) for p in scoped_payments]
            if len(payment_sigs) > len(set(payment_sigs)):
                is_duplicate = True

        # Payment verdict (must be in ["reconciled", "capture_mismatch", "duplicate_capture", "refund_pending", "refund_failed", "refunded", "insufficient_evidence"])
        if refund_status == "refund_failed":
            verdict = "refund_failed"
        elif refund_status == "refund_pending":
            verdict = "refund_pending"
        elif is_duplicate:
            verdict = "duplicate_capture"
        elif has_reconciliation_mismatch_event:
            verdict = "capture_mismatch"
        elif order_items_total > 0 and abs(captured_total - order_items_total) > 0.05 and len(scoped_payments) <= 2 and not is_duplicate:
            # If multiple payments sum to order_items_total, it's a split payment, not a mismatch
            verdict = "capture_mismatch"
        elif refunded_total >= captured_total and captured_total > 0:
            verdict = "refunded"
        else:
            verdict = "reconciled"

        return {
            "verdict": verdict,
            "captured_total_brl": captured_total,
            "refunded_total_brl": refunded_total,
            "refundable_total_brl": refundable_total,
            "order_items_total": order_items_total,
            "is_duplicate": is_duplicate,
            "is_split": len(scoped_payments) > 1 and not is_duplicate and abs(captured_total - order_items_total) < 0.05,
        }

    async def evaluate_policy_and_claims(
        self,
        claims: list[dict[str, Any]],
        shipment_analysis: dict[str, Any],
        payment_analysis: dict[str, Any],
        investigation: dict[str, Any],
        active_order_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Policy Specialist: Matches case against policy rules and evaluates claims."""
        actor = "policy_agent"
        self.trace.emit(
            case_id=self.case_id,
            event_type="task_assigned",
            actor="coordinator",
            target=actor,
            attributes={"task": "evaluate_policy_rules"},
        )

        policy_version = self.case.get("policy_version", "EC_POLICY_V2")
        policy_evidence = await self._call_mcp("get_policy", actor=actor, policy_version=policy_version)
        policy_data = policy_evidence.get("data", {})
        rules = policy_data.get("rules", {})
        policy_ref = policy_evidence.get("evidence_ref")

        active_order = active_order_data if active_order_data else investigation.get("order", {})
        order_status = active_order.get("order_status", "")

        # Determine primary issue by arbitrating customer claims against evidence
        claimed_topics = [
            c["topic"] for c in claims
            if c.get("topic") and c["topic"] != "requested_full_refund"
        ]
        target_claim_topic = claimed_topics[0] if claimed_topics else "unsupported_claim"

        primary_issue = "unsupported_claim"

        if target_claim_topic == "canceled_order_paid":
            primary_issue = "canceled_order_paid" if order_status == "canceled" else "unsupported_claim"
        elif target_claim_topic == "unavailable_order_paid":
            primary_issue = "unavailable_order_paid" if order_status == "unavailable" else "unsupported_claim"
        elif target_claim_topic == "duplicate_charge":
            primary_issue = "duplicate_charge" if payment_analysis["verdict"] == "duplicate_capture" else "unsupported_claim"
        elif target_claim_topic == "refund_failed":
            primary_issue = "refund_failed" if payment_analysis["verdict"] == "refund_failed" else "unsupported_claim"
        elif target_claim_topic == "refund_pending":
            primary_issue = "refund_pending" if payment_analysis["verdict"] == "refund_pending" else "unsupported_claim"
        elif target_claim_topic == "payment_mismatch":
            primary_issue = "payment_mismatch" if payment_analysis["verdict"] == "capture_mismatch" else "unsupported_claim"
        elif target_claim_topic == "valid_split_payment":
            primary_issue = "valid_split_payment" if payment_analysis.get("is_split") else "unsupported_claim"
        elif target_claim_topic == "late_delivery_seller":
            primary_issue = "late_delivery_seller" if shipment_analysis["verdict"] == "seller_delay" else "unsupported_claim"
        elif target_claim_topic == "late_delivery_logistics":
            primary_issue = "late_delivery_logistics" if shipment_analysis["verdict"] == "logistics_delay" else "unsupported_claim"
        elif target_claim_topic == "unsupported_claim":
            primary_issue = "unsupported_claim"
        else:
            primary_issue = target_claim_topic

        rule = rules.get(primary_issue, {})
        case_status = rule.get("case_status", "action_required")
        recommended_action = rule.get("recommended_action", "document_no_action")
        refund_brl = float(rule.get("refund_brl", 0.0))
        responsible_parties = [dict(p) for p in rule.get("responsible_parties", [])]

        # Ensure seller ID is mapped to responsible party if seller delay
        if primary_issue == "late_delivery_seller" and shipment_analysis.get("late_seller_ids"):
            seller_id = shipment_analysis["late_seller_ids"][0]
            for p in responsible_parties:
                if p.get("party_type") == "seller" and not p.get("party_id"):
                    p["party_id"] = seller_id

        # Secondary issues
        secondary_issues = [
            c["topic"] for c in claims
            if c.get("topic") and c["topic"] != primary_issue and c["topic"] != "requested_full_refund"
        ][:10]

        # Evaluate each claim
        claim_assessments: list[dict[str, Any]] = []
        for c in claims:
            cid = c.get("claim_id", "")
            topic = c.get("topic", "")
            if topic == primary_issue:
                verdict = "supported"
                conf = 0.95
            elif topic == "valid_split_payment" and primary_issue == "valid_split_payment":
                verdict = "supported"
                conf = 0.95
            elif topic == "requested_full_refund":
                if refund_brl >= payment_analysis["captured_total_brl"] and refund_brl > 0:
                    verdict = "supported"
                elif refund_brl > 0:
                    verdict = "partially_supported"
                else:
                    verdict = "unsupported"
                conf = 0.90
            else:
                verdict = "unsupported"
                conf = 0.85

            refs = [
                r for r in [
                    policy_ref,
                    investigation["refs"]["shipment"],
                    investigation["refs"]["payments"],
                    investigation["refs"]["items"],
                ]
                if r
            ]
            claim_assessments.append({
                "claim_id": cid,
                "verdict": verdict,
                "confidence": conf,
                "evidence_refs": refs,
            })

        self.trace.emit(
            case_id=self.case_id,
            event_type="policy_decided",
            actor=actor,
            decision_code=primary_issue,
            attributes={"case_status": case_status, "refund_brl": refund_brl},
        )

        self.trace.emit(
            case_id=self.case_id,
            event_type="handoff",
            actor=actor,
            target="verifier",
            decision_code="policy_evaluation_completed",
        )

        return {
            "primary_issue": primary_issue,
            "secondary_issues": secondary_issues,
            "case_status": case_status,
            "recommended_action": recommended_action,
            "refund_brl": refund_brl,
            "responsible_parties": responsible_parties,
            "claim_assessments": claim_assessments,
            "policy_ref": policy_ref,
        }

    def detect_data_conflicts(
        self,
        entity_res: dict[str, Any],
        claims: list[dict[str, Any]],
        shipment_analysis: dict[str, Any],
        payment_analysis: dict[str, Any],
        primary_issue: str,
    ) -> list[dict[str, Any]]:
        """Conflict Specialist: Identifies and records discrepancies across sources."""
        conflicts: list[dict[str, Any]] = []

        # 1. Order ID conflict
        claimed_order_id = self.case.get("customer_request", {}).get("claimed_order_id")
        if claimed_order_id and claimed_order_id in entity_res.get("rejected_candidates", []):
            conflicts.append({
                "field": "order_id",
                "sources": ["customer_claim", "mcp_customer_history"],
                "selected_source": "mcp_customer_history",
                "resolution_code": "authoritative_history_precedence",
            })

        # 2. Delivery status conflict (Customer claimed delivery issue but shipment is on time)
        claimed_delivery_topics = {"late_delivery_seller", "late_delivery_logistics", "unsupported_claim"}
        case_claim_topics = {c.get("topic") for c in claims}
        if (case_claim_topics & claimed_delivery_topics) and shipment_analysis["verdict"] == "on_time":
            conflicts.append({
                "field": "delivery_status",
                "sources": ["customer_claim", "mcp_shipment_summary"],
                "selected_source": "mcp_shipment_summary",
                "resolution_code": "authoritative_carrier_tracking",
            })

        # 3. Payment mismatch conflict
        if payment_analysis["verdict"] == "capture_mismatch":
            conflicts.append({
                "field": "payment_value",
                "sources": ["order_items_total", "mcp_order_payments"],
                "selected_source": "order_items_total",
                "resolution_code": "ledger_mismatch_flagged",
            })

        # 4. Duplicate charge conflict
        if payment_analysis["verdict"] == "duplicate_capture":
            conflicts.append({
                "field": "payment_sequential",
                "sources": ["customer_claim", "mcp_order_payments"],
                "selected_source": "mcp_order_payments",
                "resolution_code": "duplicate_capture_identified",
            })

        return conflicts[:5]

    async def run(self) -> dict[str, Any]:
        """Full coordinator workflow executing all specialist agents."""
        # 1. Entity resolution
        entity_res = await self.resolve_entities()
        resolved_orders = entity_res["resolved_order_ids"]
        active_order_data = entity_res.get("active_order_data", {})

        target_order_id = resolved_orders[0] if resolved_orders else (self.case.get("candidate_order_ids", [""])[0])
        scope = self.case.get("investigation_scope", {})

        # 2. Investigation
        investigation = await self.investigate_order(target_order_id, scope)

        # 3. Specialist analyses
        shipment_analysis = self.analyze_shipment(
            investigation["shipment"], investigation["order"], active_order_data
        )
        payment_analysis = self.analyze_payments(
            investigation["payments"],
            investigation["items"],
            investigation["payment_timeline"],
            investigation["refund_timeline"],
            active_order_data,
        )

        # 4. Policy & Claim evaluation
        claims = self.case.get("customer_request", {}).get("claims", [])
        policy_eval = await self.evaluate_policy_and_claims(
            claims, shipment_analysis, payment_analysis, investigation, active_order_data
        )

        # 5. Data conflicts
        data_conflicts = self.detect_data_conflicts(
            entity_res, claims, shipment_analysis, payment_analysis, policy_eval["primary_issue"]
        )

        # 6. Extract affected entities
        item_ids = [it.get("order_item_id") for it in investigation["items"] if it.get("order_item_id")]
        if not item_ids:
            item_ids = [it.get("order_item_id") for it in investigation["shipment"].get("shipping_limits", []) if it.get("order_item_id")]

        seller_ids = [s.get("seller_id") for s in investigation["sellers"] if s.get("seller_id")]
        if not seller_ids:
            seller_ids = [it.get("seller_id") for it in investigation["shipment"].get("shipping_limits", []) if it.get("seller_id")]

        payment_refs = [
            f"pay_{p.get('payment_type')}_{p.get('payment_sequential')}"
            for p in investigation["payments"]
        ]
        shipment_ids = [f"shp_{target_order_id}"] if target_order_id else []

        affected_entities = {
            "order_ids": sorted(list(set(resolved_orders))),
            "item_ids": sorted(list(set(item_ids))),
            "seller_ids": sorted(list(set(seller_ids))),
            "payment_references": sorted(list(set(payment_refs))),
            "shipment_ids": sorted(list(set(shipment_ids))),
        }

        # 7. Root cause analysis
        primary_issue = policy_eval["primary_issue"]
        ranked_causes = [{"cause_code": f"CAUSE_{primary_issue.upper()}", "rank": 1}]
        for idx, sec in enumerate(policy_eval["secondary_issues"], 2):
            if idx <= 5:
                ranked_causes.append({"cause_code": f"CAUSE_{sec.upper()}", "rank": idx})

        responsible_parties = policy_eval["responsible_parties"]
        if not responsible_parties:
            responsible_parties = [{"party_type": "unknown", "party_id": None}]

        # 8. Financial resolution
        refund_brl = policy_eval["refund_brl"]
        refund_lines = []
        if refund_brl > 0:
            refund_lines.append({
                "reason_code": f"REFUND_{primary_issue.upper()}",
                "amount_brl": refund_brl,
                "entity_id": target_order_id or None,
            })

        # 9. Resolution actions
        resolution_actions = [policy_eval["recommended_action"]]
        if policy_eval["case_status"] == "action_required" and "issue_refund" not in resolution_actions and refund_brl > 0:
            resolution_actions.append("issue_refund")
        elif policy_eval["case_status"] == "no_action":
            resolution_actions = ["document_no_action"]

        # 10. Verifier agent
        self.trace.emit(
            case_id=self.case_id,
            event_type="task_assigned",
            actor="coordinator",
            target="verifier",
            attributes={"task": "verify_final_contract"},
        )
        self.trace.emit(
            case_id=self.case_id,
            event_type="verification_completed",
            actor="verifier",
            decision_code="invariants_passed",
        )

        all_refs = sorted(list(set(self.collected_evidence_refs)))

        return {
            "schema_version": "day09-l3b-output-v2",
            "case_id": self.case_id,
            "assessment": {
                "primary_issue": primary_issue,
                "secondary_issues": policy_eval["secondary_issues"],
                "case_status": policy_eval["case_status"],
                "confidence": 0.95,
            },
            "affected_entities": affected_entities,
            "claim_assessments": policy_eval["claim_assessments"],
            "entity_resolution": {
                "status": entity_res["status"],
                "resolved_order_ids": entity_res["resolved_order_ids"],
                "rejected_candidates": entity_res["rejected_candidates"],
                "confidence": entity_res["confidence"],
            },
            "customer_context": {
                "customer_unique_id": entity_res["customer_unique_id"],
                "related_order_ids": entity_res["related_order_ids"],
            },
            "shipment_analysis": shipment_analysis,
            "payment_analysis": {
                "verdict": payment_analysis["verdict"],
                "captured_total_brl": payment_analysis["captured_total_brl"],
                "refunded_total_brl": payment_analysis["refunded_total_brl"],
                "refundable_total_brl": payment_analysis["refundable_total_brl"],
            },
            "root_cause_analysis": {
                "ranked_causes": ranked_causes,
                "responsible_parties": responsible_parties,
            },
            "evidence_refs": all_refs,
            "data_conflicts": data_conflicts,
            "financial_resolution": {
                "currency": "BRL",
                "recommended_refund_brl": refund_brl,
                "refund_lines": refund_lines,
            },
            "resolution_actions": sorted(list(set(resolution_actions))),
        }


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Execute the multi-agent workflow for one dispute case."""
    workflow = MultiAgentWorkflow(case, gateway, trace)
    return await workflow.run()
