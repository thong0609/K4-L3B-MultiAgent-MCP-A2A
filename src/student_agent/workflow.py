from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import OUTPUT_SCHEMA_VERSION, llm
from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter

CALL_INTERVAL_SECONDS = 0.4  # pace MCP calls; the gateway drops bursts of ~80 calls in ~25s

COORDINATOR = "coordinator"
ENTITY = "entity-agent"
ORDER = "order-agent"
SHIPMENT = "shipment-agent"
PAYMENT = "payment-agent"
POLICY = "policy-agent"
CONFLICT = "conflict-resolver"
VERIFIER = "verifier"
REASONER = "reasoning-agent"

ROOT_CAUSES = {
    "canceled_order_paid": ["ORDER_CANCELED_AFTER_CAPTURE"],
    "unavailable_order_paid": ["ITEM_UNAVAILABLE_AFTER_CAPTURE"],
    "late_delivery_seller": ["SELLER_HANDOFF_AFTER_SHIPPING_LIMIT", "DELIVERY_AFTER_ESTIMATE"],
    "late_delivery_logistics": ["CARRIER_TRANSIT_DELAY", "DELIVERY_AFTER_ESTIMATE"],
    "valid_split_payment": ["SPLIT_PAYMENT_RECONCILED"],
    "payment_mismatch": ["PAYMENT_RECONCILIATION_MISMATCH"],
    "duplicate_charge": ["DUPLICATE_PAYMENT_CAPTURE"],
    "refund_pending": ["REFUND_IN_PROGRESS"],
    "refund_failed": ["REFUND_PROCESSING_FAILED"],
    "unsupported_claim": ["NO_DEFECT_IN_AUTHORITATIVE_TIMELINE"],
    "insufficient_evidence": ["MISSING_AUTHORITATIVE_EVIDENCE"],
}

PAYMENT_VERDICT = {
    "payment_mismatch": "capture_mismatch",
    "duplicate_charge": "duplicate_capture",
    "refund_pending": "refund_pending",
    "refund_failed": "refund_failed",
}

# Evidence domains that support each conclusion (beyond the always-needed order/customer/policy).
ISSUE_DOMAINS = {
    "canceled_order_paid": {"payment", "shipment"},
    "unavailable_order_paid": {"payment", "shipment", "seller", "item"},
    "late_delivery_seller": {"shipment", "item", "seller"},
    "late_delivery_logistics": {"shipment", "item"},
    "valid_split_payment": {"payment", "item"},
    "payment_mismatch": {"payment"},
    "duplicate_charge": {"payment", "item"},
    "refund_pending": {"payment", "refund"},
    "refund_failed": {"payment", "refund"},
    "unsupported_claim": {"shipment", "payment", "item"},
    "insufficient_evidence": set(),
}

SELLER_RESPONSIBLE = {"late_delivery_seller", "unavailable_order_paid"}


def _ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _money(value: Any) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class CaseContext:
    """Per-case evidence store: the only cache, never shared between cases."""

    case: dict[str, Any]
    gateway: EvidenceGateway
    trace: TraceWriter
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    refs_by_domain: dict[str, list[str]] = field(default_factory=dict)
    failed: set[str] = field(default_factory=set)

    @property
    def case_id(self) -> str:
        return str(self.case["case_id"])

    async def fetch(self, actor: str, tool: str, **arguments: str) -> Any | None:
        key = f"{tool}:{sorted(arguments.items())}"
        if key in self.evidence:
            return self.evidence[key]["data"]
        if key in self.failed:
            return None
        try:
            await asyncio.sleep(CALL_INTERVAL_SECONDS)
            response = await self.gateway.call(tool, case_id=self.case_id, **arguments)
        except (RuntimeError, ValueError):
            # One attempt only: missing evidence stays missing instead of being guessed.
            self.failed.add(key)
            self.trace.emit(
                case_id=self.case_id,
                event_type="tool_result_consumed",
                actor=actor,
                tool_name=tool,
                decision_code="EVIDENCE_UNAVAILABLE",
            )
            return None
        self.evidence[key] = response
        self.refs_by_domain.setdefault(response["domain"], []).append(response["evidence_ref"])
        self.trace.emit(
            case_id=self.case_id,
            event_type="tool_result_consumed",
            actor=actor,
            tool_name=tool,
            evidence_refs=[response["evidence_ref"]],
        )
        return response["data"]

    def assign(self, source: str, target: str, code: str) -> None:
        self.trace.emit(
            case_id=self.case_id,
            event_type="task_assigned",
            actor=source,
            target=target,
            decision_code=code,
        )

    def handoff(self, source: str, target: str, code: str, **attributes: Any) -> None:
        self.trace.emit(
            case_id=self.case_id,
            event_type="handoff",
            actor=source,
            target=target,
            decision_code=code,
            attributes=attributes or None,
        )

    def refs(self, domains: set[str]) -> list[str]:
        return [ref for domain in sorted(domains) for ref in self.refs_by_domain.get(domain, [])]


def _owner_index(rows: list[dict[str, Any]], moment: datetime | None) -> int | None:
    """Index of the order row whose purchase is the latest one not after `moment`."""
    best: tuple[datetime, int] | None = None
    for index, row in enumerate(rows):
        purchased = _ts(row.get("order_purchase_timestamp"))
        if moment is None or purchased is None or purchased > moment:
            continue
        if best is None or purchased > best[0]:
            best = (purchased, index)
    return None if best is None else best[1]


# ---------------------------------------------------------------- entity agent


async def entity_agent(ctx: CaseContext) -> dict[str, Any]:
    request = ctx.case["customer_request"]
    claimed = request.get("claimed_order_id")
    candidates = list(
        dict.fromkeys([*([claimed] if claimed else []), *ctx.case["candidate_order_ids"]])
    )
    hint = ctx.case.get("customer_unique_id_hint")

    history = (
        await ctx.fetch(ENTITY, "get_customer_history", customer_unique_id=hint) if hint else None
    )
    history_orders = list(history.get("orders", [])) if isinstance(history, dict) else []
    known = {row.get("order_id") for row in history_orders}

    resolved = [order_id for order_id in candidates if order_id in known]
    if not resolved and claimed:
        # History unavailable: fall back to verifying the claimed order directly.
        order = await ctx.fetch(ENTITY, "get_order", order_id=claimed)
        if isinstance(order, dict) and order.get("order_id") == claimed:
            resolved = [claimed]
    rejected = [order_id for order_id in candidates if order_id not in resolved]

    if len(resolved) == 1:
        status, confidence = "resolved", 0.95 if history_orders else 0.75
    elif resolved:
        status, confidence = "ambiguous", 0.5
    else:
        status, confidence = "not_found", 0.3

    order_id = resolved[0] if resolved else None
    rows = [row for row in history_orders if row.get("order_id") == order_id]
    opened = _ts(ctx.case.get("opened_at"))
    selected = _owner_index(rows, opened)
    if selected is None and rows:
        selected = 0

    return {
        "status": status,
        "confidence": confidence,
        "order_id": order_id,
        "resolved": resolved[:1] if status == "resolved" else resolved,
        "rejected": rejected,
        "customer_unique_id": history.get("customer_unique_id")
        if isinstance(history, dict)
        else None,
        "related_order_ids": sorted(
            {row.get("order_id") for row in history_orders if row.get("order_id")}
        ),
        "rows": rows,
        "selected": selected,
    }


# ------------------------------------------------------------ specialist agents


async def order_agent(ctx: CaseContext, entity: dict[str, Any]) -> dict[str, Any]:
    order_id = entity["order_id"]
    order = await ctx.fetch(ORDER, "get_order", order_id=order_id)
    items = await ctx.fetch(ORDER, "get_order_items", order_id=order_id) or []
    products = await ctx.fetch(ORDER, "get_product_context", order_id=order_id) or []
    rows, selected = entity["rows"], entity["selected"]
    own_items = [
        item
        for item in items
        if selected is None or _owner_index(rows, _ts(item.get("shipping_limit_date"))) == selected
    ] or items[:1]
    return {
        "order": order if isinstance(order, dict) else None,
        "items": own_items,
        "product_ids": sorted({p.get("product_id") for p in products if p.get("product_id")}),
    }


async def shipment_agent(
    ctx: CaseContext, entity: dict[str, Any], row: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, Any]:
    summary = await ctx.fetch(SHIPMENT, "get_shipment_summary", order_id=entity["order_id"]) or {}
    rows, selected = entity["rows"], entity["selected"]
    events = [
        event
        for event in summary.get("events", [])
        if _owner_index(rows, _ts(event.get("event_at"))) == selected
    ]
    carrier = _ts(row.get("order_delivered_carrier_date"))
    delivered = _ts(row.get("order_delivered_customer_date"))
    estimated = _ts(row.get("order_estimated_delivery_date"))
    late_sellers = sorted(
        {
            item["seller_id"]
            for item in items
            if carrier
            and _ts(item.get("shipping_limit_date"))
            and carrier > _ts(item["shipping_limit_date"])
        }
    )
    late_event_actors = {
        event.get("actor")
        for event in events
        if event.get("event_type") == "delivered_late" and event.get("status") == "confirmed"
    }
    status = row.get("order_status")
    if status in {"canceled", "unavailable"} or delivered is None:
        verdict = "insufficient_evidence"
    elif estimated and delivered > estimated:
        verdict = "seller_delay" if late_sellers else "logistics_delay"
        expected_actor = "seller" if late_sellers else "logistics_provider"
        if late_event_actors and expected_actor not in late_event_actors:
            verdict = "conflicting"
    else:
        verdict = "on_time"
    return {
        "verdict": verdict,
        "late_seller_ids": late_sellers if verdict == "seller_delay" else [],
        "timeline_complete": bool(carrier and delivered and estimated),
        "summary": summary,
        "events": events,
    }


async def payment_agent(
    ctx: CaseContext, entity: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, Any]:
    order_id = entity["order_id"]
    timeline = await ctx.fetch(PAYMENT, "get_payment_timeline", order_id=order_id) or {}
    refunds = await ctx.fetch(PAYMENT, "get_refund_timeline", order_id=order_id) or {}
    rows, selected = entity["rows"], entity["selected"]

    def mine(event: dict[str, Any]) -> bool:
        return _owner_index(rows, _ts(event.get("event_at"))) == selected

    all_events = timeline.get("events", [])
    captures_all = [e for e in all_events if e.get("event_type") == "captured"]
    payments = timeline.get("payments", [])
    own_payments = [
        payment for payment, capture in zip(payments, captures_all, strict=False) if mine(capture)
    ]
    events = [e for e in all_events if mine(e)]
    captures = [
        e for e in events if e.get("event_type") == "captured" and e.get("status") == "confirmed"
    ]
    refund_events = [e for e in refunds.get("events", []) if mine(e)]

    captured = round(sum(_money(e.get("amount_brl")) for e in captures), 2)
    refunded = round(
        sum(
            _money(e.get("amount_brl"))
            for e in refund_events
            if e.get("status") in {"completed", "succeeded", "refunded", "confirmed"}
        ),
        2,
    )
    expected = round(sum(_money(i.get("price")) + _money(i.get("freight_value")) for i in items), 2)
    amounts = [_money(e.get("amount_brl")) for e in captures]
    return {
        "captured": captured,
        "refunded": refunded,
        "refundable": max(round(captured - refunded, 2), 0.0),
        "expected": expected,
        "mismatch": any(
            e.get("event_type") == "reconciliation_mismatch" and e.get("status") != "resolved"
            for e in events
        ),
        "duplicate": len(amounts) > 1 and len(set(amounts)) < len(amounts) and captured > expected,
        "split": len({p.get("payment_type") for p in own_payments}) > 1
        and abs(captured - expected) < 0.01,
        "refund_status": {e.get("status") for e in refund_events},
        "events": events,
        "refund_events": refund_events,
        "payment_types": [p.get("payment_type") for p in own_payments],
        "payment_refs": list(
            dict.fromkeys(f"{order_id}:{p.get('payment_sequential')}" for p in own_payments)
        ),
    }


def classify(row: dict[str, Any], shipment: dict[str, Any], payment: dict[str, Any]) -> str:
    status = row.get("order_status")
    if status == "canceled" and payment["captured"] > 0:
        return "canceled_order_paid"
    if status == "unavailable" and payment["captured"] > 0:
        return "unavailable_order_paid"
    if "failed" in payment["refund_status"]:
        return "refund_failed"
    if "pending" in payment["refund_status"]:
        return "refund_pending"
    if payment["mismatch"]:
        return "payment_mismatch"
    if payment["duplicate"]:
        return "duplicate_charge"
    if shipment["verdict"] == "seller_delay":
        return "late_delivery_seller"
    if shipment["verdict"] == "logistics_delay":
        return "late_delivery_logistics"
    if payment["split"]:
        return "valid_split_payment"
    return "unsupported_claim"


# ---------------------------------------------------------------- policy agent


async def policy_agent(ctx: CaseContext, issue: str, seller_ids: list[str]) -> dict[str, Any]:
    policy = await ctx.fetch(POLICY, "get_policy", policy_version=ctx.case["policy_version"]) or {}
    rule = (policy.get("rules") or {}).get(issue)
    if rule is None:
        return {"rule": None}
    parties = []
    for party in rule.get("responsible_parties", []):
        party_id = party.get("party_id")
        if party.get("party_type") == "seller":
            # The policy template carries a generic seller; responsibility follows case evidence.
            party_id = seller_ids[0] if seller_ids else None
        parties.append({"party_type": party.get("party_type", "unknown"), "party_id": party_id})
    return {
        "rule": rule,
        "case_status": rule.get("case_status"),
        "action": rule.get("recommended_action"),
        "refund": _money(rule.get("refund_brl")),
        "parties": parties,
    }


# ----------------------------------------------------------- conflict resolver


def conflict_resolver(
    entity: dict[str, Any], order: dict[str, Any] | None, summary: dict[str, Any]
) -> list[dict[str, Any]]:
    row = entity["rows"][entity["selected"]] if entity["selected"] is not None else None
    if row is None:
        return []
    conflicts: list[dict[str, Any]] = []
    checks = [
        ("order_purchase_timestamp", order, "order_purchase_timestamp", "get_order"),
        ("order_status", order, "order_status", "get_order"),
        ("order_delivered_customer_date", order, "order_delivered_customer_date", "get_order"),
        ("order_estimated_delivery_date", order, "order_estimated_delivery_date", "get_order"),
        ("order_delivered_customer_date", summary, "delivered_customer_at", "get_shipment_summary"),
    ]
    for field_name, source, source_field, tool in checks:
        if not isinstance(source, dict) or source_field not in source:
            continue
        if source.get(source_field) != row.get(field_name):
            conflicts.append(
                {
                    "field": field_name,
                    "sources": [tool, "get_customer_history"],
                    "selected_source": "get_customer_history",
                    "resolution_code": "LATEST_ROW_BEFORE_CASE_OPENED",
                }
            )
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for conflict in conflicts:
        unique.setdefault((conflict["field"], conflict["sources"][0]), conflict)
    return list(unique.values())[:5]


# ----------------------------------------------------------------- verifier


def verifier(output: dict[str, Any], captured: float) -> list[str]:
    problems: list[str] = []
    status = output["assessment"]["case_status"]
    refund = output["financial_resolution"]["recommended_refund_brl"]
    if status == "no_action" and refund > 0:
        problems.append("NO_ACTION_WITH_REFUND")
    if refund > captured + 0.01 and captured > 0:
        problems.append("REFUND_EXCEEDS_CAPTURE")
    if (
        abs(
            sum(line["amount_brl"] for line in output["financial_resolution"]["refund_lines"])
            - refund
        )
        > 0.01
    ):
        problems.append("REFUND_LINES_MISMATCH")
    sellers = {
        p["party_id"]
        for p in output["root_cause_analysis"]["responsible_parties"]
        if p["party_type"] == "seller"
    }
    if sellers - set(output["affected_entities"]["seller_ids"]):
        problems.append("SELLER_OUT_OF_SCOPE")
    if not output["evidence_refs"]:
        problems.append("NO_EVIDENCE")
    return problems


# --------------------------------------------------------------- coordinator


def _claim_assessments(
    case: dict[str, Any],
    issue: str,
    refund: float,
    captured: float,
    refs: list[str],
    confidence: float,
    llm_verdicts: dict[str, str],
) -> list[dict[str, Any]]:
    result = []
    for claim in case["customer_request"].get("claims", [])[:5]:
        topic = claim.get("topic")
        matches = topic == issue and issue not in {"unsupported_claim", "insufficient_evidence"}
        allowed = (
            {"supported", "partially_supported"}
            if matches
            else {"unsupported", "insufficient_evidence"}
        )
        if topic != "requested_full_refund" and llm_verdicts.get(claim["claim_id"]) in allowed:
            # Refund coverage is arithmetic from policy; the model only refines topical verdicts
            # and is ignored when its verdict contradicts the verified primary issue.
            verdict = llm_verdicts[claim["claim_id"]]
        elif topic == "requested_full_refund":
            if refund <= 0:
                verdict = "unsupported"
            elif refund + 0.01 >= captured:
                verdict = "supported"
            else:
                verdict = "partially_supported"
        elif issue == "insufficient_evidence":
            verdict = "insufficient_evidence"
        elif issue == "unsupported_claim":
            verdict = "unsupported"
        else:
            verdict = "supported" if topic == issue else "unsupported"
        result.append(
            {
                "claim_id": claim["claim_id"],
                "verdict": verdict,
                "confidence": confidence,
                "evidence_refs": refs[:30],
            }
        )
    return result


def _insufficient(ctx: CaseContext, entity: dict[str, Any]) -> dict[str, Any]:
    refs = ctx.refs({"customer", "order"})
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": ctx.case_id,
        "assessment": {
            "primary_issue": "insufficient_evidence",
            "secondary_issues": [],
            "case_status": "needs_investigation",
            "confidence": 0.3,
        },
        "affected_entities": {
            "order_ids": entity["resolved"],
            "item_ids": [],
            "seller_ids": [],
            "payment_references": [],
            "shipment_ids": [],
        },
        "entity_resolution": {
            "status": entity["status"],
            "resolved_order_ids": entity["resolved"],
            "rejected_candidates": entity["rejected"],
            "confidence": entity["confidence"],
        },
        "customer_context": {
            "customer_unique_id": entity["customer_unique_id"],
            "related_order_ids": entity["related_order_ids"],
        },
        "shipment_analysis": {
            "verdict": "insufficient_evidence",
            "late_seller_ids": [],
            "timeline_complete": False,
        },
        "payment_analysis": {
            "verdict": "insufficient_evidence",
            "captured_total_brl": None,
            "refunded_total_brl": None,
            "refundable_total_brl": None,
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": "MISSING_AUTHORITATIVE_EVIDENCE", "rank": 1}],
            "responsible_parties": [{"party_type": "unknown", "party_id": None}],
        },
        "evidence_refs": refs,
        "data_conflicts": [],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": 0.0,
            "refund_lines": [],
        },
        "resolution_actions": ["escalate_manual_review"],
    }


# ----------------------------------------------------------- reasoning agent


def _facts(
    case: dict[str, Any],
    row: dict[str, Any],
    items: list[dict[str, Any]],
    shipment: dict[str, Any],
    payment: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Normalized evidence for the authoritative order version only (no customer free text)."""
    carrier = _ts(row.get("order_delivered_carrier_date"))
    delivered = _ts(row.get("order_delivered_customer_date"))
    estimated = _ts(row.get("order_estimated_delivery_date"))
    return {
        "case_opened_at": case.get("opened_at"),
        "customer_claims": [
            {"claim_id": c.get("claim_id"), "topic": c.get("topic")}
            for c in case["customer_request"].get("claims", [])
        ],
        "order": {
            key: row.get(key)
            for key in (
                "order_status",
                "order_purchase_timestamp",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            )
        },
        "items": [
            {
                "seller_id": i.get("seller_id"),
                "shipping_limit_date": i.get("shipping_limit_date"),
                "price": _money(i.get("price")),
                "freight_value": _money(i.get("freight_value")),
            }
            for i in items
        ],
        "derived": {
            "carrier_handoff_after_shipping_limit": any(
                carrier and limit and carrier > limit
                for limit in (_ts(i.get("shipping_limit_date")) for i in items)
            ),
            "delivered_after_estimate": bool(delivered and estimated and delivered > estimated),
            "expected_item_total_brl": payment["expected"],
            "captured_total_brl": payment["captured"],
        },
        "shipment_events": [
            {k: e.get(k) for k in ("event_at", "event_type", "actor", "status")}
            for e in shipment["events"]
        ],
        "payment_types": payment["payment_types"],
        "payment_events": [
            {k: e.get(k) for k in ("event_at", "event_type", "amount_brl", "status")}
            for e in payment["events"]
        ],
        "refund_events": [
            {k: e.get(k) for k in ("event_at", "event_type", "amount_brl", "status")}
            for e in payment["refund_events"]
        ],
        "policy_refund_brl_by_issue": {
            name: _money(rule.get("refund_brl")) for name, rule in rules.items()
        },
    }


async def reasoning_agent(
    ctx: CaseContext, facts: dict[str, Any], rule_issue: str
) -> tuple[str, dict[str, Any] | None, str]:
    """Ask Qwen3-8B for the assessment, then cross-check it against the deterministic signals.

    Returns (issue, accepted_llm_answer_or_None, decision_code).
    """
    if not llm.enabled():
        return rule_issue, None, "RULES_ONLY"
    claim_ids = [c["claim_id"] for c in facts["customer_claims"] if c.get("claim_id")]
    ctx.assign(COORDINATOR, REASONER, "ASSESS_CASE")
    # Run generation off the event loop so the MCP session stays responsive.
    answer = await asyncio.to_thread(llm.assess, facts, claim_ids)
    if answer is None:
        issue, accepted, code = rule_issue, None, "LLM_INVALID_RULES_FALLBACK"
    elif answer["primary_issue"] == rule_issue:
        issue, accepted, code = rule_issue, answer, "LLM_CONFIRMED"
    else:
        issue, accepted, code = rule_issue, None, "LLM_OVERRULED_BY_EVIDENCE"
    ctx.handoff(
        REASONER,
        VERIFIER,
        code,
        model=llm.model_name()[-80:],
        llm_issue=answer["primary_issue"] if answer else None,
        rule_issue=rule_issue,
    )
    return issue, accepted, code


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Coordinator: entity resolution, specialist fan-out, policy, conflicts, verification."""
    ctx = CaseContext(case, gateway, trace)

    ctx.assign(COORDINATOR, ENTITY, "RESOLVE_ENTITY")
    entity = await entity_agent(ctx)
    ctx.handoff(
        ENTITY,
        COORDINATOR,
        f"ENTITY_{entity['status'].upper()}",
        resolved=len(entity["resolved"]),
        rejected=len(entity["rejected"]),
    )
    if entity["status"] != "resolved" or entity["selected"] is None:
        output = _insufficient(ctx, entity)
        trace.emit(
            case_id=ctx.case_id,
            event_type="verification_completed",
            actor=VERIFIER,
            decision_code="ESCALATED",
        )
        return output
    row = entity["rows"][entity["selected"]]

    ctx.assign(COORDINATOR, ORDER, "COLLECT_ORDER_CONTEXT")
    order = await order_agent(ctx, entity)
    ctx.handoff(ORDER, COORDINATOR, "ORDER_CONTEXT_READY", items=len(order["items"]))
    items = order["items"]
    seller_ids = sorted({item["seller_id"] for item in items if item.get("seller_id")})

    ctx.assign(COORDINATOR, SHIPMENT, "ANALYZE_SHIPMENT")
    shipment = await shipment_agent(ctx, entity, row, items)
    ctx.handoff(SHIPMENT, COORDINATOR, f"SHIPMENT_{shipment['verdict'].upper()}")

    ctx.assign(COORDINATOR, PAYMENT, "ANALYZE_PAYMENT")
    payment = await payment_agent(ctx, entity, items)
    ctx.handoff(PAYMENT, COORDINATOR, "PAYMENT_ANALYZED", captured=payment["captured"])

    rule_issue = classify(row, shipment, payment)
    policy_doc = await ctx.fetch(POLICY, "get_policy", policy_version=case["policy_version"]) or {}
    facts = _facts(case, row, items, shipment, payment, policy_doc.get("rules") or {})
    issue, llm_answer, reasoning_code = await reasoning_agent(ctx, facts, rule_issue)
    if issue in SELLER_RESPONSIBLE:
        await ctx.fetch(ORDER, "get_sellers", order_id=entity["order_id"])

    ctx.assign(COORDINATOR, POLICY, "APPLY_POLICY")
    policy = await policy_agent(ctx, issue, seller_ids)
    if policy["rule"] is None:
        output = _insufficient(ctx, entity)
        trace.emit(
            case_id=ctx.case_id,
            event_type="verification_completed",
            actor=VERIFIER,
            decision_code="POLICY_MISSING",
        )
        return output
    trace.emit(
        case_id=ctx.case_id,
        event_type="policy_decided",
        actor=POLICY,
        decision_code=issue.upper(),
        evidence_refs=ctx.refs({"policy"}) or None,
        attributes={"refund_brl": policy["refund"], "action": policy["action"]},
    )
    ctx.handoff(POLICY, CONFLICT, "POLICY_APPLIED")

    conflicts = conflict_resolver(entity, order["order"], shipment["summary"])
    ctx.handoff(CONFLICT, VERIFIER, "CONFLICTS_RESOLVED", conflicts=len(conflicts))

    domains = {"order", "customer", "policy", "product"} | ISSUE_DOMAINS[issue]
    evidence_refs = ctx.refs(domains)[:30]
    confidence = 0.9 if issue != "unsupported_claim" else 0.8
    if llm_answer is not None:
        confidence = min(max(llm_answer["confidence"], 0.6), 0.95)
    elif reasoning_code == "LLM_OVERRULED_BY_EVIDENCE":
        confidence = 0.7
    refund = policy["refund"]
    refund_lines = (
        [{"reason_code": policy["action"], "amount_brl": refund, "entity_id": entity["order_id"]}]
        if refund > 0
        else []
    )
    output = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": ctx.case_id,
        "assessment": {
            "primary_issue": issue,
            "secondary_issues": [],
            "case_status": policy["case_status"],
            "confidence": confidence,
        },
        "affected_entities": {
            "order_ids": [entity["order_id"]],
            "item_ids": sorted(
                {item["order_item_id"] for item in items if item.get("order_item_id")}
            ),
            "seller_ids": seller_ids,
            "payment_references": payment["payment_refs"],
            "shipment_ids": [],
        },
        "claim_assessments": _claim_assessments(
            case,
            issue,
            refund,
            payment["captured"],
            evidence_refs,
            confidence,
            llm_answer["claim_verdicts"] if llm_answer else {},
        ),
        "entity_resolution": {
            "status": entity["status"],
            "resolved_order_ids": entity["resolved"],
            "rejected_candidates": entity["rejected"],
            "confidence": entity["confidence"],
        },
        "customer_context": {
            "customer_unique_id": entity["customer_unique_id"],
            "related_order_ids": entity["related_order_ids"],
        },
        "shipment_analysis": {
            "verdict": shipment["verdict"],
            "late_seller_ids": shipment["late_seller_ids"],
            "timeline_complete": shipment["timeline_complete"],
        },
        "payment_analysis": {
            "verdict": PAYMENT_VERDICT.get(
                issue, "refunded" if payment["refunded"] > 0 else "reconciled"
            ),
            "captured_total_brl": payment["captured"],
            "refunded_total_brl": payment["refunded"],
            "refundable_total_brl": payment["refundable"],
        },
        "root_cause_analysis": {
            "ranked_causes": [
                {"cause_code": code, "rank": rank}
                for rank, code in enumerate(ROOT_CAUSES[issue], 1)
            ],
            "responsible_parties": policy["parties"],
        },
        "evidence_refs": evidence_refs,
        "data_conflicts": conflicts,
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": refund,
            "refund_lines": refund_lines,
        },
        "resolution_actions": [policy["action"]] if policy["action"] else [],
    }

    problems = verifier(output, payment["captured"])
    if problems:
        output["assessment"]["confidence"] = min(confidence, 0.5)
    trace.emit(
        case_id=ctx.case_id,
        event_type="verification_completed",
        actor=VERIFIER,
        decision_code="PASSED" if not problems else "FLAGGED",
        evidence_refs=evidence_refs[:20],
        attributes={"problems": ",".join(problems) or None},
    )
    return output
