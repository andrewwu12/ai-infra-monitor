#!/usr/bin/env python3
"""
AI Infrastructure Capex Cycle Monitor - Data Collector

Gathers multi-layer data to assess where we are in the AI infrastructure
investment cycle. Combines live Treasury yield data with quarterly earnings
data from hyperscalers and infrastructure suppliers.

Usage: python3 collect_data.py
Output: data/latest.json + append to data/history.jsonl
"""

import csv
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
LATEST_JSON = DATA_DIR / "latest.json"
HISTORY_JSONL = DATA_DIR / "history.jsonl"

TREASURY_URL_2026 = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/daily-treasury-rates.csv/2026/all?"
    "type=daily_treasury_yield_curve&field_tdr_date_value=2026&page&_format=csv"
)
TREASURY_URL_2025 = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/daily-treasury-rates.csv/2025/all?"
    "type=daily_treasury_yield_curve&field_tdr_date_value=2025&page&_format=csv"
)

# Column names in Treasury CSV (used for parsing)
MATURITY_COLS = {
    "1 Mo": "1m",
    "2 Mo": "2m",
    "3 Mo": "3m",
    "6 Mo": "6m",
    "1 Yr": "1y",
    "2 Yr": "2y",
    "3 Yr": "3y",
    "5 Yr": "5y",
    "7 Yr": "7y",
    "10 Yr": "10y",
    "20 Yr": "20y",
    "30 Yr": "30y",
}


# ---------------------------------------------------------------------------
# Layer 1: Hyperscaler Capex (Quarterly, from latest earnings)
# ---------------------------------------------------------------------------
# Sources: Q4 2025 / FY2025 earnings calls and 10-Ks
# MSFT: FY Q2 2025 (Dec quarter) reported Jan 2025; FY Q2 2026 (Dec quarter) reported Jan 2026
# AMZN: Q4 2025 reported Feb 2026
# GOOG: Q4 2025 reported Feb 2026
# META: Q4 2025 reported Jan 2026
#
# All dollar figures in $B USD. Approximate values from public filings.

HYPERSCALER_CAPEX = {
    "last_updated": "2026-02-20",
    "companies": {
        "MSFT": {
            "name": "Microsoft",
            "fiscal_quarter": "FY2026 Q2 (Dec 2025)",
            "quarterly_capex_B": 22.6,  # ~$22.6B in Dec quarter (cloud + AI build)
            "capex_ttm_B": 78.0,  # trailing 4 quarters
            "capex_yoy_growth_pct": 68.0,  # massive AI-driven capex ramp
            "capex_pct_revenue": 29.5,  # capex / revenue
            "capex_pct_opcf": 62.0,  # capex / operating cash flow
            "operating_margin_pct": 44.6,  # operating income / revenue
            "operating_margin_yoy_chg_ppt": -0.8,  # slight compression from AI spend
            "revenue_growth_yoy_pct": 12.3,
            "notes": "Satya guided $80B+ FY2026 capex. AI demand exceeding supply.",
        },
        "AMZN": {
            "name": "Amazon",
            "fiscal_quarter": "Q4 2025",
            "quarterly_capex_B": 26.3,  # record quarter for AWS buildout
            "capex_ttm_B": 86.0,  # trailing 4 quarters
            "capex_yoy_growth_pct": 42.0,
            "capex_pct_revenue": 12.8,
            "capex_pct_opcf": 68.0,
            "operating_margin_pct": 11.2,
            "operating_margin_yoy_chg_ppt": 2.1,  # improving from retail efficiency
            "revenue_growth_yoy_pct": 10.5,
            "notes": "Jassy guided $100B+ 2026 capex. AWS backlog $190B+.",
        },
        "GOOG": {
            "name": "Alphabet",
            "fiscal_quarter": "Q4 2025",
            "quarterly_capex_B": 14.3,
            "capex_ttm_B": 52.5,
            "capex_yoy_growth_pct": 55.0,
            "capex_pct_revenue": 13.8,
            "capex_pct_opcf": 46.0,
            "operating_margin_pct": 32.0,
            "operating_margin_yoy_chg_ppt": 1.5,
            "revenue_growth_yoy_pct": 12.0,
            "notes": "Pichai: AI is driving search + cloud. Guided $75B 2026 capex.",
        },
        "META": {
            "name": "Meta Platforms",
            "fiscal_quarter": "Q4 2025",
            "quarterly_capex_B": 15.8,
            "capex_ttm_B": 51.0,
            "capex_yoy_growth_pct": 36.0,
            "capex_pct_revenue": 28.5,
            "capex_pct_opcf": 55.0,
            "operating_margin_pct": 41.0,
            "operating_margin_yoy_chg_ppt": 2.0,
            "revenue_growth_yoy_pct": 21.0,
            "notes": "Zuckerberg guided $60-65B 2026 capex. Llama adoption growing.",
        },
    },
}


# ---------------------------------------------------------------------------
# Layer 2: Infrastructure Supplier Tightness (NVDA primarily)
# ---------------------------------------------------------------------------
# Source: NVDA FY2026 Q4 (Jan 2026) earnings reported Feb 2026

INFRA_SUPPLIERS = {
    "last_updated": "2026-02-20",
    "companies": {
        "NVDA": {
            "name": "NVIDIA",
            "fiscal_quarter": "FY2026 Q4 (Jan 2026)",
            "data_center_revenue_B": 39.3,  # ~$39B quarterly data center revenue
            "data_center_revenue_ttm_B": 135.0,
            "dc_revenue_yoy_growth_pct": 73.0,  # slowing from 200%+ but still massive
            "dc_revenue_qoq_growth_pct": 12.0,
            "gross_margin_pct": 73.0,  # normalizing from 75%+ peak
            "gross_margin_yoy_chg_ppt": -3.5,  # slight compression from Blackwell ramp
            "operating_margin_pct": 62.0,
            "inventory_days": 85,  # elevated for Blackwell ramp
            "book_to_bill": 1.4,  # demand still exceeding supply
            "backlog_growth_pct": 50.0,  # substantial backlog
            "notes": "Blackwell fully ramped. Demand visibility through FY2027. Sovereign AI a new driver.",
        },
    },
    "supply_chain": {
        "tsmc_utilization_pct": 95,  # near full utilization
        "copackaging_lead_time_weeks": 52,  # still extended
        "hbm_supply_constraint": "moderate",  # Samsung ramping, SK Hynix expanding
        "power_constraint_severity": "high",  # data center power becoming binding
        "notes": "Power availability and cooling are emerging as primary bottlenecks in 2026.",
    },
}


# ---------------------------------------------------------------------------
# Layer 3: Monetization Signals
# ---------------------------------------------------------------------------
# Source: Respective Q4 2025 / recent earnings

MONETIZATION = {
    "last_updated": "2026-02-20",
    "cloud_growth": {
        "MSFT_Azure": {
            "growth_yoy_pct": 31.0,  # Azure growth rate (constant currency)
            "ai_contribution_ppt": 13.0,  # ~13 points from AI services
            "notes": "AI contributing meaningfully to Azure growth acceleration.",
        },
        "AMZN_AWS": {
            "growth_yoy_pct": 19.5,
            "ai_contribution_ppt": 6.0,
            "notes": "AWS reaccelerating. Bedrock + custom silicon gaining traction.",
        },
        "GOOG_Cloud": {
            "growth_yoy_pct": 30.0,
            "ai_contribution_ppt": 10.0,
            "notes": "GCP growing strongly. Vertex AI and Gemini driving new workloads.",
        },
    },
    "ai_revenue_signals": {
        "msft_copilot_seats_M": 5.0,  # ~5M paid Copilot seats
        "openai_arr_B": 11.6,  # OpenAI annual revenue run rate
        "anthropic_arr_B": 4.0,  # Anthropic estimated ARR
        "enterprise_ai_adoption_pct": 45,  # % of Fortune 500 with AI in production
        "notes": "Enterprise AI spending accelerating but ROI measurement still nascent.",
    },
    "margin_signals": {
        "cloud_margin_trend": "expanding",  # cloud margins still improving
        "ai_incremental_margin_pct": 25,  # AI workloads lower margin than traditional cloud
        "inference_cost_decline_yoy_pct": -55,  # inference costs dropping rapidly
        "notes": "Inference cost declines helping adoption but compressing per-unit revenue.",
    },
}


# ---------------------------------------------------------------------------
# Layer 4: Macro Stress (hardcoded portions)
# ---------------------------------------------------------------------------
# Live yields fetched from Treasury; these are supplemental indicators

MACRO_HARDCODED = {
    "last_updated": "2026-02-20",
    "fed_funds_rate_pct": 4.00,  # Fed cut to 4.00-4.25 range
    "fed_dots_terminal_pct": 3.50,  # median dot for end of 2026
    "ig_credit_spread_bps": 85,  # investment grade OAS
    "hy_credit_spread_bps": 310,  # high yield OAS
    "ig_spread_6mo_chg_bps": -10,  # tightened slightly
    "hy_spread_6mo_chg_bps": -20,
    "vix": 16.5,
    "us_cpi_yoy_pct": 2.8,  # latest CPI reading
    "us_unemployment_pct": 4.1,
    "ism_manufacturing": 50.5,  # barely expansionary
    "data_center_power_cost_index": 115,  # indexed to 100 = 2024 avg
    "tech_labor_market": "tight",  # AI/ML talent still scarce
    "notes": "Macro environment benign but not easy. Rates higher-for-longer narrative persists.",
}


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------


def fetch_treasury_csv(url: str) -> list[dict]:
    """Fetch and parse Treasury yield curve CSV. Returns list of row dicts."""
    req = Request(url, headers={"User-Agent": "AIInfraMonitor/1.0"})
    try:
        with urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8")
    except (URLError, HTTPError) as e:
        print(f"  WARNING: Failed to fetch {url}: {e}", file=sys.stderr)
        return []

    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        rows.append(row)
    return rows


def parse_yield(val: str) -> float | None:
    """Parse a yield string like '4.08' to float, or None if missing."""
    if val is None:
        return None
    val = val.strip()
    if val == "" or val == "N/A":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def get_latest_yields(rows_2026: list[dict], rows_2025: list[dict]) -> dict:
    """Extract latest yields and compute spreads + YoY comparison."""
    if not rows_2026:
        return {"error": "No 2026 Treasury data available"}

    latest = rows_2026[0]  # first row is most recent date
    date_str = latest.get("Date", "unknown")

    maturities = {}
    for csv_col, short_name in MATURITY_COLS.items():
        val = parse_yield(latest.get(csv_col))
        if val is not None:
            maturities[short_name] = val

    # Key yields
    y10 = maturities.get("10y")
    y2 = maturities.get("2y")
    y3m = maturities.get("3m")
    y30 = maturities.get("30y")

    spreads = {}
    if y10 is not None and y2 is not None:
        spreads["10y_2y"] = round(y10 - y2, 2)
    if y10 is not None and y3m is not None:
        spreads["10y_3m"] = round(y10 - y3m, 2)
    if y30 is not None and y10 is not None:
        spreads["30y_10y"] = round(y30 - y10, 2)

    # YoY comparison: find the closest date ~1 year ago
    yoy = {}
    if rows_2025:
        # Try to find a date close to 1 year before the latest date
        try:
            latest_date = datetime.strptime(date_str, "%m/%d/%Y")
            target_date = latest_date.replace(year=latest_date.year - 1)
            best_row = None
            best_delta = None
            for row in rows_2025:
                try:
                    rd = datetime.strptime(row["Date"], "%m/%d/%Y")
                    delta = abs((rd - target_date).days)
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_row = row
                except (ValueError, KeyError):
                    continue

            if best_row is not None:
                yoy["comparison_date"] = best_row.get("Date", "unknown")
                y10_prev = parse_yield(best_row.get("10 Yr"))
                y2_prev = parse_yield(best_row.get("2 Yr"))
                if y10_prev is not None and y10 is not None:
                    yoy["10y_change_bps"] = round((y10 - y10_prev) * 100, 1)
                if y2_prev is not None and y2 is not None:
                    yoy["2y_change_bps"] = round((y2 - y2_prev) * 100, 1)
                # Previous spreads
                y3m_prev = parse_yield(best_row.get("3 Mo"))
                if y10_prev and y2_prev:
                    prev_10_2 = round(y10_prev - y2_prev, 2)
                    yoy["10y_2y_spread_prev"] = prev_10_2
                    if "10y_2y" in spreads:
                        yoy["10y_2y_spread_change"] = round(
                            spreads["10y_2y"] - prev_10_2, 2
                        )
        except (ValueError, KeyError):
            pass

    # Recent trend: last 5 trading days for 10Y
    trend_10y = []
    for row in rows_2026[:5]:
        val = parse_yield(row.get("10 Yr"))
        if val is not None:
            trend_10y.append({"date": row.get("Date", ""), "yield": val})

    return {
        "date": date_str,
        "maturities": maturities,
        "spreads": spreads,
        "yoy_comparison": yoy,
        "recent_trend_10y": trend_10y,
    }


# ---------------------------------------------------------------------------
# Scoring Functions
# ---------------------------------------------------------------------------


def clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


def compute_capex_acceleration_score(capex_data: dict) -> tuple[float, dict]:
    """
    Capex Acceleration Score (0-100):
    - Higher when YoY capex growth is large and accelerating
    - Higher when capex/revenue is expanding (companies committing more)
    - Moderated if capex/OCF is dangerously high (unsustainable)
    """
    companies = capex_data["companies"]
    yoy_growths = [c["capex_yoy_growth_pct"] for c in companies.values()]
    capex_rev_ratios = [c["capex_pct_revenue"] for c in companies.values()]
    capex_ocf_ratios = [c["capex_pct_opcf"] for c in companies.values()]

    avg_yoy = sum(yoy_growths) / len(yoy_growths)
    avg_capex_rev = sum(capex_rev_ratios) / len(capex_rev_ratios)
    avg_capex_ocf = sum(capex_ocf_ratios) / len(capex_ocf_ratios)

    total_ttm = sum(c["capex_ttm_B"] for c in companies.values())

    # YoY growth component: 50% growth = 50 score, 100% = 80, cap at 95
    growth_score = clamp(avg_yoy * 0.8 + 10, 0, 95)

    # Revenue commitment component: 15% = 50, 25% = 75, 35% = 90
    commitment_score = clamp((avg_capex_rev - 5) * 3, 0, 95)

    # Sustainability penalty: if capex/OCF > 70%, start penalizing
    sustainability_penalty = clamp((avg_capex_ocf - 70) * 1.5, 0, 30)

    score = clamp(
        0.50 * growth_score + 0.35 * commitment_score + 15 - sustainability_penalty
    )

    details = {
        "avg_yoy_growth_pct": round(avg_yoy, 1),
        "avg_capex_pct_revenue": round(avg_capex_rev, 1),
        "avg_capex_pct_opcf": round(avg_capex_ocf, 1),
        "total_capex_ttm_B": round(total_ttm, 1),
        "growth_score": round(growth_score, 1),
        "commitment_score": round(commitment_score, 1),
        "sustainability_penalty": round(sustainability_penalty, 1),
        "companies": {
            ticker: {
                "capex_ttm_B": c["capex_ttm_B"],
                "yoy_growth_pct": c["capex_yoy_growth_pct"],
                "capex_pct_revenue": c["capex_pct_revenue"],
            }
            for ticker, c in companies.items()
        },
    }
    return round(score, 1), details


def compute_bottleneck_power_score(supplier_data: dict) -> tuple[float, dict]:
    """
    Bottleneck Power Score (0-100):
    - Higher when NVDA revenue momentum is strong
    - Higher when gross margins are elevated (pricing power)
    - Higher when book-to-bill > 1 (demand > supply)
    - Higher when supply chain is tight
    """
    nvda = supplier_data["companies"]["NVDA"]
    sc = supplier_data["supply_chain"]

    # Revenue momentum: 50% YoY = 50, 100% = 75, 150% = 90
    rev_momentum = clamp(nvda["dc_revenue_yoy_growth_pct"] * 0.6 + 15, 0, 95)

    # Margin power: 70% GM = 70, 75% = 80, 65% = 55
    margin_power = clamp(nvda["gross_margin_pct"] * 1.1 - 10, 0, 95)

    # Demand excess: book-to-bill 1.0 = 40, 1.5 = 70, 2.0 = 90
    btb_score = clamp((nvda["book_to_bill"] - 0.8) * 60 + 20, 0, 95)

    # Supply tightness
    util = sc["tsmc_utilization_pct"]
    supply_tightness = clamp((util - 70) * 3, 0, 95)

    score = clamp(
        0.35 * rev_momentum
        + 0.20 * margin_power
        + 0.25 * btb_score
        + 0.20 * supply_tightness
    )

    details = {
        "nvda_dc_revenue_ttm_B": nvda["data_center_revenue_ttm_B"],
        "nvda_dc_yoy_growth_pct": nvda["dc_revenue_yoy_growth_pct"],
        "nvda_gross_margin_pct": nvda["gross_margin_pct"],
        "nvda_book_to_bill": nvda["book_to_bill"],
        "tsmc_utilization_pct": sc["tsmc_utilization_pct"],
        "power_constraint": sc["power_constraint_severity"],
        "rev_momentum_score": round(rev_momentum, 1),
        "margin_power_score": round(margin_power, 1),
        "btb_score": round(btb_score, 1),
        "supply_tightness_score": round(supply_tightness, 1),
    }
    return round(score, 1), details


def compute_monetization_score(monet_data: dict) -> tuple[float, dict]:
    """
    Monetization Realization Score (0-100):
    - Higher when cloud growth is accelerating (especially AI contribution)
    - Higher when AI revenue signals are strong
    - Penalized if margins deteriorating faster than revenue growing
    """
    cloud = monet_data["cloud_growth"]
    ai_rev = monet_data["ai_revenue_signals"]
    margins = monet_data["margin_signals"]

    # Cloud growth composite
    cloud_growths = [v["growth_yoy_pct"] for v in cloud.values()]
    ai_contributions = [v["ai_contribution_ppt"] for v in cloud.values()]
    avg_cloud_growth = sum(cloud_growths) / len(cloud_growths)
    avg_ai_contribution = sum(ai_contributions) / len(ai_contributions)

    # Cloud growth score: 20% = 50, 30% = 70, 40% = 85
    cloud_score = clamp(avg_cloud_growth * 2.0 + 10, 0, 95)

    # AI contribution score: 5ppt = 40, 10ppt = 60, 15ppt = 80
    ai_contrib_score = clamp(avg_ai_contribution * 4 + 20, 0, 95)

    # Enterprise adoption: 30% = 50, 50% = 70, 70% = 85
    adoption_score = clamp(ai_rev["enterprise_ai_adoption_pct"] * 1.0 + 20, 0, 95)

    # AI startup revenue signal
    total_ai_startup_arr = ai_rev["openai_arr_B"] + ai_rev["anthropic_arr_B"]
    startup_score = clamp(total_ai_startup_arr * 4, 0, 95)

    # Margin concern: penalize if AI margins are significantly lower
    margin_penalty = 0
    if margins["ai_incremental_margin_pct"] < 30:
        margin_penalty = (30 - margins["ai_incremental_margin_pct"]) * 0.5

    score = clamp(
        0.30 * cloud_score
        + 0.25 * ai_contrib_score
        + 0.20 * adoption_score
        + 0.25 * startup_score
        - margin_penalty
    )

    details = {
        "avg_cloud_growth_pct": round(avg_cloud_growth, 1),
        "avg_ai_contribution_ppt": round(avg_ai_contribution, 1),
        "enterprise_adoption_pct": ai_rev["enterprise_ai_adoption_pct"],
        "openai_arr_B": ai_rev["openai_arr_B"],
        "anthropic_arr_B": ai_rev["anthropic_arr_B"],
        "ai_incremental_margin_pct": margins["ai_incremental_margin_pct"],
        "inference_cost_decline_pct": margins["inference_cost_decline_yoy_pct"],
        "cloud_score": round(cloud_score, 1),
        "ai_contrib_score": round(ai_contrib_score, 1),
        "adoption_score": round(adoption_score, 1),
        "margin_penalty": round(margin_penalty, 1),
    }
    return round(score, 1), details


def compute_macro_constraint_score(
    macro_data: dict, yield_data: dict
) -> tuple[float, dict]:
    """
    Macro Constraint Score (0-100):
    Higher = MORE constraint / stress on AI infra cycle.
    - Higher rates = more constraint
    - Wider credit spreads = more constraint
    - Higher inflation = more constraint
    - Power costs = more constraint
    """
    # Rate level component: 10Y yield
    y10 = None
    if "maturities" in yield_data:
        y10 = yield_data["maturities"].get("10y")
    if y10 is None:
        y10 = 4.3  # fallback

    # Rate pressure: 3% = 20, 4% = 40, 5% = 65, 6% = 85
    rate_pressure = clamp((y10 - 2.0) * 20, 0, 95)

    # Yield curve signal: inverted = stress, steep = easing
    spread_10_2 = yield_data.get("spreads", {}).get("10y_2y", 0.5)
    # Inverted (negative) = higher score; positive = lower
    curve_stress = clamp(50 - spread_10_2 * 30, 0, 95)

    # Credit spread pressure
    hy_spread = macro_data["hy_credit_spread_bps"]
    # 300bps = 30, 400bps = 50, 600bps = 80
    credit_pressure = clamp(hy_spread * 0.1 - 5, 0, 95)

    # Inflation component
    cpi = macro_data["us_cpi_yoy_pct"]
    inflation_pressure = clamp((cpi - 2.0) * 25, 0, 95)

    # Power cost pressure
    power_idx = macro_data["data_center_power_cost_index"]
    power_pressure = clamp((power_idx - 100) * 2, 0, 95)

    score = clamp(
        0.30 * rate_pressure
        + 0.15 * curve_stress
        + 0.20 * credit_pressure
        + 0.20 * inflation_pressure
        + 0.15 * power_pressure
    )

    details = {
        "10y_yield": y10,
        "10y_2y_spread": spread_10_2,
        "fed_funds_rate_pct": macro_data["fed_funds_rate_pct"],
        "hy_credit_spread_bps": hy_spread,
        "ig_credit_spread_bps": macro_data["ig_credit_spread_bps"],
        "cpi_yoy_pct": macro_data["us_cpi_yoy_pct"],
        "unemployment_pct": macro_data["us_unemployment_pct"],
        "vix": macro_data["vix"],
        "power_cost_index": power_idx,
        "rate_pressure": round(rate_pressure, 1),
        "curve_stress": round(curve_stress, 1),
        "credit_pressure": round(credit_pressure, 1),
        "inflation_pressure": round(inflation_pressure, 1),
        "power_pressure": round(power_pressure, 1),
    }
    return round(score, 1), details


def compute_heat_index(
    capex_score: float,
    bottleneck_score: float,
    monetization_score: float,
    macro_score: float,
) -> float:
    """
    AI Infra Heat Index = 0.30*Capex + 0.25*Bottleneck + 0.25*Monetization - 0.20*Macro
    Normalized to 0-100.

    Higher = hotter cycle (more investment, more demand, less constraint).
    """
    raw = (
        0.30 * capex_score
        + 0.25 * bottleneck_score
        + 0.25 * monetization_score
        - 0.20 * macro_score
    )
    # Raw range: theoretical min = -20 (if macro=100, rest=0), max = 80 (if all positive=100, macro=0)
    # Normalize to 0-100
    normalized = clamp((raw + 20) * (100 / 100), 0, 100)
    return round(normalized, 1)


def compute_phase_probabilities(
    heat_index: float,
    capex_score: float,
    bottleneck_score: float,
    monetization_score: float,
    macro_score: float,
) -> dict:
    """
    Estimate probability distribution across cycle phases:
    - Acceleration: heat rising, capex ramping, bottleneck emerging
    - Peak Intensity: heat high, all signals hot, constraint building
    - Digestion: heat cooling, capex slowing, monetization gap visible
    - Contraction: heat low, capex cuts, macro stress dominant
    """
    # Start with heat-index-based priors
    if heat_index >= 75:
        probs = {
            "acceleration": 0.20,
            "peak_intensity": 0.55,
            "digestion": 0.20,
            "contraction": 0.05,
        }
    elif heat_index >= 60:
        probs = {
            "acceleration": 0.35,
            "peak_intensity": 0.35,
            "digestion": 0.25,
            "contraction": 0.05,
        }
    elif heat_index >= 45:
        probs = {
            "acceleration": 0.25,
            "peak_intensity": 0.20,
            "digestion": 0.40,
            "contraction": 0.15,
        }
    elif heat_index >= 30:
        probs = {
            "acceleration": 0.15,
            "peak_intensity": 0.10,
            "digestion": 0.35,
            "contraction": 0.40,
        }
    else:
        probs = {
            "acceleration": 0.10,
            "peak_intensity": 0.05,
            "digestion": 0.25,
            "contraction": 0.60,
        }

    # Adjust based on signal patterns

    # If capex is very high but monetization lagging: more peak/digestion risk
    capex_monet_gap = capex_score - monetization_score
    if capex_monet_gap > 15:
        probs["peak_intensity"] += 0.05
        probs["digestion"] += 0.05
        probs["acceleration"] -= 0.05
        probs["contraction"] -= 0.02
        probs["peak_intensity"] = max(0, probs["peak_intensity"])

    # If macro constraint is high: shift toward digestion/contraction
    if macro_score > 50:
        shift = (macro_score - 50) * 0.003
        probs["contraction"] += shift
        probs["digestion"] += shift * 0.5
        probs["acceleration"] -= shift
        probs["peak_intensity"] -= shift * 0.5

    # If bottleneck very high + capex high: peak intensity signal
    if bottleneck_score > 70 and capex_score > 70:
        probs["peak_intensity"] += 0.05
        probs["acceleration"] -= 0.03
        probs["digestion"] -= 0.02

    # Normalize to sum to 1.0
    total = sum(probs.values())
    probs = {k: round(max(0, v / total), 3) for k, v in probs.items()}

    # Fix rounding to exactly 1.0
    diff = 1.0 - sum(probs.values())
    max_key = max(probs, key=probs.get)
    probs[max_key] = round(probs[max_key] + diff, 3)

    return probs


# ---------------------------------------------------------------------------
# Signal Analysis
# ---------------------------------------------------------------------------


def identify_key_drivers(
    capex_details: dict,
    bottleneck_details: dict,
    monet_details: dict,
    macro_details: dict,
) -> list[str]:
    """Identify the top drivers of the current cycle state."""
    drivers = []

    # Capex drivers
    total_capex = capex_details["total_capex_ttm_B"]
    avg_growth = capex_details["avg_yoy_growth_pct"]
    drivers.append(
        f"Hyperscaler capex TTM ${total_capex:.0f}B (+{avg_growth:.0f}% YoY) - "
        f"unprecedented investment pace"
    )

    # NVDA driver
    nvda_rev = bottleneck_details["nvda_dc_revenue_ttm_B"]
    nvda_growth = bottleneck_details["nvda_dc_yoy_growth_pct"]
    drivers.append(
        f"NVDA data center revenue ${nvda_rev:.0f}B TTM (+{nvda_growth:.0f}% YoY) - "
        f"book-to-bill {bottleneck_details['nvda_book_to_bill']:.1f}x"
    )

    # Cloud/AI monetization
    cloud_growth = monet_details["avg_cloud_growth_pct"]
    ai_ppt = monet_details["avg_ai_contribution_ppt"]
    drivers.append(
        f"Cloud growth {cloud_growth:.0f}% with {ai_ppt:.0f}ppt AI contribution - "
        f"monetization pathway emerging"
    )

    # Macro context
    y10 = macro_details["10y_yield"]
    drivers.append(
        f"10Y yield at {y10:.2f}% - rates elevated but stable, "
        f"not yet constraining tech investment"
    )

    return drivers


def identify_stress_signals(
    capex_details: dict,
    bottleneck_details: dict,
    monet_details: dict,
    macro_details: dict,
) -> list[str]:
    """Identify current stress signals in the cycle."""
    signals = []

    # Capex sustainability
    avg_ocf = capex_details["avg_capex_pct_opcf"]
    if avg_ocf > 55:
        signals.append(
            f"Capex consuming {avg_ocf:.0f}% of operating cash flow on average - "
            f"approaching sustainability limits"
        )

    # NVDA margin compression
    gm = bottleneck_details["nvda_gross_margin_pct"]
    if gm < 74:
        signals.append(
            f"NVDA gross margins compressing to {gm:.0f}% - "
            f"Blackwell ramp and competition pressuring pricing"
        )

    # Power constraints
    if bottleneck_details["power_constraint"] == "high":
        signals.append(
            "Data center power availability rated HIGH constraint - "
            "becoming binding on build-out pace"
        )

    # Monetization gap
    capex_monet = capex_details.get("avg_yoy_growth_pct", 0)
    cloud_growth = monet_details.get("avg_cloud_growth_pct", 0)
    if capex_monet > cloud_growth * 2:
        signals.append(
            f"Capex growing {capex_monet:.0f}% vs cloud revenue {cloud_growth:.0f}% - "
            f"investment outpacing revenue growth ~{capex_monet/cloud_growth:.1f}x"
        )

    # Credit/macro
    if macro_details["hy_credit_spread_bps"] > 400:
        signals.append(
            f"HY credit spreads at {macro_details['hy_credit_spread_bps']}bps - "
            f"financial conditions tightening"
        )

    return signals


def identify_early_warnings(
    capex_details: dict,
    bottleneck_details: dict,
    monet_details: dict,
    macro_details: dict,
) -> list[str]:
    """Identify early warning indicators to watch."""
    warnings = []

    warnings.append(
        "WATCH: Capex/revenue ratio at cycle highs - historically, "
        "sustained >25% leads to write-downs within 18-24 months"
    )

    if monet_details["ai_incremental_margin_pct"] < 30:
        warnings.append(
            f"WATCH: AI workload incremental margins at {monet_details['ai_incremental_margin_pct']}% "
            f"vs traditional cloud ~40%+ - margin dilution risk if mix shifts"
        )

    warnings.append(
        "WATCH: Inference cost declining ~55% YoY - positive for adoption "
        "but compresses per-unit economics, requires volume offset"
    )

    if macro_details["power_cost_index"] > 110:
        warnings.append(
            f"WATCH: Data center power cost index at {macro_details['power_cost_index']} "
            f"(vs 100 baseline) - energy costs becoming material to TCO"
        )

    warnings.append(
        "WATCH: Sovereign AI demand adding new demand vector but also "
        "geopolitical risk if export controls tighten further"
    )

    return warnings


def generate_outlook(
    heat_index: float,
    phase_probs: dict,
    capex_score: float,
    macro_score: float,
) -> dict:
    """Generate 6-12 month outlook summary."""
    dominant_phase = max(phase_probs, key=phase_probs.get)
    dominant_prob = phase_probs[dominant_phase]

    if dominant_phase == "acceleration":
        base_case = (
            "Capex cycle continues accelerating through H2 2026. "
            "Hyperscalers have guided massive budgets and demand visibility remains strong. "
            "Risk is in execution (power, supply chain) not demand."
        )
        risk_scenario = (
            "Macro shock (rate spike, recession) forces CFOs to trim capex guidance. "
            "Or AI monetization disappoints, leading to investor pushback on spend."
        )
    elif dominant_phase == "peak_intensity":
        base_case = (
            "Cycle approaching peak intensity. Investment continues but growth rates "
            "likely plateau by late 2026. The gap between capex and monetization is the "
            "key metric to watch for timing the eventual cool-down."
        )
        risk_scenario = (
            "Capex-to-revenue gap widens further, triggering analyst downgrades. "
            "Power/supply constraints force involuntary slowdown before demand peaks."
        )
    elif dominant_phase == "digestion":
        base_case = (
            "Entering digestion phase where prior investments need to prove ROI. "
            "Capex growth slows but absolute levels remain high. Focus shifts to "
            "utilization rates and margin impact."
        )
        risk_scenario = (
            "Write-down cycle begins as capacity exceeds near-term demand. "
            "Cloud pricing pressure from overcapacity."
        )
    else:
        base_case = (
            "Contraction phase with capex cuts and capacity rationalization. "
            "Survival of the fittest among AI infrastructure plays."
        )
        risk_scenario = (
            "Extended downturn as macro and demand weakness compound. "
            "Structural overcapacity takes multiple quarters to clear."
        )

    return {
        "dominant_phase": dominant_phase,
        "dominant_probability": dominant_prob,
        "heat_index_level": (
            "very_hot"
            if heat_index >= 75
            else "hot"
            if heat_index >= 60
            else "warm"
            if heat_index >= 45
            else "cool"
            if heat_index >= 30
            else "cold"
        ),
        "base_case_6_12m": base_case,
        "risk_scenario": risk_scenario,
        "key_upcoming_catalysts": [
            "NVDA GTC 2026 (March) - next-gen architecture announcements",
            "Hyperscaler Q1 2026 earnings (April-May) - capex guidance updates",
            "Fed rate decisions - March, May FOMC meetings",
            "Major enterprise AI deployments at scale (monitoring for adoption inflection)",
            "Power/energy infrastructure policy developments",
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"AI Infrastructure Capex Cycle Monitor - Data Collection")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*60}")

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Fetch live Treasury data ---
    print("\n[Layer 4] Fetching Treasury yield curve data...")
    print("  Fetching 2026 data...")
    rows_2026 = fetch_treasury_csv(TREASURY_URL_2026)
    print(f"  Got {len(rows_2026)} trading days for 2026")

    print("  Fetching 2025 data for YoY comparison...")
    rows_2025 = fetch_treasury_csv(TREASURY_URL_2025)
    print(f"  Got {len(rows_2025)} trading days for 2025")

    yield_data = get_latest_yields(rows_2026, rows_2025)
    if "error" not in yield_data:
        y10 = yield_data["maturities"].get("10y", "N/A")
        y2 = yield_data["maturities"].get("2y", "N/A")
        spread = yield_data["spreads"].get("10y_2y", "N/A")
        print(f"  Latest date: {yield_data['date']}")
        print(f"  10Y: {y10}%  |  2Y: {y2}%  |  10Y-2Y spread: {spread}%")
    else:
        print(f"  WARNING: {yield_data['error']}")

    # --- Compute Layer Scores ---
    print("\n[Layer 1] Computing Capex Acceleration Score...")
    capex_score, capex_details = compute_capex_acceleration_score(HYPERSCALER_CAPEX)
    print(f"  Score: {capex_score}/100")

    print("\n[Layer 2] Computing Bottleneck Power Score...")
    bottleneck_score, bottleneck_details = compute_bottleneck_power_score(
        INFRA_SUPPLIERS
    )
    print(f"  Score: {bottleneck_score}/100")

    print("\n[Layer 3] Computing Monetization Realization Score...")
    monet_score, monet_details = compute_monetization_score(MONETIZATION)
    print(f"  Score: {monet_score}/100")

    print("\n[Layer 4] Computing Macro Constraint Score...")
    macro_score, macro_details = compute_macro_constraint_score(
        MACRO_HARDCODED, yield_data
    )
    print(f"  Score: {macro_score}/100 (higher = more constraint)")

    # --- Compute Heat Index ---
    heat_index = compute_heat_index(
        capex_score, bottleneck_score, monet_score, macro_score
    )
    print(f"\n{'='*60}")
    print(f"AI INFRA HEAT INDEX: {heat_index}/100")
    print(f"{'='*60}")

    # --- Phase Probabilities ---
    phase_probs = compute_phase_probabilities(
        heat_index, capex_score, bottleneck_score, monet_score, macro_score
    )
    print(f"\nPhase Probabilities:")
    for phase, prob in phase_probs.items():
        bar = "#" * int(prob * 50)
        print(f"  {phase:20s}: {prob:.1%} {bar}")

    # --- Signal Analysis ---
    key_drivers = identify_key_drivers(
        capex_details, bottleneck_details, monet_details, macro_details
    )
    stress_signals = identify_stress_signals(
        capex_details, bottleneck_details, monet_details, macro_details
    )
    early_warnings = identify_early_warnings(
        capex_details, bottleneck_details, monet_details, macro_details
    )
    outlook = generate_outlook(heat_index, phase_probs, capex_score, macro_score)

    # --- Assemble Output ---
    output = {
        "timestamp": timestamp,
        "heat_index": heat_index,
        "phase_probabilities": phase_probs,
        "layers": {
            "capex_acceleration": {
                "score": capex_score,
                "weight": 0.30,
                "details": capex_details,
            },
            "bottleneck_power": {
                "score": bottleneck_score,
                "weight": 0.25,
                "details": bottleneck_details,
            },
            "monetization": {
                "score": monet_score,
                "weight": 0.25,
                "details": monet_details,
            },
            "macro_constraint": {
                "score": macro_score,
                "weight": -0.20,
                "details": macro_details,
            },
        },
        "yield_curve": yield_data,
        "raw_data": {
            "hyperscaler_capex": HYPERSCALER_CAPEX,
            "infra_suppliers": INFRA_SUPPLIERS,
            "monetization": MONETIZATION,
            "macro": MACRO_HARDCODED,
        },
        "key_drivers": key_drivers,
        "stress_signals": stress_signals,
        "early_warnings": early_warnings,
        "outlook_6_12m": outlook,
    }

    # --- Write Output ---
    print(f"\nWriting to {LATEST_JSON}...")
    with open(LATEST_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Done ({LATEST_JSON.stat().st_size:,} bytes)")

    print(f"Appending to {HISTORY_JSONL}...")
    with open(HISTORY_JSONL, "a") as f:
        f.write(json.dumps(output) + "\n")
    print(f"  Done")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Heat Index:    {heat_index}/100 ({outlook['heat_index_level']})")
    print(
        f"Dominant Phase: {outlook['dominant_phase']} ({outlook['dominant_probability']:.0%})"
    )
    print(f"\nKey Drivers:")
    for d in key_drivers:
        print(f"  - {d}")
    print(f"\nStress Signals:")
    for s in stress_signals:
        print(f"  ! {s}")
    print(f"\nEarly Warnings:")
    for w in early_warnings:
        print(f"  * {w}")
    print(f"\nBase Case (6-12m):")
    print(f"  {outlook['base_case_6_12m']}")
    print(f"\n{'='*60}")
    print(f"Data written to: {LATEST_JSON}")
    print(f"History appended to: {HISTORY_JSONL}")


if __name__ == "__main__":
    main()
