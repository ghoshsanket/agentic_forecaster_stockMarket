"""Risk Agent — ATR-based stop-loss / take-profit (PAPER-DEFINED).

Directional confidence:
    if p_up >= .5: direction = UP,   confidence = p_up
    else:          direction = DOWN, confidence = 1 - p_up

Confidence bands:
    confidence > .80: HIGH   lambda_SL=.8  lambda_TP=1.5
    confidence > .60: MEDIUM lambda_SL=1.0 lambda_TP=1.0
    otherwise:        LOW    lambda_SL=1.2 lambda_TP=.8

UP:
    SL = close - lambda_SL * ATR
    TP = close + lambda_TP * ATR

DOWN:
    SL = close + lambda_SL * ATR
    TP = close - lambda_TP * ATR

RRR       = abs(TP - close) / abs(close - SL)
RiskScore = confidence / max(ATR, 1e-8)

No account position sizing (reconstruction does not invent sizing).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskDecision:
    ticker: str
    date: str
    direction: str
    confidence: float
    confidence_level: str
    close: float
    atr: float
    lambda_sl: float
    lambda_tp: float
    stop_loss: float
    take_profit: float
    rrr: float
    risk_score: float
    reason_codes: list[str] = None

    def __post_init__(self):
        if self.reason_codes is None:
            self.reason_codes = []


class RiskAgent:
    def __init__(self):
        pass

    def decide(
        self,
        ticker: str,
        date: str,
        p_up: float,
        close: float,
        atr: float,
    ) -> RiskDecision:
        if p_up >= 0.5:
            direction = "UP"
            confidence = p_up
        else:
            direction = "DOWN"
            confidence = 1.0 - p_up

        if confidence > 0.80:
            level = "HIGH"
            lambda_sl, lambda_tp = 0.8, 1.5
        elif confidence > 0.60:
            level = "MEDIUM"
            lambda_sl, lambda_tp = 1.0, 1.0
        else:
            level = "LOW"
            lambda_sl, lambda_tp = 1.2, 0.8

        atr = max(atr, 1e-8)

        if direction == "UP":
            sl = close - lambda_sl * atr
            tp = close + lambda_tp * atr
        else:
            sl = close + lambda_sl * atr
            tp = close - lambda_tp * atr

        rrr = abs(tp - close) / max(abs(close - sl), 1e-8)
        risk_score = confidence / atr

        reason_codes = [
            f"direction={direction}",
            f"confidence_level={level}",
            f"lambda_SL={lambda_sl}",
            f"lambda_tp={lambda_tp}",
        ]

        return RiskDecision(
            ticker=ticker,
            date=str(date),
            direction=direction,
            confidence=float(confidence),
            confidence_level=level,
            close=float(close),
            atr=float(atr),
            lambda_sl=lambda_sl,
            lambda_tp=lambda_tp,
            stop_loss=float(sl),
            take_profit=float(tp),
            rrr=float(rrr),
            risk_score=float(risk_score),
            reason_codes=reason_codes,
        )
