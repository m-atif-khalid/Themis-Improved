"""Crime likelihood + time-to-event prediction.

Inputs:
- World state (Chaotic/Peaceful) from UI
- Crime severity derived from CASE_CRIME.Crime_Name
- Mental state derived from MEDICAL_HISTORY.Psychological_Profile

Models:
- XGBoost for probability of reoffending
- lifelines (CoxPH) for time-to-event
- SHAP for explanation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import mysql.connector
from lifelines import CoxPHFitter
from xgboost import XGBClassifier

END_STATUSES = {"Closed", "Convicted", "Resolved"}


@dataclass
class FeatureRow:
    world_state: int
    severity_score: float
    mental_state_score: float
    time_days: float
    event_occurred: int


def _mental_state_score(profile: Optional[str]) -> float:
    if not profile:
        return 0.3
    profile_l = profile.lower()
    high_risk = ["psychotic", "aggressive", "violent", "unstable", "bipolar", "paranoid"]
    medium_risk = ["depressed", "anxious", "ptsd", "trauma"]

    if any(k in profile_l for k in high_risk):
        return 0.9
    if any(k in profile_l for k in medium_risk):
        return 0.6
    return 0.4


def _crime_severity_score(crime_name: Optional[str]) -> float:
    if not crime_name:
        return 0.3
    name = crime_name.lower()
    high = ["murder", "homicide", "terror", "rape", "kidnap"]
    medium = ["assault", "robbery", "armed", "fraud", "arson"]
    if any(k in name for k in high):
        return 0.9
    if any(k in name for k in medium):
        return 0.6
    return 0.4


def _world_state_score(world_state: str) -> int:
    return 1 if world_state.lower() == "chaotic" else 0


def _fetch_case_history(db_config: Dict[str, Any]) -> pd.DataFrame:
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT
                f.FIR_ID,
                f.Incident_Date,
                cc.Crime_Name,
                mh.Psychological_Profile,
                csh.Status,
                csh.Status_Date
            FROM FIR f
            LEFT JOIN CASE_CRIME cc ON cc.FIR_ID = f.FIR_ID AND (cc.Is_Primary = TRUE OR cc.Is_Primary IS NULL)
            LEFT JOIN CASE_PARTICIPATION cp ON cp.FIR_ID = f.FIR_ID AND (cp.Is_Primary = TRUE OR cp.Role IN ('Accused', 'Suspect'))
            LEFT JOIN MEDICAL_HISTORY mh ON mh.CNIC = cp.CNIC
            LEFT JOIN CASE_STATUS_HISTORY csh ON csh.FIR_ID = f.FIR_ID
            """
        )
        rows = cursor.fetchall()
        return pd.DataFrame(rows)
    finally:
        cursor.close()
        connection.close()


def _build_feature_rows(
    rows: pd.DataFrame,
    world_state: str,
) -> List[FeatureRow]:
    features: List[FeatureRow] = []
    for _, row in rows.iterrows():
        incident_date = row.get("Incident_Date")
        status_date = row.get("Status_Date")
        if not isinstance(incident_date, (date, pd.Timestamp)):
            continue

        event_occurred = 1 if (row.get("Status") in END_STATUSES) else 0
        if isinstance(status_date, (date, pd.Timestamp)):
            time_days = max((pd.to_datetime(status_date) - pd.to_datetime(incident_date)).days, 0)
        else:
            time_days = 0

        features.append(
            FeatureRow(
                world_state=_world_state_score(world_state),
                severity_score=_crime_severity_score(row.get("Crime_Name")),
                mental_state_score=_mental_state_score(row.get("Psychological_Profile")),
                time_days=float(time_days),
                event_occurred=event_occurred,
            )
        )
    return features


def _to_dataframe(features: Iterable[FeatureRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "world_state": f.world_state,
                "severity_score": f.severity_score,
                "mental_state_score": f.mental_state_score,
                "time_days": f.time_days,
                "event_occurred": f.event_occurred,
            }
            for f in features
        ]
    )


def _danger_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    if prob < 0.7:
        return "Medium"
    return "High"


def _train_models(df: pd.DataFrame) -> Tuple[XGBClassifier, CoxPHFitter]:
    x = df[["world_state", "severity_score", "mental_state_score"]]
    y = df["event_occurred"]

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
    )
    clf.fit(x, y)

    cph = CoxPHFitter()
    cph_df = df[["time_days", "event_occurred", "world_state", "severity_score", "mental_state_score"]]
    if (cph_df["time_days"] > 0).any():
        cph.fit(cph_df, duration_col="time_days", event_col="event_occurred")

    return clf, cph


def predict_likelihood(
    world_state: str,
    crime_name: Optional[str],
    psychological_profile: Optional[str],
    clf: XGBClassifier,
    cph: CoxPHFitter,
) -> Dict[str, Any]:
    features = pd.DataFrame(
        [
            {
                "world_state": _world_state_score(world_state),
                "severity_score": _crime_severity_score(crime_name),
                "mental_state_score": _mental_state_score(psychological_profile),
            }
        ]
    )

    probability = float(clf.predict_proba(features)[0][1])

    predicted_time_days = None
    if hasattr(cph, "params_"):
        surv = cph.predict_survival_function(features.assign(time_days=1, event_occurred=1))
        if not surv.empty:
            predicted_time_days = float(surv.index[surv.iloc[:, 0].lt(0.5).idxmax()])

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(features)
    shap_summary = dict(zip(features.columns, map(float, shap_values[0])))

    return {
        "probability": probability,
        "danger_level": _danger_level(probability),
        "predicted_time_days": predicted_time_days,
        "shap_values": shap_summary,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Likelihood + time-to-event prediction")
    parser.add_argument("--world-state", required=True, choices=["Chaotic", "Peaceful"])
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-user", default="root")
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-name", default="themis")
    parser.add_argument("--crime-name", default=None)
    parser.add_argument("--psych-profile", default=None)
    parser.add_argument("--csv", default="backend/csv_data/likelihood_features.csv")
    parser.add_argument("--use-csv", action="store_true")
    return parser


def _load_csv(csv_path: str, world_state: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["world_state"] = df["world_state"].fillna(_world_state_score(world_state))
    return df


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.use_csv:
        df = _load_csv(args.csv, args.world_state)
    else:
        db_config = {
            "host": args.db_host,
            "user": args.db_user,
            "password": args.db_password,
            "database": args.db_name,
        }
        raw = _fetch_case_history(db_config)
        features = _build_feature_rows(raw, args.world_state)
        df = _to_dataframe(features)

    if df.empty:
        print("No training data available.")
        return

    clf, cph = _train_models(df)

    result = predict_likelihood(
        world_state=args.world_state,
        crime_name=args.crime_name,
        psychological_profile=args.psych_profile,
        clf=clf,
        cph=cph,
    )

    print("Likelihood result:")
    print(result)


if __name__ == "__main__":
    main()
