interface RiskSummaryProps {
  hazard: number | null;
  uncertainty: number | null;
  routeRisk: number | null;
  forecastDay: number;
}

function formatValue(
  value: number | null,
): string {
  if (value === null) {
    return "—";
  }

  return value.toFixed(3);
}

export function RiskSummary({
  hazard,
  uncertainty,
  routeRisk,
  forecastDay,
}: RiskSummaryProps) {
  return (
    <div className="risk-summary">
      <div className="risk-card">
        <div className="risk-card-label">
          FLOOD RISK
        </div>

        <div className="risk-card-value">
          {formatValue(hazard)}
        </div>

        <small>
          {hazard === null
            ? "Select a location"
            : "Local hazard"}
        </small>
      </div>

      <div className="risk-card">
        <div className="risk-card-label">
          UNCERTAINTY
        </div>

        <div className="risk-card-value">
          {formatValue(uncertainty)}
        </div>

        <small>
          {uncertainty === null
            ? "Calculate evacuation"
            : "Forecast uncertainty"}
        </small>
      </div>

      <div className="risk-card">
        <div className="risk-card-label">
          ROUTE RISK
        </div>

        <div className="risk-card-value">
          {formatValue(routeRisk)}
        </div>

        <small>
          {routeRisk === null
            ? "Calculate evacuation"
            : "Recommended route"}
        </small>
      </div>

      <div className="risk-card">
        <div className="risk-card-label">
          FORECAST
        </div>

        <div className="risk-card-value">
          DAY {forecastDay}
        </div>

        <small>
          Selected forecast
        </small>
      </div>
    </div>
  );
}

export default RiskSummary;
