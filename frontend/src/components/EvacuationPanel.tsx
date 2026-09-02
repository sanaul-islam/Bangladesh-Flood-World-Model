import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  MapPinned,
  Navigation,
  Shield,
  TrendingDown,
} from "lucide-react";

import type {
  EvacuationResponse,
  HazardResponse,
} from "../api/types";

interface EvacuationPanelProps {
  hazard: HazardResponse | null;

  evacuation:
    | EvacuationResponse
    | null;

  loading: boolean;

  onFindSafestShelter: () => void;
}

function riskLabel(
  value: number,
): string {
  if (value < 0.25) {
    return "Low";
  }

  if (value < 0.5) {
    return "Moderate";
  }

  if (value < 0.75) {
    return "High";
  }

  return "Severe";
}

function percentage(
  value: number,
): string {
  return `${Math.round(
    value * 100,
  )}%`;
}

export function EvacuationPanel({
  hazard,
  evacuation,
  loading,
  onFindSafestShelter,
}: EvacuationPanelProps) {
  if (loading) {
    return (
      <div className="evacuation-panel">
        <div className="evacuation-empty">
          <Navigation
            size={22}
          />

          <strong>
            Calculating safest evacuation route
          </strong>

          <span>
            Evaluating forecast hazard, road risk,
            bridge exposure, travel time, and
            shelter accessibility.
          </span>
        </div>
      </div>
    );
  }

  const recommendation =
    evacuation?.result
      ?.recommended_shelter;

  const route =
    evacuation?.result?.route;

  const alternatives =
    evacuation?.result
      ?.alternatives ?? [];

  const weights =
    evacuation?.result
      ?.ranking_weights;

  if (
    !evacuation ||
    !recommendation
  ) {
    return (
      <div className="evacuation-panel">
        <div className="evacuation-intro">
          <div className="panel-title">
            Risk-aware planning
          </div>

          <p>
            Calculate an evacuation plan using
            the selected forecast day and the
            risk-aware road network.
          </p>
        </div>

        {hazard && (
          <div className="evacuation-card">
            <div className="panel-title">
              Forecast condition
            </div>

            <div className="route-stat-grid">
              <div className="route-stat">
                <div className="route-stat-label">
                  Flood risk
                </div>

                <div className="route-stat-value">
                  {hazard.hazard_score.toFixed(
                    3,
                  )}
                </div>

                <small>
                  {riskLabel(
                    hazard.hazard_score,
                  )}
                </small>
              </div>

              <div className="route-stat">
                <div className="route-stat-label">
                  Forecast
                </div>

                <div className="route-stat-value">
                  Day{" "}
                  {hazard.forecast_day}
                </div>

                <small>
                  Planning condition
                </small>
              </div>
            </div>
          </div>
        )}

        <button
          type="button"
          className="toolbar-button primary-button evacuation-action"
          onClick={
            onFindSafestShelter
          }
          disabled={!hazard}
        >
          <Navigation
            size={15}
          />

          Find safest shelter
        </button>
      </div>
    );
  }

  return (
    <div className="evacuation-panel">
      <div className="recommendation-card">
        <div className="recommendation-label">
          <CheckCircle2 size={15} />

          RECOMMENDED SHELTER
        </div>

        <div className="recommendation-name">
          Shelter #
          {
            recommendation.shelter_id
          }
        </div>

        <div className="recommendation-location">
          <MapPinned size={13} />

          {
            recommendation.latitude.toFixed(
              5,
            )
          }
          ,{" "}
          {
            recommendation.longitude.toFixed(
              5,
            )
          }
        </div>

        <div className="route-stat-grid">
          <div className="route-stat">
            <div className="route-stat-label">
              Road distance
            </div>

            <div className="route-stat-value">
              {
                recommendation.road_distance_km.toFixed(
                  2,
                )
              }{" "}
              km
            </div>
          </div>

          <div className="route-stat">
            <div className="route-stat-label">
              Travel time
            </div>

            <div className="route-stat-value">
              {
                recommendation.travel_time_min.toFixed(
                  1,
                )
              }{" "}
              min
            </div>
          </div>

          <div className="route-stat">
            <div className="route-stat-label">
              Destination hazard
            </div>

            <div className="route-stat-value">
              {
                recommendation.destination_hazard.toFixed(
                  3,
                )
              }
            </div>
          </div>

          <div className="route-stat">
            <div className="route-stat-label">
              Combined score
            </div>

            <div className="route-stat-value">
              {
                recommendation.combined_score.toFixed(
                  3,
                )
              }
            </div>
          </div>
        </div>
      </div>

      <div className="evacuation-card">
        <div className="panel-title">
          Why this shelter?
        </div>

        <div className="explanation-list">
          <div>
            <TrendingDown size={13} />
            Highest-ranked reachable candidate
          </div>

          <div>
            <Shield size={13} />
            Destination hazard is considered
          </div>

          <div>
            <Clock3 size={13} />
            Travel time influences the score
          </div>

          <div>
            <MapPinned size={13} />
            Bridge exposure is considered
          </div>

          <div>
            <Navigation size={13} />
            Route flood risk is considered
          </div>
        </div>
      </div>

      {weights && (
        <div className="evacuation-card">
          <div className="panel-title">
            Ranking weights
          </div>

          <div className="weight-list">
            <div>
              <span>
                Route risk
              </span>

              <strong>
                {percentage(
                  weights.route_risk,
                )}
              </strong>
            </div>

            <div>
              <span>
                Destination hazard
              </span>

              <strong>
                {percentage(
                  weights.destination_hazard,
                )}
              </strong>
            </div>

            <div>
              <span>
                Population exposure
              </span>

              <strong>
                {percentage(
                  weights.destination_population_exposure,
                )}
              </strong>
            </div>

            <div>
              <span>
                Travel time
              </span>

              <strong>
                {percentage(
                  weights.travel_time,
                )}
              </strong>
            </div>

            <div>
              <span>
                Bridge exposure
              </span>

              <strong>
                {percentage(
                  weights.bridge_exposure,
                )}
              </strong>
            </div>
          </div>
        </div>
      )}

      {route && (
        <div className="evacuation-card">
          <div className="panel-title">
            Route intelligence
          </div>

          <div className="route-stat-grid">
            <div className="route-stat">
              <div className="route-stat-label">
                Road distance
              </div>

              <div className="route-stat-value">
                {
                  route.statistics.road_distance_km.toFixed(
                    2,
                  )
                }{" "}
                km
              </div>
            </div>

            <div className="route-stat">
              <div className="route-stat-label">
                Travel time
              </div>

              <div className="route-stat-value">
                {
                  route.statistics.estimated_travel_time_min.toFixed(
                    1,
                  )
                }{" "}
                min
              </div>
            </div>

            <div className="route-stat">
              <div className="route-stat-label">
                Mean flood risk
              </div>

              <div className="route-stat-value">
                {
                  route.statistics.mean_flood_risk.toFixed(
                    3,
                  )
                }
              </div>
            </div>

            <div className="route-stat">
              <div className="route-stat-label">
                Maximum risk
              </div>

              <div className="route-stat-value">
                {
                  route.statistics.maximum_flood_risk.toFixed(
                    3,
                  )
                }
              </div>
            </div>

            <div className="route-stat">
              <div className="route-stat-label">
                Uncertainty
              </div>

              <div className="route-stat-value">
                {
                  route.statistics.mean_uncertainty_risk.toFixed(
                    3,
                  )
                }
              </div>
            </div>

            <div className="route-stat">
              <div className="route-stat-label">
                Bridge edges
              </div>

              <div className="route-stat-value">
                {
                  route.statistics.bridge_edges
                }
              </div>
            </div>
          </div>
        </div>
      )}

      {alternatives.length > 0 && (
        <div className="evacuation-card">
          <div className="panel-title">
            Alternative shelters
          </div>

          <div className="alternative-list">
            {alternatives
              .slice(0, 5)
              .map(
                (
                  alternative,
                ) => (
                  <div
                    key={
                      alternative.shelter_id
                    }
                    className="alternative-row"
                  >
                    <div>
                      <strong>
                        #
                        {
                          alternative.shelter_id
                        }
                      </strong>

                      <small>
                        {
                          alternative.road_distance_km.toFixed(
                            2,
                          )
                        }{" "}
                        km ·{" "}
                        {
                          alternative.travel_time_min.toFixed(
                            1,
                          )
                        }{" "}
                        min
                      </small>
                    </div>

                    <span>
                      Risk{" "}
                      {
                        alternative.mean_flood_risk.toFixed(
                          2,
                        )
                      }
                    </span>

                    <strong>
                      {
                        alternative.combined_score.toFixed(
                          3,
                        )
                      }
                    </strong>
                  </div>
                ),
              )}
          </div>
        </div>
      )}

      {!evacuation.result
        .system
        .live_shelter_availability_available && (
        <div className="panel-note">
          <AlertTriangle size={13} />

          Shelter capacity and live availability
          are not available from the current
          static inventory.
        </div>
      )}

      <button
        type="button"
        className="toolbar-button secondary-button evacuation-action"
        onClick={
          onFindSafestShelter
        }
      >
        <Navigation
          size={15}
        />

        Recalculate
      </button>
    </div>
  );
}

export default EvacuationPanel;
