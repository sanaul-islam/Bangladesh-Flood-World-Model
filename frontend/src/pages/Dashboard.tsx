import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  ChevronDown,
  Crosshair,
  LoaderCircle,
  MapPin,
  Navigation,
  Search,
  ShieldCheck,
} from "lucide-react";

import {
  evacuate,
  getForecast,
  getHazard,
  getHazardGrid,
  getHealth,
  getShelters,
} from "../api/client";

import type {
  EvacuationResponse,
  ForecastResponse,
  HazardGridPoint,
  HazardResponse,
  Shelter,
} from "../api/types";

import {
  searchPlaces,
  type GeocodingResult,
} from "../api/geocoding";

import Header from "../components/Header";
import RiskSummary from "../components/RiskSummary";
import MapView from "../components/MapView";
import EvacuationPanel from "../components/EvacuationPanel";

const DEFAULT_LATITUDE =
  23.8103;

const DEFAULT_LONGITUDE =
  90.4125;

const FORECAST_SAMPLE = 0;

const DEFAULT_FORECAST_DAYS = [
  1,
  2,
  3,
  4,
  5,
  6,
  7,
];

export default function Dashboard() {
  const [latitude, setLatitude] =
    useState(
      DEFAULT_LATITUDE,
    );

  const [longitude, setLongitude] =
    useState(
      DEFAULT_LONGITUDE,
    );

  const [locationName, setLocationName] =
    useState("Dhaka");

  const [latitudeInput, setLatitudeInput] =
    useState(
      DEFAULT_LATITUDE.toFixed(
        5,
      ),
    );

  const [longitudeInput, setLongitudeInput] =
    useState(
      DEFAULT_LONGITUDE.toFixed(
        5,
      ),
    );

  const [forecastDay, setForecastDay] =
    useState(1);

  const [forecast, setForecast] =
    useState<ForecastResponse | null>(
      null,
    );

  const [hazard, setHazard] =
    useState<HazardResponse | null>(
      null,
    );

  const [riskGrid, setRiskGrid] =
    useState<HazardGridPoint[]>(
      [],
    );

  const [shelters, setShelters] =
    useState<Shelter[]>([]);

  const [evacuation, setEvacuation] =
    useState<EvacuationResponse | null>(
      null,
    );

  const [backendOnline, setBackendOnline] =
    useState(false);

  const [hazardLoading, setHazardLoading] =
    useState(false);

  const [
    evacuationLoading,
    setEvacuationLoading,
  ] = useState(false);

  const [gridLoading, setGridLoading] =
    useState(false);

  const [pageError, setPageError] =
    useState<string | null>(
      null,
    );

  const [searchText, setSearchText] =
    useState("");

  const [
    searchResults,
    setSearchResults,
  ] = useState<GeocodingResult[]>(
    [],
  );

  const [searchLoading, setSearchLoading] =
    useState(false);

  const [searchError, setSearchError] =
    useState<string | null>(
      null,
    );

  const [
    showSuggestions,
    setShowSuggestions,
  ] = useState(false);

  const searchRequestId =
    useRef(0);

  /*
   * Initial API data.
   */
  useEffect(() => {
    let active = true;

    async function loadInitialData() {
      try {
        const [
          healthResponse,
          forecastResponse,
          sheltersResponse,
        ] = await Promise.all([
          getHealth(),
          getForecast(),
          getShelters(),
        ]);

        if (!active) {
          return;
        }

        setBackendOnline(
          healthResponse.status ===
            "ok",
        );

        setForecast(
          forecastResponse,
        );

        setShelters(
          sheltersResponse.shelters,
        );
      } catch (error) {
        if (!active) {
          return;
        }

        setBackendOnline(false);

        setPageError(
          error instanceof Error
            ? error.message
            : "Unable to connect to the flood response API.",
        );
      }
    }

    void loadInitialData();

    return () => {
      active = false;
    };
  }, []);

  const forecastDays =
    useMemo(() => {
      if (
        forecast?.forecast_days &&
        forecast.forecast_days
          .length > 0
      ) {
        return forecast.forecast_days;
      }

      return DEFAULT_FORECAST_DAYS;
    }, [forecast]);

  /*
   * Autocomplete.
   */
  useEffect(() => {
    const query =
      searchText.trim();

    if (query.length < 2) {
      return;
    }

    const requestId =
      ++searchRequestId.current;

    const timer =
      window.setTimeout(
        async () => {
          setSearchLoading(true);

          try {
            const results =
              await searchPlaces(
                query,
                latitude,
                longitude,
              );

            if (
              requestId !==
              searchRequestId.current
            ) {
              return;
            }

            setSearchResults(
              results,
            );

            setSearchError(null);
          } catch (error) {
            if (
              requestId !==
              searchRequestId.current
            ) {
              return;
            }

            setSearchResults([]);

            setSearchError(
              error instanceof Error
                ? error.message
                : "Search failed.",
            );
          } finally {
            if (
              requestId ===
              searchRequestId.current
            ) {
              setSearchLoading(
                false,
              );
            }
          }
        },
        300,
      );

    return () => {
      window.clearTimeout(
        timer,
      );
    };
  }, [
    searchText,
    latitude,
    longitude,
  ]);

  /*
   * Forecast hazard + risk surface.
   *
   * Day 5 means all of these requests
   * use Day 5.
   */
  useEffect(() => {
    let active = true;

    async function loadForecastView() {
      setHazardLoading(true);
      setGridLoading(true);

      setPageError(null);
      setEvacuation(null);

      try {
        const [
          hazardResponse,
          gridResponse,
        ] =
          await Promise.all([
            getHazard(
              latitude,
              longitude,
              FORECAST_SAMPLE,
              forecastDay,
            ),

            getHazardGrid(
              latitude,
              longitude,
              FORECAST_SAMPLE,
              forecastDay,
            ),
          ]);

        if (!active) {
          return;
        }

        setHazard(
          hazardResponse,
        );

        setRiskGrid(
          gridResponse.points,
        );
      } catch (error) {
        if (!active) {
          return;
        }

        setHazard(null);
        setRiskGrid([]);

        setPageError(
          error instanceof Error
            ? error.message
            : "Unable to load forecast risk.",
        );
      } finally {
        if (active) {
          setHazardLoading(
            false,
          );

          setGridLoading(
            false,
          );
        }
      }
    }

    void loadForecastView();

    return () => {
      active = false;
    };
  }, [
    latitude,
    longitude,
    forecastDay,
  ]);

  /*
   * Evacuation.
   */
  async function handleFindSafestShelter() {
    setEvacuationLoading(
      true,
    );

    setPageError(null);

    try {
      const response =
        await evacuate(
          latitude,
          longitude,
          FORECAST_SAMPLE,
          forecastDay,
          5,
        );

      setEvacuation(
        response,
      );
    } catch (error) {
      setEvacuation(null);

      setPageError(
        error instanceof Error
          ? error.message
          : "Unable to calculate evacuation.",
      );
    } finally {
      setEvacuationLoading(
        false,
      );
    }
  }

  /*
   * Search result selection.
   */
  function selectLocation(
    result: GeocodingResult,
  ) {
    setLatitude(
      result.latitude,
    );

    setLongitude(
      result.longitude,
    );

    setLatitudeInput(
      result.latitude.toFixed(
        5,
      ),
    );

    setLongitudeInput(
      result.longitude.toFixed(
        5,
      ),
    );

    setLocationName(
      result.name ||
        result.address,
    );

    setSearchText(
      result.name ||
        result.address,
    );

    setSearchResults([]);

    setShowSuggestions(
      false,
    );

    setPageError(null);
  }

  /*
   * Map click.
   */
  function handleMapClick(
    nextLatitude: number,
    nextLongitude: number,
  ) {
    setLatitude(
      nextLatitude,
    );

    setLongitude(
      nextLongitude,
    );

    setLatitudeInput(
      nextLatitude.toFixed(
        5,
      ),
    );

    setLongitudeInput(
      nextLongitude.toFixed(
        5,
      ),
    );

    setLocationName(
      "Selected map location",
    );

    setSearchText("");

    setSearchResults([]);

    setShowSuggestions(
      false,
    );
  }

  /*
   * Manual coordinates.
   */
  function applyCoordinates() {
    const nextLatitude =
      Number(
        latitudeInput,
      );

    const nextLongitude =
      Number(
        longitudeInput,
      );

    if (
      !Number.isFinite(
        nextLatitude,
      ) ||
      !Number.isFinite(
        nextLongitude,
      ) ||
      nextLatitude < -90 ||
      nextLatitude > 90 ||
      nextLongitude < -180 ||
      nextLongitude > 180
    ) {
      setPageError(
        "Enter valid latitude and longitude values.",
      );

      return;
    }

    setLatitude(
      nextLatitude,
    );

    setLongitude(
      nextLongitude,
    );

    setLocationName(
      "Manual coordinate selection",
    );

    setPageError(null);
  }

  /*
   * Current browser location.
   */
  function useCurrentLocation() {
    if (
      !navigator.geolocation
    ) {
      setPageError(
        "Geolocation is not available in this browser.",
      );

      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const nextLatitude =
          position.coords.latitude;

        const nextLongitude =
          position.coords.longitude;

        setLatitude(
          nextLatitude,
        );

        setLongitude(
          nextLongitude,
        );

        setLatitudeInput(
          nextLatitude.toFixed(
            5,
          ),
        );

        setLongitudeInput(
          nextLongitude.toFixed(
            5,
          ),
        );

        setLocationName(
          "Current location",
        );

        setSearchText("");

        setSearchResults([]);

        setShowSuggestions(
          false,
        );

        setPageError(null);
      },
      () => {
        setPageError(
          "Unable to access your current location.",
        );
      },
      {
        enableHighAccuracy: true,
        timeout: 10_000,
        maximumAge: 60_000,
      },
    );
  }

  return (
    <div className="app-shell">
      <Header
        backendOnline={
          backendOnline
        }
      />

      <main className="dashboard">
        <section className="toolbar-card">
          <div className="toolbar-main">
            <div className="search-wrap">
              <div className="place-search">
                <Search
                  size={16}
                  strokeWidth={2}
                />

                <input
                  value={
                    searchText
                  }
                  onChange={(
                    event,
                  ) => {
                    setSearchText(
                      event.target
                        .value,
                    );

                    setSearchError(
                      null,
                    );

                    setShowSuggestions(
                      true,
                    );
                  }}
                  onFocus={() => {
                    if (
                      searchText.trim()
                        .length >=
                      2
                    ) {
                      setShowSuggestions(
                        true,
                      );
                    }
                  }}
                  placeholder="Search a place"
                  aria-label="Search a place"
                  autoComplete="off"
                />

                {searchLoading && (
                  <LoaderCircle
                    size={15}
                    className="spin"
                  />
                )}
              </div>

              {showSuggestions &&
                searchText.trim()
                  .length >= 2 && (
                  <div className="search-dropdown">
                    {searchLoading &&
                      searchResults.length ===
                        0 && (
                        <div className="search-status">
                          <LoaderCircle
                            size={14}
                            className="spin"
                          />

                          Searching places…
                        </div>
                      )}

                    {!searchLoading &&
                      searchError && (
                        <div className="search-status search-error">
                          {searchError}
                        </div>
                      )}

                    {!searchLoading &&
                      !searchError &&
                      searchResults.length ===
                        0 && (
                        <div className="search-status">
                          No places found
                        </div>
                      )}

                    {searchResults.map(
                      (
                        result,
                      ) => (
                        <button
                          key={
                            result.id
                          }
                          type="button"
                          className={`search-result ${
                            result.trusted
                              ? "search-result-trusted"
                              : ""
                          }`}
                          onMouseDown={(
                            event,
                          ) =>
                            event.preventDefault()
                          }
                          onClick={() =>
                            selectLocation(
                              result,
                            )
                          }
                        >
                          <span className="result-icon">
                            <MapPin
                              size={
                                15
                              }
                            />
                          </span>

                          <span className="result-copy">
                            <strong>
                              {
                                result.name
                              }
                            </strong>

                            <small>
                              {
                                result.address
                              }
                            </small>

                            {result.trusted && (
                              <span className="trusted-badge">
                                VERIFIED PLACE
                              </span>
                            )}
                          </span>
                        </button>
                      ),
                    )}
                  </div>
                )}
            </div>

            <div className="forecast-selector">
              <span className="forecast-label">
                Forecast
              </span>

              <div className="select-wrap">
                <select
                  value={
                    forecastDays.includes(
                      forecastDay,
                    )
                      ? forecastDay
                      : forecastDays[0]
                  }
                  onChange={(
                    event,
                  ) =>
                    setForecastDay(
                      Number(
                        event.target
                          .value,
                      ),
                    )
                  }
                >
                  {forecastDays.map(
                    (day) => (
                      <option
                        key={day}
                        value={day}
                      >
                        Day {day}
                      </option>
                    ),
                  )}
                </select>

                <ChevronDown
                  size={14}
                />
              </div>
            </div>

            <button
              type="button"
              className="toolbar-button secondary-button"
              onClick={
                useCurrentLocation
              }
            >
              <Crosshair
                size={15}
              />

              My location
            </button>

            <button
              type="button"
              className="toolbar-button primary-button"
              onClick={
                handleFindSafestShelter
              }
              disabled={
                evacuationLoading ||
                hazardLoading
              }
            >
              {evacuationLoading ? (
                <LoaderCircle
                  size={15}
                  className="spin"
                />
              ) : (
                <Navigation
                  size={15}
                />
              )}

              {evacuationLoading
                ? "Planning…"
                : "Find safest shelter"}
            </button>
          </div>

          <div className="coordinate-row">
            <div className="coordinate-field">
              <label>
                Latitude
              </label>

              <input
                value={
                  latitudeInput
                }
                onChange={(
                  event,
                ) =>
                  setLatitudeInput(
                    event.target
                      .value,
                  )
                }
              />
            </div>

            <div className="coordinate-field">
              <label>
                Longitude
              </label>

              <input
                value={
                  longitudeInput
                }
                onChange={(
                  event,
                ) =>
                  setLongitudeInput(
                    event.target
                      .value,
                  )
                }
              />
            </div>

            <button
              type="button"
              className="coordinate-apply"
              onClick={
                applyCoordinates
              }
            >
              Apply
            </button>

            <div className="selected-location-bar">
              <MapPin
                size={13}
              />

              <span>
                {locationName}
              </span>
            </div>
          </div>
        </section>

        {pageError && (
          <div className="error-banner">
            <strong>
              Forecast update
            </strong>

            <span>
              {pageError}
            </span>
          </div>
        )}

        <section className="risk-strip">
          <RiskSummary
            hazard={
              hazard?.hazard_score ??
              null
            }
            uncertainty={
              evacuation?.result
                ?.route?.statistics
                ?.mean_uncertainty_risk ??
              null
            }
            routeRisk={
              evacuation?.result
                ?.route?.statistics
                ?.mean_total_risk ??
              null
            }
            forecastDay={
              forecastDay
            }
          />
        </section>

        <section className="workspace">
          <div className="map-card">
            <div className="map-card-header">
              <div>
                <span className="eyebrow">
                  FLOOD RISK MAP
                </span>

                <h2>
                  Forecast spatial risk
                </h2>
              </div>

              <div className="map-status">
                <span className="status-dot" />

                {gridLoading
                  ? "Updating risk surface"
                  : `Forecast Day ${forecastDay}`}
              </div>
            </div>

            <MapView
              latitude={
                latitude
              }
              longitude={
                longitude
              }
              shelters={
                shelters
              }
              evacuation={
                evacuation
              }
              hazardScore={
                hazard?.hazard_score ??
                null
              }
              riskGrid={
                riskGrid
              }
              onMapClick={
                handleMapClick
              }
            />
          </div>

          <aside className="side-panel">
            <div className="side-panel-top">
              <div>
                <span className="eyebrow">
                  DECISION SUPPORT
                </span>

                <h2>
                  Evacuation plan
                </h2>
              </div>

              <ShieldCheck
                size={20}
              />
            </div>

            <EvacuationPanel
              hazard={
                hazard
              }
              evacuation={
                evacuation
              }
              loading={
                evacuationLoading
              }
              onFindSafestShelter={
                handleFindSafestShelter
              }
            />
          </aside>
        </section>
      </main>
    </div>
  );
}
