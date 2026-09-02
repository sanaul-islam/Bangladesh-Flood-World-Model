import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";

maplibregl.setWorkerUrl(workerUrl);

type Coordinate = [number, number];
type DataRecord = Record<string, unknown>;

interface MapViewProps {
  latitude?: number | string;
  longitude?: number | string;

  selectedLocation?: unknown;
  location?: unknown;

  shelters?: unknown;
  shelterData?: unknown;

  evacuation?: unknown;

  hazardScore?: number | null;
  riskGrid?: unknown;
  hazardGrid?: unknown;
  floodRiskGrid?: unknown;

  onMapClick?: (
    latitude: number,
    longitude: number,
  ) => void;

  flyTo?: unknown;
  flyToZoom?: number;
}

const DEFAULT_CENTER: Coordinate = [
  90.4125,
  23.8103,
];

const DEFAULT_ZOOM = 10.5;

const ROUTE_SOURCE_ID =
  "fwm-route-source";

const ROUTE_OUTLINE_ID =
  "fwm-route-outline";

const ROUTE_CORE_ID =
  "fwm-route-core";

const RISK_SOURCE_ID =
  "fwm-risk-source";

const RISK_LAYER_ID =
  "fwm-risk-layer";

function asRecord(
  value: unknown,
): DataRecord | null {
  if (
    typeof value === "object" &&
    value !== null
  ) {
    return value as DataRecord;
  }

  return null;
}

function toNumber(
  value: unknown,
): number | null {
  if (
    typeof value === "number" &&
    Number.isFinite(value)
  ) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number(value);

    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return null;
}

function getCoordinate(
  value: unknown,
): Coordinate | null {
  if (
    Array.isArray(value) &&
    value.length >= 2
  ) {
    const longitude =
      toNumber(value[0]);

    const latitude =
      toNumber(value[1]);

    if (
      longitude !== null &&
      latitude !== null
    ) {
      return [
        longitude,
        latitude,
      ];
    }
  }

  const item =
    asRecord(value);

  if (!item) {
    return null;
  }

  const direct =
    getCoordinate(
      item.coordinates,
    );

  if (direct) {
    return direct;
  }

  const location =
    asRecord(item.location);

  const geometry =
    asRecord(item.geometry);

  const nested =
    getCoordinate(
      location?.coordinates,
    ) ??
    getCoordinate(
      geometry?.coordinates,
    );

  if (nested) {
    return nested;
  }

  const longitude =
    toNumber(item.longitude) ??
    toNumber(item.lon) ??
    toNumber(item.lng);

  const latitude =
    toNumber(item.latitude) ??
    toNumber(item.lat);

  if (
    longitude !== null &&
    latitude !== null
  ) {
    return [
      longitude,
      latitude,
    ];
  }

  return null;
}

function getResult(
  evacuation: unknown,
): DataRecord | null {
  const root =
    asRecord(evacuation);

  if (!root) {
    return null;
  }

  return (
    asRecord(root.result) ??
    root
  );
}

function getRecommendedShelter(
  evacuation: unknown,
): DataRecord | null {
  const result =
    getResult(evacuation);

  if (!result) {
    return null;
  }

  return (
    asRecord(
      result.recommended_shelter,
    ) ??
    asRecord(
      result.recommendedShelter,
    ) ??
    asRecord(
      result.recommendation,
    ) ??
    asRecord(
      result.best_shelter,
    )
  );
}

function getRoute(
  evacuation: unknown,
): Coordinate[] {
  const root =
    asRecord(evacuation);

  if (!root) {
    return [];
  }

  const result =
    asRecord(root.result);

  const resultRoute =
    asRecord(result?.route);

  const rootRoute =
    asRecord(root.route);

  const candidates: unknown[] = [
    asRecord(resultRoute?.route)
      ?.coordinates,

    resultRoute?.coordinates,

    asRecord(rootRoute?.route)
      ?.coordinates,

    rootRoute?.coordinates,

    asRecord(resultRoute?.geometry)
      ?.coordinates,

    asRecord(rootRoute?.geometry)
      ?.coordinates,
  ];

  for (const candidate of candidates) {
    if (!Array.isArray(candidate)) {
      continue;
    }

    const coordinates =
      candidate
        .map(getCoordinate)
        .filter(
          (
            point,
          ): point is Coordinate =>
            point !== null,
        );

    if (coordinates.length >= 2) {
      return coordinates;
    }
  }

  return [];
}

function propertyNumber(
  item: DataRecord,
  names: string[],
): number | null {
  for (const name of names) {
    const value =
      toNumber(item[name]);

    if (value !== null) {
      return value;
    }
  }

  return null;
}

function shelterName(
  shelter: DataRecord,
): string {
  const values = [
    shelter.name,
    shelter.shelter_name,
    shelter.shelterName,
    shelter.title,
  ];

  for (const value of values) {
    if (
      typeof value === "string" &&
      value.trim()
    ) {
      return value;
    }
  }

  return "Evacuation shelter";
}

function shelterId(
  shelter: DataRecord,
): string {
  const values = [
    shelter.id,
    shelter.shelter_id,
    shelter.shelterId,
    shelter.code,
  ];

  for (const value of values) {
    if (
      typeof value === "string" ||
      typeof value === "number"
    ) {
      return String(value);
    }
  }

  return "—";
}

function escapeHtml(
  value: string,
): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function userMarkerHtml(): string {
  return `
    <div class="fwm-user-marker">
      <div class="fwm-user-pulse"></div>
      <div class="fwm-user-core">
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
        >
          <path
            d="M12 2.8C7.9 2.8 4.6 6.1 4.6 10.2
               C4.6 15.4 12 21.2 12 21.2
               C12 21.2 19.4 15.4 19.4 10.2
               C19.4 6.1 16.1 2.8 12 2.8Z"
            fill="#2563eb"
          />
          <circle
            cx="12"
            cy="10.2"
            r="3.2"
            fill="white"
          />
        </svg>
      </div>
    </div>
  `;
}

function shelterMarkerHtml(
  recommended: boolean,
): string {
  const color =
    recommended
      ? "#dc2626"
      : "#16a34a";

  return `
    <div class="${
      recommended
        ? "fwm-shelter-marker fwm-shelter-recommended"
        : "fwm-shelter-marker"
    }">
      <svg
        width="${recommended ? 32 : 26}"
        height="${recommended ? 32 : 26}"
        viewBox="0 0 32 32"
        fill="none"
      >
        <path
          d="M4.5 14L16 4.5L27.5 14V27.5H4.5V14Z"
          fill="${color}"
          stroke="white"
          stroke-width="2"
          stroke-linejoin="round"
        />
        <path
          d="M11 27.5V18H21V27.5"
          fill="white"
        />
        <path
          d="M8 13.5H24"
          stroke="white"
          stroke-width="1.8"
          stroke-linecap="round"
        />
      </svg>
    </div>
  `;
}

function startMarkerHtml(): string {
  return `
    <div class="fwm-route-origin">
      <span></span>
    </div>
  `;
}

function popupHtml(
  shelter: DataRecord,
  recommended: boolean,
): string {
  const name =
    escapeHtml(
      shelterName(shelter),
    );

  const id =
    escapeHtml(
      shelterId(shelter),
    );

  const risk =
    propertyNumber(
      shelter,
      [
        "hazard_score",
        "hazardScore",
        "risk",
        "mean_flood_risk",
      ],
    );

  const distance =
    propertyNumber(
      shelter,
      [
        "road_distance_km",
        "distance_km",
        "distance",
      ],
    );

  return `
    <div class="fwm-popup">
      <div class="fwm-popup-kicker ${
        recommended
          ? "is-recommended"
          : ""
      }">
        ${
          recommended
            ? "RECOMMENDED SHELTER"
            : "EVACUATION SHELTER"
        }
      </div>

      <div class="fwm-popup-title">
        ${name}
      </div>

      <div class="fwm-popup-id">
        Facility ID ${id}
      </div>

      ${
        risk !== null
          ? `
            <div class="fwm-popup-row">
              <span>Flood risk</span>
              <strong>${risk.toFixed(3)}</strong>
            </div>
          `
          : ""
      }

      ${
        distance !== null
          ? `
            <div class="fwm-popup-row">
              <span>Road distance</span>
              <strong>${distance.toFixed(2)} km</strong>
            </div>
          `
          : ""
      }
    </div>
  `;
}

function riskGeoJson(
  value: unknown,
): {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    properties: {
      hazard_score: number;
    };
    geometry: {
      type: "Point";
      coordinates: Coordinate;
    };
  }>;
} {
  const root =
    asRecord(value);

  let rawPoints: unknown[] = [];

  if (Array.isArray(root?.points)) {
    rawPoints =
      root.points;
  } else {
    const data =
      asRecord(root?.data);

    if (
      Array.isArray(data?.points)
    ) {
      rawPoints =
        data.points;
    }
  }

  const features =
    rawPoints
      .map(asRecord)
      .filter(
        (
          point,
        ): point is DataRecord =>
          point !== null,
      )
      .map((point) => {
        const latitude =
          propertyNumber(
            point,
            [
              "latitude",
              "lat",
              "y",
            ],
          );

        const longitude =
          propertyNumber(
            point,
            [
              "longitude",
              "lon",
              "lng",
              "x",
            ],
          );

        const risk =
          propertyNumber(
            point,
            [
              "hazard_score",
              "hazardScore",
              "risk",
              "risk_score",
              "value",
            ],
          );

        if (
          latitude === null ||
          longitude === null ||
          risk === null
        ) {
          return null;
        }

        return {
          type: "Feature" as const,
          properties: {
            hazard_score:
              Math.max(
                0,
                Math.min(1, risk),
              ),
          },
          geometry: {
            type: "Point" as const,
            coordinates: [
              longitude,
              latitude,
            ] as Coordinate,
          },
        };
      })
      .filter(
        (
          feature,
        ): feature is {
          type: "Feature";
          properties: {
            hazard_score: number;
          };
          geometry: {
            type: "Point";
            coordinates: Coordinate;
          };
        } =>
          feature !== null,
      );

  return {
    type: "FeatureCollection",
    features,
  };
}

export function MapView({
  latitude,
  longitude,
  selectedLocation,
  location,
  shelters: sheltersProp,
  shelterData,
  evacuation,
  riskGrid,
  hazardGrid,
  floodRiskGrid,
  onMapClick,
  flyTo,
  flyToZoom,
}: MapViewProps) {
  const containerRef =
    useRef<HTMLDivElement | null>(
      null,
    );

  const mapRef =
    useRef<maplibregl.Map | null>(
      null,
    );

  const shelterMarkersRef =
    useRef<maplibregl.Marker[]>([]);

  const routeMarkersRef =
    useRef<maplibregl.Marker[]>([]);

  const [
    mapReady,
    setMapReady,
  ] = useState(false);

  const [
    clickedLocation,
    setClickedLocation,
  ] = useState<Coordinate | null>(
    null,
  );

  const locationRecord =
    asRecord(
      selectedLocation,
    ) ??
    asRecord(location);

  const baseLatitude =
    toNumber(latitude) ??
    toNumber(locationRecord?.latitude) ??
    toNumber(locationRecord?.lat) ??
    23.8103;

  const baseLongitude =
    toNumber(longitude) ??
    toNumber(locationRecord?.longitude) ??
    toNumber(locationRecord?.lon) ??
    toNumber(locationRecord?.lng) ??
    90.4125;

  const displayLongitude =
    clickedLocation?.[0] ??
    baseLongitude;

  const displayLatitude =
    clickedLocation?.[1] ??
    baseLatitude;

  const effectiveRiskGrid =
    riskGrid ??
    hazardGrid ??
    floodRiskGrid ??
    null;

  /*
   * Keep shelter normalization inside the shelter effect.
   * This avoids unstable derived-array hook dependencies.
   */

  /* ---------------------------------------------------------------------- */
  /* Map creation                                                            */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    if (
      !containerRef.current ||
      mapRef.current
    ) {
      return;
    }

    const key =
      import.meta.env
        .VITE_GEOAPIFY_API_KEY as
        | string
        | undefined;

    const style =
      key
        ? `https://maps.geoapify.com/v1/styles/osm-bright-smooth/style.json?apiKey=${key}`
        : "https://tiles.openfreemap.org/styles/liberty";

    const map =
      new maplibregl.Map({
        container:
          containerRef.current,
        style,
        center:
          DEFAULT_CENTER,
        zoom:
          DEFAULT_ZOOM,
        attributionControl:
          false,
      });

    mapRef.current =
      map;

    const handleLoad =
      () => {
        if (
          !map.getSource(
            ROUTE_SOURCE_ID,
          )
        ) {
          map.addSource(
            ROUTE_SOURCE_ID,
            {
              type: "geojson",
              data: {
                type: "Feature",
                properties: {},
                geometry: {
                  type: "LineString",
                  coordinates: [],
                },
              },
            },
          );
        }

        if (
          !map.getLayer(
            ROUTE_OUTLINE_ID,
          )
        ) {
          map.addLayer({
            id:
              ROUTE_OUTLINE_ID,
            type: "line",
            source:
              ROUTE_SOURCE_ID,
            layout: {
              "line-cap":
                "round",
              "line-join":
                "round",
            },
            paint: {
              "line-color":
                "#0f172a",
              "line-width":
                9,
              "line-opacity":
                0.72,
            },
          });
        }

        if (
          !map.getLayer(
            ROUTE_CORE_ID,
          )
        ) {
          map.addLayer({
            id:
              ROUTE_CORE_ID,
            type: "line",
            source:
              ROUTE_SOURCE_ID,
            layout: {
              "line-cap":
                "round",
              "line-join":
                "round",
            },
            paint: {
              "line-color":
                "#2563eb",
              "line-width":
                5,
              "line-opacity":
                1,
            },
          });
        }

        if (
          !map.getSource(
            RISK_SOURCE_ID,
          )
        ) {
          map.addSource(
            RISK_SOURCE_ID,
            {
              type: "geojson",
              data:
                riskGeoJson(
                  null,
                ),
            },
          );
        }

        if (
          !map.getLayer(
            RISK_LAYER_ID,
          )
        ) {
          map.addLayer({
            id:
              RISK_LAYER_ID,
            type: "heatmap",
            source:
              RISK_SOURCE_ID,
            paint: {
              "heatmap-weight": [
                "interpolate",
                ["linear"],
                [
                  "get",
                  "hazard_score",
                ],
                0,
                0,
                0.3,
                0.2,
                0.6,
                0.65,
                1,
                1,
              ],

              "heatmap-intensity": [
                "interpolate",
                ["linear"],
                ["zoom"],
                5,
                0.8,
                10,
                1.2,
                14,
                1.5,
              ],

              "heatmap-radius": [
                "interpolate",
                ["linear"],
                ["zoom"],
                5,
                18,
                10,
                30,
                14,
                42,
              ],

              "heatmap-opacity":
                0.42,

              "heatmap-color": [
                "interpolate",
                ["linear"],
                [
                  "heatmap-density",
                ],
                0,
                "rgba(59,130,246,0)",
                0.25,
                "rgba(34,197,94,0.18)",
                0.5,
                "rgba(250,204,21,0.30)",
                0.7,
                "rgba(249,115,22,0.45)",
                0.85,
                "rgba(239,68,68,0.58)",
                1,
                "rgba(127,29,29,0.68)",
              ],
            },
          });
        }

        setMapReady(true);

        window.setTimeout(
          () => {
            map.resize();
          },
          100,
        );
      };

    map.once(
      "load",
      handleLoad,
    );

    const resizeObserver =
      new ResizeObserver(
        () => {
          map.resize();
        },
      );

    resizeObserver.observe(
      containerRef.current,
    );

    return () => {
      resizeObserver.disconnect();

      shelterMarkersRef.current.forEach(
        (marker) =>
          marker.remove(),
      );

      routeMarkersRef.current.forEach(
        (marker) =>
          marker.remove(),
      );

      shelterMarkersRef.current = [];
      routeMarkersRef.current = [];

      map.remove();

      mapRef.current =
        null;

      setMapReady(false);
    };
  }, []);

  /* ---------------------------------------------------------------------- */
  /* Map click                                                               */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    if (
      !mapReady ||
      !mapRef.current ||
      !onMapClick
    ) {
      return;
    }

    const map =
      mapRef.current;

    const handleClick =
      (
        event: maplibregl.MapMouseEvent,
      ) => {
        const nextLongitude =
          event.lngLat.lng;

        const nextLatitude =
          event.lngLat.lat;

        setClickedLocation([
          nextLongitude,
          nextLatitude,
        ]);

        onMapClick(
          nextLatitude,
          nextLongitude,
        );
      };

    map.on(
      "click",
      handleClick,
    );

    map.getCanvas().style.cursor =
      "crosshair";

    return () => {
      map.off(
        "click",
        handleClick,
      );

      map.getCanvas().style.cursor =
        "";
    };
  }, [
    mapReady,
    onMapClick,
  ]);

  /* ---------------------------------------------------------------------- */
  /* Selected location marker                                                */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    if (
      !mapReady ||
      !mapRef.current
    ) {
      return;
    }

    const map =
      mapRef.current;

    const element =
      document.createElement(
        "div",
      );

    element.innerHTML =
      userMarkerHtml();

    const marker =
      new maplibregl.Marker({
        element,
        anchor:
          "center",
      })
        .setLngLat([
          displayLongitude,
          displayLatitude,
        ])
        .addTo(map);

    return () => {
      marker.remove();
    };
  }, [
    mapReady,
    displayLongitude,
    displayLatitude,
  ]);

  /* ---------------------------------------------------------------------- */
  /* Shelters                                                                */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    if (
      !mapReady ||
      !mapRef.current
    ) {
      return;
    }

    const map =
      mapRef.current;

    const rawShelters =
      sheltersProp ??
      asRecord(
        shelterData,
      )?.shelters ??
      shelterData ??
      [];

    const shelters: DataRecord[] =
      Array.isArray(rawShelters)
        ? rawShelters
            .map(asRecord)
            .filter(
              (
                item,
              ): item is DataRecord =>
                item !== null,
            )
        : [];

    shelterMarkersRef.current.forEach(
      (marker) =>
        marker.remove(),
    );

    shelterMarkersRef.current = [];

    const recommended =
      getRecommendedShelter(
        evacuation,
      );

    const recommendedPoint =
      recommended
        ? getCoordinate(
            recommended,
          )
        : null;

    shelters.forEach(
      (shelter) => {
        const point =
          getCoordinate(
            shelter,
          );

        if (!point) {
          return;
        }

        const isRecommended =
          recommendedPoint !==
            null &&
          Math.abs(
            point[0] -
              recommendedPoint[0],
          ) < 0.000001 &&
          Math.abs(
            point[1] -
              recommendedPoint[1],
          ) < 0.000001;

        const element =
          document.createElement(
            "div",
          );

        element.innerHTML =
          shelterMarkerHtml(
            isRecommended,
          );

        const marker =
          new maplibregl.Marker({
            element,
            anchor:
              "center",
          })
            .setLngLat(point)
            .setPopup(
              new maplibregl.Popup({
                offset: 18,
                maxWidth:
                  "300px",
              }).setHTML(
                popupHtml(
                  shelter,
                  isRecommended,
                ),
              ),
            )
            .addTo(map);

        shelterMarkersRef.current.push(
          marker,
        );
      },
    );

    return () => {
      shelterMarkersRef.current.forEach(
        (marker) =>
          marker.remove(),
      );

      shelterMarkersRef.current = [];
    };
  }, [
    mapReady,
    sheltersProp,
    shelterData,
    evacuation,
  ]);

  /* ---------------------------------------------------------------------- */
  /* Risk grid                                                               */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    if (
      !mapReady ||
      !mapRef.current
    ) {
      return;
    }

    const source =
      mapRef.current.getSource(
        RISK_SOURCE_ID,
      );

    if (
      !(
        source instanceof
        maplibregl.GeoJSONSource
      )
    ) {
      return;
    }

    source.setData(
      riskGeoJson(
        effectiveRiskGrid,
      ),
    );
  }, [
    mapReady,
    effectiveRiskGrid,
  ]);

  /* ---------------------------------------------------------------------- */
  /* Evacuation route                                                        */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    if (
      !mapReady ||
      !mapRef.current
    ) {
      return;
    }

    const map =
      mapRef.current;

    const source =
      map.getSource(
        ROUTE_SOURCE_ID,
      );

    if (
      !(
        source instanceof
        maplibregl.GeoJSONSource
      )
    ) {
      return;
    }

    const route =
      getRoute(
        evacuation,
      );

    source.setData({
      type: "Feature",
      properties: {},
      geometry: {
        type: "LineString",
        coordinates: route,
      },
    });

    routeMarkersRef.current.forEach(
      (marker) =>
        marker.remove(),
    );

    routeMarkersRef.current = [];

    const result =
      getResult(
        evacuation,
      );

    const routeObject =
      asRecord(
        result?.route,
      );

    const start =
      getCoordinate(
        routeObject?.start,
      ) ??
      [
        displayLongitude,
        displayLatitude,
      ];

    const startElement =
      document.createElement(
        "div",
      );

    startElement.innerHTML =
      startMarkerHtml();

    const startMarker =
      new maplibregl.Marker({
        element:
          startElement,
        anchor:
          "center",
      })
        .setLngLat(start)
        .addTo(map);

    routeMarkersRef.current.push(
      startMarker,
    );

    const recommended =
      getRecommendedShelter(
        evacuation,
      );

    if (recommended) {
      const destination =
        getCoordinate(
          recommended,
        );

      if (destination) {
        const destinationElement =
          document.createElement(
            "div",
          );

        destinationElement.innerHTML =
          shelterMarkerHtml(
            true,
          );

        const destinationMarker =
          new maplibregl.Marker({
            element:
              destinationElement,
            anchor:
              "center",
          })
            .setLngLat(
              destination,
            )
            .setPopup(
              new maplibregl.Popup({
                offset: 20,
                maxWidth:
                  "300px",
              }).setHTML(
                popupHtml(
                  recommended,
                  true,
                ),
              ),
            )
            .addTo(map);

        routeMarkersRef.current.push(
          destinationMarker,
        );
      }
    }

    if (route.length >= 2) {
      const bounds =
        new maplibregl.LngLatBounds();

      route.forEach(
        (point) =>
          bounds.extend(point),
      );

      map.fitBounds(
        bounds,
        {
          padding: 90,
          maxZoom: 15,
          duration: 800,
        },
      );
    }

    return () => {
      routeMarkersRef.current.forEach(
        (marker) =>
          marker.remove(),
      );

      routeMarkersRef.current = [];
    };
  }, [
    mapReady,
    evacuation,
    displayLongitude,
    displayLatitude,
  ]);

  /* ---------------------------------------------------------------------- */
  /* External fly-to                                                         */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    if (
      !mapReady ||
      !mapRef.current
    ) {
      return;
    }

    const point =
      getCoordinate(
        flyTo,
      );

    if (!point) {
      return;
    }

    mapRef.current.flyTo({
      center: point,
      zoom:
        flyToZoom ?? 13,
      duration: 700,
    });
  }, [
    mapReady,
    flyTo,
    flyToZoom,
  ]);

  const riskActive =
    riskGeoJson(
      effectiveRiskGrid,
    ).features.length > 0;

  return (
    <div className="fwm-map-shell">
      <div
        ref={containerRef}
        className="fwm-map"
      />

      <div className="fwm-map-header">
        <div className="fwm-map-status">
          <span
            className={
              mapReady
                ? "fwm-status-dot fwm-status-live"
                : "fwm-status-dot"
            }
          />

          <div>
            <div className="fwm-map-title">
              Flood intelligence map
            </div>

            <div className="fwm-map-subtitle">
              {mapReady
                ? "Interactive decision-support view"
                : "Loading map…"}
            </div>
          </div>
        </div>

        {riskActive && (
          <div className="fwm-map-chip">
            Spatial risk active
          </div>
        )}
      </div>

      {mapReady && (
        <div className="fwm-click-hint">
          Click map to set location
        </div>
      )}

      <div className="fwm-map-legend">
        <div className="fwm-legend-heading">
          MAP LEGEND
        </div>

        <div className="fwm-legend-item">
          <span className="fwm-legend-house fwm-green" />
          <span>
            Evacuation shelter
          </span>
        </div>

        <div className="fwm-legend-item">
          <span className="fwm-legend-house fwm-red" />
          <span>
            Recommended shelter
          </span>
        </div>

        <div className="fwm-legend-item">
          <span className="fwm-legend-location" />
          <span>
            Selected location
          </span>
        </div>

        <div className="fwm-legend-item">
          <span className="fwm-legend-route" />
          <span>
            Evacuation route
          </span>
        </div>

        <div className="fwm-risk-scale">
          <span>Lower</span>

          <span className="fwm-risk-gradient" />

          <span>Higher</span>
        </div>
      </div>

      <div className="fwm-attribution">
        Flood World Model · OpenStreetMap
      </div>
    </div>
  );
}

export default MapView;
