import type {
  EvacuationResponse,
  ForecastResponse,
  HazardGridResponse,
  HazardResponse,
  HealthResponse,
  RouteResponse,
  SheltersResponse,
} from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "";

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options?.headers ?? {}),
      },
    },
  );

  const contentType =
    response.headers.get(
      "content-type",
    ) ?? "";

  if (
    !contentType.includes(
      "application/json",
    )
  ) {
    const text =
      await response.text();

    if (!response.ok) {
      throw new Error(
        `${response.status}: ${
          text || "Request failed."
        }`,
      );
    }

    return text as T;
  }

  const data =
    (await response.json()) as unknown;

  if (!response.ok) {
    let message =
      "Request failed.";

    if (
      typeof data === "object" &&
      data !== null &&
      "detail" in data
    ) {
      const detail = (
        data as {
          detail?: unknown;
        }
      ).detail;

      if (
        typeof detail ===
        "string"
      ) {
        message = detail;
      } else if (
        detail !== undefined
      ) {
        message =
          JSON.stringify(detail);
      }
    }

    throw new Error(
      `${response.status}: ${message}`,
    );
  }

  return data as T;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>(
    "/health",
  );
}

export async function getForecast(): Promise<ForecastResponse> {
  return request<ForecastResponse>(
    "/api/v1/forecast",
  );
}

export async function getHazard(
  latitude: number,
  longitude: number,
  forecastSample: number,
  forecastDay: number,
): Promise<HazardResponse> {
  const params =
    new URLSearchParams({
      latitude: String(latitude),
      longitude: String(longitude),
      forecast_sample: String(
        forecastSample,
      ),
      forecast_day: String(
        forecastDay,
      ),
    });

  return request<HazardResponse>(
    `/api/v1/hazard?${params.toString()}`,
  );
}

export async function getHazardGrid(
  latitude: number,
  longitude: number,
  forecastSample: number,
  forecastDay: number,
): Promise<HazardGridResponse> {
  const params =
    new URLSearchParams({
      center_latitude: String(
        latitude,
      ),
      center_longitude: String(
        longitude,
      ),
      forecast_sample: String(
        forecastSample,
      ),
      forecast_day: String(
        forecastDay,
      ),
      radius_degrees: "0.30",
      rows: "9",
      columns: "9",
    });

  return request<HazardGridResponse>(
    `/api/v1/hazard/grid?${params.toString()}`,
  );
}

export async function getShelters(): Promise<SheltersResponse> {
  return request<SheltersResponse>(
    "/api/v1/shelters",
  );
}

export async function route(
  startLatitude: number,
  startLongitude: number,
  goalLatitude: number,
  goalLongitude: number,
  maxDistanceKm = 30,
  maxExpandedNodes = 100_000,
): Promise<RouteResponse> {
  return request<RouteResponse>(
    "/api/v1/route",
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        start_latitude:
          startLatitude,
        start_longitude:
          startLongitude,
        goal_latitude:
          goalLatitude,
        goal_longitude:
          goalLongitude,
        max_distance_km:
          maxDistanceKm,
        max_expanded_nodes:
          maxExpandedNodes,
      }),
    },
  );
}

export async function evacuate(
  latitude: number,
  longitude: number,
  forecastSample: number,
  forecastDay: number,
  candidateShelters = 5,
): Promise<EvacuationResponse> {
  return request<EvacuationResponse>(
    "/api/v1/evacuate",
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        latitude,
        longitude,
        forecast_sample:
          forecastSample,
        forecast_day:
          forecastDay,
        candidate_shelters:
          candidateShelters,
      }),
    },
  );
}
