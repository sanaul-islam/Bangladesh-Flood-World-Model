import {
  findTrustedPlaces,
  type TrustedPlace,
} from "./trustedPlaces";

export interface GeocodingResult {
  id: string;

  name: string;
  address: string;

  latitude: number;
  longitude: number;

  trusted?: boolean;
}

interface GeoapifyFeature {
  properties?: {
    place_id?: string;

    name?: string;

    street?: string;
    housenumber?: string;

    city?: string;
    state?: string;
    postcode?: string;
    country?: string;

    formatted?: string;

    lat?: number;
    lon?: number;
  };
}

interface GeoapifyResponse {
  features?: GeoapifyFeature[];
}

const GEOAPIFY_URL =
  "https://api.geoapify.com/v1/geocode/autocomplete";

const MIN_QUERY_LENGTH = 2;
const REQUEST_LIMIT = 6;

function normalizeText(
  value: string,
): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");
}

function trustedToResult(
  place: TrustedPlace,
): GeocodingResult {
  return {
    id: place.id,
    name: place.name,
    address: place.address,

    latitude:
      place.latitude,

    longitude:
      place.longitude,

    trusted: true,
  };
}

function normalizeGeoapifyResult(
  feature: GeoapifyFeature,
  index: number,
): GeocodingResult | null {
  const properties =
    feature.properties;

  if (!properties) {
    return null;
  }

  const latitude =
    Number(properties.lat);

  const longitude =
    Number(properties.lon);

  if (
    !Number.isFinite(latitude) ||
    !Number.isFinite(longitude)
  ) {
    return null;
  }

  const name =
    properties.name?.trim() ||
    properties.formatted
      ?.split(",")[0]
      ?.trim() ||
    `Place ${index + 1}`;

  const address =
    properties.formatted?.trim() ||
    [
      properties.street &&
      properties.housenumber
        ? `${properties.housenumber} ${properties.street}`
        : properties.street,

      properties.city,
      properties.state,
      properties.postcode,
      properties.country,
    ]
      .filter(Boolean)
      .join(", ");

  return {
    id:
      properties.place_id ||
      `geoapify-${index}-${latitude}-${longitude}`,

    name,

    address:
      address || name,

    latitude,
    longitude,

    trusted: false,
  };
}

function deduplicateResults(
  results: GeocodingResult[],
): GeocodingResult[] {
  const seen = new Set<string>();
  const output: GeocodingResult[] = [];

  for (const result of results) {
    const key =
      `${result.name}|${result.latitude.toFixed(5)}|${result.longitude.toFixed(5)}`
        .toLowerCase();

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    output.push(result);
  }

  return output;
}

async function searchGeoapify(
  query: string,
  latitude?: number,
  longitude?: number,
): Promise<GeocodingResult[]> {
  const apiKey =
    import.meta.env
      .VITE_GEOAPIFY_API_KEY;

  if (!apiKey) {
    throw new Error(
      "VITE_GEOAPIFY_API_KEY is not configured.",
    );
  }

  const params =
    new URLSearchParams({
      text: query,

      apiKey,

      limit: String(
        REQUEST_LIMIT,
      ),

      lang: "en",
    });

  if (
    Number.isFinite(latitude) &&
    Number.isFinite(longitude)
  ) {
    params.set(
      "bias",
      `proximity:${longitude},${latitude}`,
    );
  }

  const response =
    await fetch(
      `${GEOAPIFY_URL}?${params.toString()}`,
    );

  if (!response.ok) {
    throw new Error(
      `Place search failed (${response.status}).`,
    );
  }

  const data =
    (await response.json()) as GeoapifyResponse;

  const results =
    (data.features ?? [])
      .map(
        normalizeGeoapifyResult,
      )
      .filter(
        (
          result,
        ): result is GeocodingResult =>
          result !== null,
      );

  return deduplicateResults(
    results,
  );
}

export async function searchPlaces(
  query: string,
  latitude?: number,
  longitude?: number,
): Promise<GeocodingResult[]> {
  const normalized =
    normalizeText(query);

  if (
    normalized.length <
    MIN_QUERY_LENGTH
  ) {
    return [];
  }

  /*
   * Trusted places have priority.
   */
  const trusted =
    findTrustedPlaces(
      normalized,
    );

  if (trusted.length > 0) {
    return trusted.map(
      trustedToResult,
    );
  }

  return searchGeoapify(
    query,
    latitude,
    longitude,
  );
}
