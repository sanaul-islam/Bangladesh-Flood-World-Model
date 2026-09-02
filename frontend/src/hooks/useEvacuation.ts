import { useCallback, useState } from "react";
import { evacuate, getHazard } from "../api/client";
import type {
  EvacuationResponse,
  HazardResponse,
} from "../api/types";

export function useEvacuation() {
  const [hazard, setHazard] = useState<HazardResponse | null>(null);
  const [evacuation, setEvacuation] =
    useState<EvacuationResponse | null>(null);
  const [loadingHazard, setLoadingHazard] = useState(false);
  const [loadingEvacuation, setLoadingEvacuation] =
    useState(false);
  const [error, setError] = useState<string | null>(null);

  const inspectLocation = useCallback(
    async (
      latitude: number,
      longitude: number,
      forecastSample: number,
      forecastDay: number,
    ) => {
      setLoadingHazard(true);
      setError(null);

      try {
        const result = await getHazard(
          latitude,
          longitude,
          forecastSample,
          forecastDay,
        );

        setHazard(result);
        setEvacuation(null);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Unable to inspect this location.",
        );
      } finally {
        setLoadingHazard(false);
      }
    },
    [],
  );

  const findSafestShelter = useCallback(
    async (
      latitude: number,
      longitude: number,
      forecastSample: number,
      forecastDay: number,
    ) => {
      setLoadingEvacuation(true);
      setError(null);

      try {
        const result = await evacuate(
          latitude,
          longitude,
          forecastSample,
          forecastDay,
          5,
        );

        setEvacuation(result);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Unable to calculate evacuation.",
        );
      } finally {
        setLoadingEvacuation(false);
      }
    },
    [],
  );

  return {
    hazard,
    evacuation,
    loadingHazard,
    loadingEvacuation,
    error,
    inspectLocation,
    findSafestShelter,
  };
}
