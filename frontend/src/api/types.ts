export interface ResponseMetadata {
  api_version: string;
  model_version: string;
  data_version: string;
  generated_at: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface ForecastResponse {
  forecast_sample: number;
  forecast_days: number[];

  latitude_points: number;
  longitude_points: number;

  latitude_min: number;
  latitude_max: number;

  longitude_min: number;
  longitude_max: number;

  metadata?: ResponseMetadata;
}

export interface LocationRequest {
  latitude: number;
  longitude: number;
}

export interface HazardResponse {
  latitude: number;
  longitude: number;

  forecast_sample: number;
  forecast_day: number;

  grid_latitude: number;
  grid_longitude: number;

  hazard_score: number;

  population_exposure: number | null;
  population_component: number | null;
  population_density: number | null;

  sampling_method: string;
  grid_distance_degrees: number | null;

  metadata?: ResponseMetadata;
}

export interface HazardGridPoint {
  latitude: number;
  longitude: number;
  hazard_score: number;
}

export interface HazardGridResponse {
  forecast_sample: number;
  forecast_day: number;

  center_latitude: number;
  center_longitude: number;

  radius_degrees: number;

  rows: number;
  columns: number;

  points: HazardGridPoint[];

  metadata?: ResponseMetadata;
}

export interface Shelter {
  shelter_id: number;

  latitude: number;
  longitude: number;

  road_node: number | null;
  snap_distance_m: number | null;

  mapped: boolean;
}

export interface SheltersResponse {
  total_shelters: number;
  mapped_shelters: number;
  unmapped_shelters: number;

  shelters: Shelter[];
}

export interface RouteStatistics {
  road_edges: number;
  road_distance_km: number;
  estimated_travel_time_min: number;

  risk_cost: number;

  mean_flood_risk: number;
  maximum_flood_risk: number;

  mean_uncertainty_risk: number;
  maximum_uncertainty_risk: number;

  mean_total_risk: number;

  maximum_bridge_risk: number;
  bridge_edges: number;
}

export interface RouteGeometry {
  nodes: number[];
  edge_ids: number[];

  /*
   * Backend format:
   * [longitude, latitude]
   */
  coordinates: [number, number][];
}

export interface RouteEndpoint {
  latitude: number;
  longitude: number;

  nearest_node: number;
  snap_distance_m: number;
}

export interface RouteRoutingMetadata {
  algorithm: string;
  expanded_nodes: number;
  max_search_distance_km: number;
}

export interface RouteResult {
  start: RouteEndpoint;
  goal: RouteEndpoint;

  routing: RouteRoutingMetadata;

  route: RouteGeometry;

  statistics: RouteStatistics;
}

export interface RouteResponse {
  status: string;
  result: RouteResult;
  metadata?: ResponseMetadata;
}

export interface ShelterRecommendation {
  shelter_id: number;

  latitude: number;
  longitude: number;

  road_node: number;

  snap_distance_m: number;

  straight_distance_km: number;
  road_distance_km: number;
  travel_time_min: number;

  risk_cost: number;

  mean_flood_risk: number;
  maximum_flood_risk: number;

  mean_uncertainty: number;
  maximum_uncertainty: number;

  maximum_bridge_risk: number;
  bridge_edges: number;

  destination_hazard: number;
  destination_population_exposure: number;
  destination_population_component: number;
  destination_population_density: number;

  availability: string;
  capacity: number | null;

  source: string;

  route_risk_score: number;
  travel_time_score: number;
  bridge_score: number;
  accessibility_score: number;

  combined_score: number;
  rank: number;
}

export interface EvacuationSystemInfo {
  name: string;

  forecast_sample: number;
  forecast_day: number;

  candidate_shelters: number;
  reachable_shelters: number;

  total_mapped_shelters: number;

  database_mode: string;

  shelter_capacity_available: boolean;
  live_shelter_availability_available: boolean;
}

export interface EvacuationUserInfo {
  latitude: number;
  longitude: number;

  nearest_road_node: number;
  snap_distance_m: number;
}

export interface RankingWeights {
  route_risk: number;
  destination_hazard: number;
  destination_population_exposure: number;
  travel_time: number;
  bridge_exposure: number;
}

export interface EvacuationResult {
  system: EvacuationSystemInfo;

  user: EvacuationUserInfo;

  ranking_weights: RankingWeights;

  recommended_shelter: ShelterRecommendation;

  route: RouteResult;

  alternatives: ShelterRecommendation[];
}

export interface EvacuationResponse {
  status: string;
  result: EvacuationResult;

  metadata?: ResponseMetadata;
}
