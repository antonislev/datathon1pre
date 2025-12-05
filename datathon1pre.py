import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta

# --- Configuration ---
# Bounding box for a fictional city center (like Thessaloniki) to generate mock data
CENTER_LAT, CENTER_LON = 40.63, 22.94
RADIUS_KM = 3.0
ACCEPTABLE_SERVICE_RADIUS_M = 50.0  # Threshold for walking distance from L/U zone

def generate_mock_data(n_deliveries=5000, n_zones=50, n_shops=1000):
    """
    Generates synthetic data matching the challenge description for testing.
    In a real scenario, this function would be replaced by your data loading/cleaning logic.
    """
    np.random.seed(42)
    print(f"Generating {n_deliveries} mock delivery stops...")
    
    # 1. Mock Last-Mile Delivery Data
    # Simulate clustering around a few "hotspots"
    hotspot_lats = [CENTER_LAT + 0.01, CENTER_LAT - 0.005, CENTER_LAT + 0.02]
    hotspot_lons = [CENTER_LON - 0.02, CENTER_LON + 0.01, CENTER_LON + 0.005]
    
    # Randomly select which hotspot each delivery belongs to
    hotspot_indices = np.random.randint(0, len(hotspot_lats), n_deliveries)
    
    # Generate coordinates centered around hotspots with some normal noise
    delivery_lat = np.array([hotspot_lats[i] + np.random.normal(0, 0.01) for i in hotspot_indices])
    delivery_lon = np.array([hotspot_lons[i] + np.random.normal(0, 0.01) for i in hotspot_indices])

    # Generate realistic timestamps (focusing on 7 AM to 3 PM activity)
    start_time = datetime(2025, 10, 1)
    timestamps = [start_time + timedelta(minutes=int(t)) 
                  for t in np.random.randint(7 * 60, 15 * 60, n_deliveries)]
    
    delivery_data = pd.DataFrame({
        'delivery_id': range(n_deliveries),
        'latitude': delivery_lat,
        'longitude': delivery_lon,
        'timestamp': timestamps,
        'weekday': [ts.strftime('%A') for ts in timestamps],
        'product_type': np.random.choice(['Retail', 'Food/Perishables', 'Electronics', 'Bulk Goods'], n_deliveries, p=[0.4, 0.3, 0.1, 0.2]),
        'cargo_volume_kg': np.random.lognormal(2, 0.8, n_deliveries).round(1) # Lognormal distribution for cargo
    })
    
    # 2. Mock Current Loading/Unloading Zone Locations
    lu_zones = pd.DataFrame({
        'lu_id': range(n_zones),
        'latitude': CENTER_LAT + np.random.uniform(-0.025, 0.025, n_zones),
        'longitude': CENTER_LON + np.random.uniform(-0.025, 0.025, n_zones),
        'capacity_slots': np.random.randint(1, 6, n_zones),
    })

    # 3. Mock OSM Shop Distribution (simplified)
    osm_shops = pd.DataFrame({
        'shop_id': range(n_shops),
        'latitude': CENTER_LAT + np.random.uniform(-0.03, 0.03, n_shops),
        'longitude': CENTER_LON + np.random.uniform(-0.03, 0.03, n_shops),
        'category': np.random.choice(['supermarket', 'bakery', 'clothing', 'office', 'restaurant'], n_shops),
    })
    
    return delivery_data, lu_zones, osm_shops

# --- Phase 1: Quantitative Assessment ---

def analyze_inefficiencies(delivery_data, lu_zones):
    """
    Identifies Unserved Demand and calculates utilization metrics.
    
    Returns:
        D_unserved: DataFrame of delivery stops far from any L/U zone.
        lu_utilization: DataFrame with utilization metrics for existing zones.
        temporal_peaks: Series or DataFrame showing peak demand hours.
    """
    # 1. Calculate Distance to Nearest L/U Zone
    
    # Prepare coordinates for distance calculation
    delivery_coords = delivery_data[['latitude', 'longitude']].values
    lu_coords = lu_zones[['latitude', 'longitude']].values

    # Convert distance from degrees (used in coordinates) to meters
    # A quick-and-dirty conversion for Thessaloniki area (approx 1 degree lat = 111.1km)
    # The cdist returns Euclidean distance in degrees. Convert to meters.
    # Note: Use Haversine distance for more accuracy in a real application.
    DEGREE_TO_METER = 111000 # Approximation for the region
    
    # Calculate pairwise distances (in degrees)
    distances_deg = cdist(delivery_coords, lu_coords, metric='euclidean')
    
    # Find the minimum distance (in meters) for each delivery stop
    min_distances_deg = distances_deg.min(axis=1)
    min_distances_m = min_distances_deg * DEGREE_TO_METER
    
    delivery_data['min_dist_m'] = min_distances_m
    delivery_data['nearest_lu_id'] = lu_zones.iloc[distances_deg.argmin(axis=1)]['lu_id'].values

    # 2. Identify Unserved Demand
    D_unserved = delivery_data[delivery_data['min_dist_m'] > ACCEPTABLE_SERVICE_RADIUS_M]
    print(f"\n[ASSESSMENT] Total Unserved Stops: {len(D_unserved)} out of {len(delivery_data)} ({len(D_unserved)/len(delivery_data):.1%})")
    
    # 3. Existing L/U Zone Utilization (Simplified)
    # Count how many stops fall within the service radius of each existing zone
    served_stops = delivery_data[delivery_data['min_dist_m'] <= ACCEPTABLE_SERVICE_RADIUS_M]
    
    lu_utilization = served_stops.groupby('nearest_lu_id').agg(
        total_deliveries=('delivery_id', 'count'),
        total_cargo_kg=('cargo_volume_kg', 'sum')
    ).reset_index()
    
    lu_utilization = lu_zones.merge(lu_utilization, left_on='lu_id', right_on='nearest_lu_id', how='left').fillna(0)
    
    # Utilization metric: deliveries per slot. A high value suggests over-subscription.
    lu_utilization['deliveries_per_slot'] = lu_utilization['total_deliveries'] / lu_utilization['capacity_slots']
    lu_utilization = lu_utilization.sort_values('deliveries_per_slot', ascending=False)
    
    print(f"[ASSESSMENT] Top 5 Over-subscribed Zones (Deliveries/Slot):\n{lu_utilization[['lu_id', 'deliveries_per_slot']].head(5)}")
    print(f"[ASSESSMENT] Top 5 Underutilized Zones (Deliveries/Slot):\n{lu_utilization[lu_utilization['total_deliveries'] == 0][['lu_id', 'deliveries_per_slot']].head(5)}")

    # 4. Temporal Demand Patterns
    # Calculate peak demand by hour
    delivery_data['hour'] = delivery_data['timestamp'].dt.hour
    temporal_peaks = delivery_data.groupby('hour')['delivery_id'].count().sort_values(ascending=False)
    
    return D_unserved, lu_utilization, temporal_peaks

# --- Phase 2: Optimization and Network Proposal ---

def propose_new_zones(D_unserved):
    """
    Applies DBSCAN clustering to unserved demand to propose new L/U zone locations.
    
    Returns:
        proposed_zones: DataFrame of new zone coordinates, covering deliveries, and avg distance saved.
    """
    if D_unserved.empty:
        print("\n[PROPOSAL] No unserved demand found. Network is already optimal.")
        return pd.DataFrame()

    unserved_coords = D_unserved[['latitude', 'longitude']].values
    
    # DBSCAN parameters:
    # eps (Epsilon) is the maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # We use 75m (0.000675 degrees) slightly larger than the 50m service radius to group stops.
    # min_samples is the number of samples in a neighborhood for a point to be considered as a core point.
    EPS_DEG = 75.0 / 111000 # Convert 75m to degrees (approximation)
    MIN_SAMPLES = 15 # Require at least 15 unserved stops to justify a new zone
    
    print(f"\n[PROPOSAL] Starting DBSCAN with EPS={75.0}m and Min Samples={MIN_SAMPLES}...")
    
    # Fit the DBSCAN model
    db = DBSCAN(eps=EPS_DEG, min_samples=MIN_SAMPLES, algorithm='ball_tree', metric='euclidean').fit(unserved_coords)
    D_unserved['cluster'] = db.labels_
    
    # Extract clusters (ignoring noise labeled -1)
    clustered_deliveries = D_unserved[D_unserved['cluster'] != -1]
    
    if clustered_deliveries.empty:
        print("[PROPOSAL] DBSCAN found no significant clusters (below min_samples threshold).")
        return pd.DataFrame()
        
    # Calculate the centroid for each cluster (Proposed New Zone Location)
    proposed_zones = clustered_deliveries.groupby('cluster').agg(
        proposed_latitude=('latitude', 'mean'),
        proposed_longitude=('longitude', 'mean'),
        total_deliveries_covered=('delivery_id', 'count'),
        avg_dist_saved_m=('min_dist_m', 'mean') # Average distance driver currently walks (potential saving)
    ).reset_index()
    
    # Add a unique ID for the proposed zone
    proposed_zones['proposed_zone_id'] = [f"NEW_{i+1}" for i in range(len(proposed_zones))]
    
    # Implement a priority score (Normalized Volume * Average Inefficiency)
    # The higher the score, the more critical the new zone is.
    
    # Normalization for fair comparison
    max_deliveries = proposed_zones['total_deliveries_covered'].max()
    max_dist_saved = proposed_zones['avg_dist_saved_m'].max()
    
    proposed_zones['norm_deliveries'] = proposed_zones['total_deliveries_covered'] / max_deliveries
    proposed_zones['norm_dist_saved'] = proposed_zones['avg_dist_saved_m'] / max_dist_saved
    
    # Priority Score: prioritize high volume in highly inefficient areas
    proposed_zones['priority_score'] = (proposed_zones['norm_deliveries'] * 0.6) + (proposed_zones['norm_dist_saved'] * 0.4)
    
    proposed_zones = proposed_zones.sort_values('priority_score', ascending=False)
    
    print(f"[PROPOSAL] Identified {len(proposed_zones)} high-priority locations for new L/U zones.")
    return proposed_zones

# --- Main Execution ---

if __name__ == "__main__":
    # --- STEP 1: LOAD AND CLEAN DATA ---
    DELIVERY_DATA, LU_ZONES, OSM_SHOPS = generate_mock_data()
    
    # --- STEP 2: QUANTITATIVE ASSESSMENT ---
    D_UNSERVED, LU_UTILIZATION, TEMPORAL_PEAKS = analyze_inefficiencies(DELIVERY_DATA, LU_ZONES)
    
    print("\n" + "="*80)
    print("           QUANTITATIVE ASSESSMENT SUMMARY (Inefficiencies)")
    print("="*80)
    print(f"** Peak Demand Hours (Overall)**:\n{TEMPORAL_PEAKS.head(3)}")
    print(f"\n** Primary Network Failure (Unserved Demand)**: {len(D_UNSERVED)} Stops requiring attention.")
    
    # Example of Inefficiency Finding: Over-subscribed Zones
    print("\n** Recommendation: Capacity Expansion Candidates (Over-subscribed)**")
    expansion_candidates = LU_UTILIZATION[LU_UTILIZATION['deliveries_per_slot'] > 25].head(5)
    print(expansion_candidates[['lu_id', 'latitude', 'longitude', 'total_deliveries', 'deliveries_per_slot']])
    
    # Example of Inefficiency Finding: Underutilized Zones
    print("\n** Recommendation: Removal/Repurposing Candidates (Underutilized)**")
    removal_candidates = LU_UTILIZATION[LU_UTILIZATION['total_deliveries'] < 5].head(5)
    print(removal_candidates[['lu_id', 'latitude', 'longitude', 'total_deliveries', 'capacity_slots']])
    
    
    # --- STEP 3: OPTIMIZED L/U NETWORK PROPOSAL ---
    PROPOSED_ZONES = propose_new_zones(D_UNSERVED)
    
    if not PROPOSED_ZONES.empty:
        print("\n" + "="*80)
        print("              OPTIMIZED L/U NETWORK PROPOSAL (Prioritized)")
        print("="*80)
        
        # Display the top 5 proposed locations
        print("Top 5 Proposed New L/U Zone Locations (Ranked by Priority Score):")
        
        final_recommendations = PROPOSED_ZONES[['proposed_zone_id', 'priority_score', 
                                              'proposed_latitude', 'proposed_longitude', 
                                              'total_deliveries_covered', 'avg_dist_saved_m']].head(5)
                                              
        # NOTE: In a real app, you would add logic here to determine Zone Type (e.g., 'Time-Restricted')
        # by checking the product mix (OSM_SHOPS or DELIVERY_DATA product_type) in the area of the proposed zone.
        
        print(final_recommendations.to_string(index=False, float_format="%.5f"))
        print("\n*avg_dist_saved_m is the average distance (in meters) drivers currently walk.")
        
        print("\n\n*** Next Steps for Implementation ***")
        print("1. Replace the `generate_mock_data` function with your actual CSV/JSON loading logic.")
        print("2. Integrate a proper geospatial library (e.g., `geopy` or `haversine`) to replace the `DEGREE_TO_METER` constant for accurate distance calculation.")
        print("3. Use the coordinates from the `final_recommendations` table to plot your optimized map.")