#from src. ingest.chunk_build_uploads import load_csv_chunked
import pandas as pd
import pathlib
import joblib
import logging
from typing import Dict, Any
def load_csv_chunked(path, parse_dates=("date",), chunksize=2000_000):
    # logger.info(f"Loading {path} in chunks of {chunksize} rows...")
    return pd.read_csv(path, parse_dates=list(parse_dates), low_memory=False, chunksize=chunksize)

def analyze_logon_structure(logon_path):
    """Analyze logon.csv structure to understand available features."""
    
    print(" ANALYZING LOGON DATA STRUCTURE...")
    
    # Sample the logon data to understand structure
    logon_sample = pd.read_csv(logon_path, nrows=1000)
    
    logon_info = {
        "columns": list(logon_sample.columns),
        "dtypes": logon_sample.dtypes.to_dict(),
        "sample_data": logon_sample.head().to_dict(),
        "unique_activities": logon_sample['activity'].unique().tolist() if 'activity' in logon_sample.columns else [],
        "date_range": [str(logon_sample['date'].min()), str(logon_sample['date'].max())] if 'date' in logon_sample.columns else None
    }
    
    print(f" Logon columns: {logon_info['columns']}")
    print(f" Logon activities: {logon_info['unique_activities']}")
    
    return logon_info

def process_logon_chunks_leak_free(logon_path, temporal_cutoff=None):
    """
    Process logon data with comprehensive behavioral analysis.
    
    Expected CERT logon.csv columns:
    - id, date, user, pc, activity
    
    Activities typically include:
    - Logon, Logoff, Connect, Disconnect
    """
    
    print(" PROCESSING LOGON DATA WITH BEHAVIORAL ANALYSIS...")
    
    if temporal_cutoff is None:
        return process_logon_chunks(logon_path)
    
    # Load all raw logon data
    raw_chunks = []
    chunk_idx = 0
    
    for chunk in load_csv_chunked(logon_path):
        chunk_idx += 1
        print(f"Processing LOGON chunk {chunk_idx} with {len(chunk)} rows")
        
        # Basic data validation
        required_cols = ['date', 'user', 'pc', 'activity']
        missing_cols = [col for col in required_cols if col not in chunk.columns]
        if missing_cols:
            print(f"Missing required columns in logon data: {missing_cols}")
            continue
            
        raw_chunks.append(chunk)
    
    # Combine all raw logon data
    raw_logon = pd.concat(raw_chunks, ignore_index=True)
    print(f" Raw LOGON data: {len(raw_logon)} events")
    
    # Temporal split
    cutoff_ts = pd.Timestamp(temporal_cutoff)
    train_raw = raw_logon[raw_logon.date < cutoff_ts].copy()
    test_raw = raw_logon[raw_logon.date >= cutoff_ts].copy()
    
    print(f" LOGON temporal split: Train={len(train_raw)}, Test={len(test_raw)}")
    
    # Process separately to maintain temporal isolation
    train_processed = process_logon_features_isolated(train_raw, is_training=True)
    test_processed = process_logon_features_isolated(test_raw, is_training=False, train_data=train_raw)
    
    return pd.concat([train_processed, test_processed], ignore_index=True)

def process_logon_features_isolated(data, is_training=True, train_data=None):
    """
    Extract comprehensive behavioral features from logon data.
    
    Key Features:
    1. Session duration analysis
    2. Login frequency patterns  
    3. Off-hours access detection
    4. Device usage patterns
    5. Location anomalies
    6. Concurrent session detection
    """
    
    if len(data) == 0:
        return pd.DataFrame()
    
    print(f" Processing {'training' if is_training else 'test'} LOGON features...")
    
    # Sort data chronologically for accurate session analysis
    data = data.sort_values(['user', 'pc', 'date']).reset_index(drop=True)
    
    # ========================================================================
    # 1. SESSION DURATION ANALYSIS
    # ========================================================================
    
    def calculate_session_durations(df):
        """Calculate session durations from logon/logoff pairs."""
        
        sessions = []
        
        for user in df['user'].unique():
            user_data = df[df['user'] == user].sort_values('date')
            
            for pc in user_data['pc'].unique():
                pc_data = user_data[user_data['pc'] == pc].sort_values('date')
                
                logon_time = None
                
                for _, row in pc_data.iterrows():
                    if row['activity'] in ['Logon', 'Connect']:
                        logon_time = row['date']
                    elif row['activity'] in ['Logoff', 'Disconnect'] and logon_time is not None:
                        duration_minutes = (row['date'] - logon_time).total_seconds() / 60
                        
                        sessions.append({
                            'user': user,
                            'pc': pc,
                            'logon_time': logon_time,
                            'logoff_time': row['date'],
                            'session_duration_minutes': duration_minutes,
                            'logon_hour': logon_time.hour,
                            'logon_day_of_week': logon_time.dayofweek,
                            'is_weekend_session': logon_time.dayofweek >= 5,
                            'is_after_hours_session': logon_time.hour < 7 or logon_time.hour >= 19
                        })
                        
                        logon_time = None  # Reset for next session
        
        return pd.DataFrame(sessions)
    
    # Extract session information
    sessions_df = calculate_session_durations(data)
    
    if len(sessions_df) == 0:
        print(" No complete sessions found in logon data")
        return pd.DataFrame()
    
    # ========================================================================
    # 2. USER BEHAVIORAL PROFILING
    # ========================================================================
    
    if is_training:
        print(" Computing training user logon behavioral profiles...")
        
        # Calculate user logon patterns
        user_logon_profiles = sessions_df.groupby('user').agg({
            'session_duration_minutes': ['mean', 'std', 'min', 'max', 'count'],
            'logon_hour': ['mean', 'std'],
            'pc': 'nunique',
            'is_weekend_session': 'mean',
            'is_after_hours_session': 'mean'
        }).round(3)
        
        # Flatten column names
        user_logon_profiles.columns = [
            'avg_session_duration', 'session_duration_std', 'min_session_duration', 
            'max_session_duration', 'total_sessions',
            'avg_logon_hour', 'logon_hour_std',
            'unique_devices_used',
            'weekend_logon_ratio', 'after_hours_logon_ratio'
        ]
        
        # Save logon profiles
        profiles_dir = pathlib.Path("training_profiles")
        profiles_dir.mkdir(exist_ok=True, parents=True)
        
        logon_profile_file = profiles_dir / 'user_logon_profiles.pkl'
        joblib.dump(user_logon_profiles, logon_profile_file)
        print(f" Saved user logon behavioral profiles to {logon_profile_file}")
        
        # Daily activity patterns
        daily_activity = sessions_df.groupby(['user', sessions_df['logon_time'].dt.date]).agg({
            'session_duration_minutes': 'sum',
            'pc': 'nunique'
        }).groupby('user').agg({
            'session_duration_minutes': ['mean', 'std'],
            'pc': 'mean'
        }).round(3)
        
        daily_activity.columns = ['avg_daily_duration', 'daily_duration_std', 'avg_daily_devices']
        
        # Combine profiles
        combined_profiles = user_logon_profiles.join(daily_activity, how='outer')
        
        # Save combined profiles
        combined_profile_file = profiles_dir / 'user_logon_combined_profiles.pkl'
        joblib.dump(combined_profiles, combined_profile_file)
        
    else:
        print(" Loading training user logon behavioral profiles for test data...")
        
        profiles_dir = pathlib.Path("training_profiles")
        
        try:
            combined_profiles = joblib.load(profiles_dir / 'user_logon_combined_profiles.pkl')
            print(f" Loaded logon profiles for {len(combined_profiles)} users")
            
        except FileNotFoundError:
            if train_data is not None:
                print(" Computing fallback logon profiles from provided train_data...")
                train_sessions = calculate_session_durations(train_data)
                
                if len(train_sessions) > 0:
                    combined_profiles = train_sessions.groupby('user').agg({
                        'session_duration_minutes': ['mean', 'std', 'count'],
                        'logon_hour': ['mean', 'std'],
                        'pc': 'nunique',
                        'is_weekend_session': 'mean',
                        'is_after_hours_session': 'mean'
                    }).round(3)
                    
                    combined_profiles.columns = [
                        'avg_session_duration', 'session_duration_std', 'total_sessions',
                        'avg_logon_hour', 'logon_hour_std', 'unique_devices_used',
                        'weekend_logon_ratio', 'after_hours_logon_ratio'
                    ]
                else:
                    combined_profiles = pd.DataFrame()
            else:
                combined_profiles = pd.DataFrame()
    
    # ========================================================================
    # 3. ANOMALY DETECTION FEATURES
    # ========================================================================
    
    def calculate_logon_anomalies(sessions, profiles):
        """Calculate logon behavioral anomalies."""
        
        if len(profiles) == 0:
            sessions['session_duration_anomaly'] = 0
            sessions['logon_hour_anomaly'] = 0
            sessions['device_usage_anomaly'] = 0
            return sessions
        
        # Session duration anomalies
        for user in sessions['user'].unique():
            user_sessions = sessions[sessions['user'] == user]
            
            if user in profiles.index:
                profile = profiles.loc[user]
                
                # Duration anomaly (z-score)
                avg_duration = profile.get('avg_session_duration', user_sessions['session_duration_minutes'].mean())
                std_duration = profile.get('session_duration_std', user_sessions['session_duration_minutes'].std())
                
                if std_duration > 0:
                    duration_zscore = abs(user_sessions['session_duration_minutes'] - avg_duration) / std_duration
                else:
                    duration_zscore = 0
                
                sessions.loc[sessions['user'] == user, 'session_duration_anomaly'] = duration_zscore
                
                # Hour anomaly
                avg_hour = profile.get('avg_logon_hour', user_sessions['logon_hour'].mean())
                hour_std = profile.get('logon_hour_std', user_sessions['logon_hour'].std())
                
                if hour_std > 0:
                    hour_zscore = abs(user_sessions['logon_hour'] - avg_hour) / hour_std
                else:
                    hour_zscore = 0
                
                sessions.loc[sessions['user'] == user, 'logon_hour_anomaly'] = hour_zscore
                
                # Device usage anomaly (new device detection)
                historical_devices = profile.get('unique_devices_used', 1)
                current_device_novelty = 1.0 / (historical_devices + 1)  # Higher for users with fewer historical devices
                
                sessions.loc[sessions['user'] == user, 'device_usage_anomaly'] = current_device_novelty
            else:
                # New user - all activity is somewhat anomalous
                sessions.loc[sessions['user'] == user, 'session_duration_anomaly'] = 0.5
                sessions.loc[sessions['user'] == user, 'logon_hour_anomaly'] = 0.5
                sessions.loc[sessions['user'] == user, 'device_usage_anomaly'] = 1.0
        
        return sessions
    
    # Apply anomaly detection
    sessions_with_anomalies = calculate_logon_anomalies(sessions_df, combined_profiles if len(sessions_df) > 0 else pd.DataFrame())
    
    # ========================================================================
    # 4. CONCURRENT SESSION DETECTION
    # ========================================================================
    
    def detect_concurrent_sessions(sessions):
        """Detect potentially suspicious concurrent sessions."""
        
        sessions['concurrent_sessions'] = 0
        sessions['multiple_device_sessions'] = False
        sessions['geographic_anomaly'] = False  # Placeholder for future geo analysis
        
        for user in sessions['user'].unique():
            user_sessions = sessions[sessions['user'] == user].sort_values('logon_time')
            
            for idx, session in user_sessions.iterrows():
                # Count overlapping sessions
                overlapping = user_sessions[
                    (user_sessions['logon_time'] <= session['logoff_time']) &
                    (user_sessions['logoff_time'] >= session['logon_time']) &
                    (user_sessions.index != idx)
                ]
                
                sessions.loc[idx, 'concurrent_sessions'] = len(overlapping)
                
                # Multiple device sessions (same time, different PCs)
                multi_device = overlapping[overlapping['pc'] != session['pc']]
                sessions.loc[idx, 'multiple_device_sessions'] = len(multi_device) > 0
        
        return sessions
    
    sessions_with_concurrent = detect_concurrent_sessions(sessions_with_anomalies)
    
    # ========================================================================
    # 5. AGGREGATE TO USER-DATE LEVEL FOR UPLOAD CORRELATION
    # ========================================================================
    
    # Aggregate session data to daily user summaries for correlation with uploads
    daily_logon_summary = sessions_with_concurrent.groupby([
        'user', 
        sessions_with_concurrent['logon_time'].dt.date
    ]).agg({
        'session_duration_minutes': ['sum', 'count', 'mean', 'max'],
        'session_duration_anomaly': 'max',
        'logon_hour_anomaly': 'max',
        'device_usage_anomaly': 'max',
        'concurrent_sessions': 'max',
        'multiple_device_sessions': 'any',
        'is_after_hours_session': 'any',
        'is_weekend_session': 'any',
        'pc': 'nunique'
    }).round(3)
    
    # Flatten column names
    daily_logon_summary.columns = [
        'total_session_time', 'logon_count', 'avg_session_duration', 'max_session_duration',
        'max_session_duration_anomaly', 'max_logon_hour_anomaly', 'max_device_usage_anomaly',
        'max_concurrent_sessions', 'had_multiple_device_sessions', 
        'had_after_hours_logon', 'had_weekend_logon', 'unique_devices_used_today'
    ]
    
    # Reset index to get user and date as columns
    daily_logon_summary = daily_logon_summary.reset_index()
    daily_logon_summary.columns = ['user', 'date'] + list(daily_logon_summary.columns[2:])
    
    # Convert date back to datetime for merging
    daily_logon_summary['date'] = pd.to_datetime(daily_logon_summary['date'])
    
    print(f" {'Training' if is_training else 'Test'} LOGON processing complete:")
    print(f"   - Sessions analyzed: {len(sessions_with_concurrent)}")
    print(f"   - Daily summaries: {len(daily_logon_summary)}")
    print(f"   - Users with logon data: {daily_logon_summary['user'].nunique()}")
    
    return daily_logon_summary

# ========================================================================
# INTEGRATION WITH UPLOAD DATA
# ========================================================================

def merge_logon_with_uploads(uploads_df, logon_summary_df):
    """
    Merge logon behavioral data with upload events.
    
    This provides critical context for each upload:
    - Was the user logged in when the upload occurred?
    - Were there any logon anomalies that day?
    - Did the user have concurrent sessions?
    """
    
    print(" MERGING LOGON BEHAVIORAL DATA WITH UPLOADS...")
    
    # Ensure date columns are datetime
    uploads_df['date'] = pd.to_datetime(uploads_df['date'])
    logon_summary_df['date'] = pd.to_datetime(logon_summary_df['date'])
    
    # Create date-only columns for merging
    uploads_df['upload_date'] = uploads_df['date'].dt.date
    logon_summary_df['logon_date'] = logon_summary_df['date'].dt.date
    
    # Merge on user and date
    if 'user' in uploads_df.columns and 'user' in logon_summary_df.columns:
        merged_df = uploads_df.merge(
            logon_summary_df.drop(columns=['date']),  # Drop original date, keep logon_date
            left_on=['user', 'upload_date'],
            right_on=['user', 'logon_date'],
            how='left'
        )
    else:
        # Only merge on date (population-level logon features)
        merged_df = uploads_df.merge(
            logon_summary_df.groupby('logon_date').mean(numeric_only=True).reset_index(),
            left_on='upload_date',
            right_on='logon_date',
            how='left'
        )
    
    # Clean up temporary columns
    merged_df = merged_df.drop(columns=['upload_date', 'logon_date'])
    
    # Fill missing logon data (days with no logons)
    logon_features = [
        'total_session_time', 'logon_count', 'avg_session_duration', 'max_session_duration',
        'max_session_duration_anomaly', 'max_logon_hour_anomaly', 'max_device_usage_anomaly',
        'max_concurrent_sessions', 'unique_devices_used_today'
    ]
    
    for feature in logon_features:
        if feature in merged_df.columns:
            merged_df[feature] = merged_df[feature].fillna(0)
    
    # Boolean features
    boolean_logon_features = [
        'had_multiple_device_sessions', 'had_after_hours_logon', 'had_weekend_logon'
    ]
    
    for feature in boolean_logon_features:
        if feature in merged_df.columns:
            merged_df[feature] = merged_df[feature].fillna(False)
    
    # Create composite risk indicators
    merged_df['logon_risk_score'] = (
        merged_df.get('max_session_duration_anomaly', 0) * 0.3 +
        merged_df.get('max_logon_hour_anomaly', 0) * 0.3 +
        merged_df.get('max_device_usage_anomaly', 0) * 0.2 +
        (merged_df.get('max_concurrent_sessions', 0) > 1).astype(float) * 0.2
    )
    
    # No logon activity indicator (suspicious for uploads)
    merged_df['no_logon_activity'] = merged_df.get('logon_count', 0) == 0
    
    print(f" Logon-Upload merge complete:")
    print(f"   - Total uploads: {len(merged_df)}")
    print(f"   - Uploads with logon data: {(merged_df['logon_count'] > 0).sum()}")
    print(f"   - Uploads with no logon activity: {merged_df['no_logon_activity'].sum()}")
    
    return merged_df

