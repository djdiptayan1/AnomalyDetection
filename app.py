import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import io
from sklearn.preprocessing import MinMaxScaler
import tempfile

st.set_page_config(page_title="Anomaly Detection", page_icon="🔍", layout="wide")


@st.cache_resource
def load_local_model():
    """Load model and scaler from local files if available"""
    try:
        model = keras.models.load_model(
            "models/autoencoder.h5",
            custom_objects={"mse": keras.losses.MeanSquaredError()},
        )
        scaler = np.load("models/scaler.npy", allow_pickle=True).item()
        return model, scaler
    except Exception as e:
        return None, None


def load_model_and_scaler(model_file, scaler_file):
    """Load the trained model and scaler from uploaded files"""
    try:
        model = keras.models.load_model(
            model_file, custom_objects={"mse": keras.losses.MeanSquaredError()}
        )
        scaler = np.load(scaler_file, allow_pickle=True).item()
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None


def process_csv_in_chunks(file, model, scaler, threshold_factor=3.0, batch_size=10000):
    """Process large CSV files in chunks"""
    results = []
    threshold = None

    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    # Get total file size for progress calculation
    file_size = os.path.getsize(tmp_file_path)

    # Use pandas chunk iterator to process large files
    progress_bar = st.progress(0)
    chunk_iterator = pd.read_csv(tmp_file_path, chunksize=batch_size)

    # Track bytes processed for progress
    bytes_processed = 0

    for i, chunk in enumerate(chunk_iterator):
        # Update progress
        if file_size > 0:
            # Estimate progress based on file position
            bytes_processed += chunk.memory_usage(deep=True).sum()
            progress = min(bytes_processed / file_size, 1.0)
            progress_bar.progress(progress)

        # Process chunk
        chunk_copy = chunk.copy()

        # Drop faultNumber if present
        if "faultNumber" in chunk_copy.columns:
            chunk_copy = chunk_copy.drop(columns=["faultNumber"])

        # Try to match features with scaler if possible
        try:
            if hasattr(scaler, "feature_names_in_"):
                expected_features = scaler.feature_names_in_
                chunk_numeric = chunk_copy[expected_features]
            else:
                # Otherwise use all numeric columns
                numerical_cols = chunk_copy.select_dtypes(include=["number"]).columns
                chunk_numeric = chunk_copy[numerical_cols]
        except Exception as e:
            st.error(f"Error with feature selection: {e}")
            st.error(
                "The uploaded data columns don't match what the model was trained on"
            )
            os.unlink(tmp_file_path)
            return None, None

        # Scale data
        try:
            scaled_chunk = scaler.transform(chunk_numeric)
        except Exception as e:
            st.error(f"Error scaling data: {e}")
            st.error(
                "Make sure the CSV data columns match what the model was trained on"
            )
            os.unlink(tmp_file_path)
            return None, None

        # Predict reconstruction error
        reconstructed = model.predict(scaled_chunk, verbose=0)
        mse = np.mean(np.power(scaled_chunk - reconstructed, 2), axis=1)

        # Set threshold dynamically based on first batch
        if threshold is None:
            threshold = np.mean(mse) + threshold_factor * np.std(mse)

        # Mark anomalies
        chunk_copy["ReconstructionError"] = mse
        chunk_copy["Anomaly"] = mse > threshold
        results.append(chunk_copy)

    # Complete progress
    progress_bar.progress(1.0)

    # Clean up temporary file
    os.unlink(tmp_file_path)

    # Combine results
    try:
        final_df = pd.concat(results, ignore_index=True)
        return final_df, threshold
    except Exception as e:
        st.error(f"Error combining results: {e}")
        return None, None


def plot_reconstruction_histogram(df, threshold, sample_size=None):
    """Plot histogram with numeric interpretation"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate histogram
    n, bins, patches = ax.hist(
        df["ReconstructionError"],
        bins=50,
        alpha=0.75,
        color="blue",
        label="Reconstruction Error",
    )

    # Add threshold line
    ax.axvline(
        threshold, color="red", linestyle="dashed", linewidth=2, label="Threshold"
    )

    # Add labels and title
    ax.set_xlabel("MSE (Reconstruction Error)")
    ax.set_ylabel("Frequency")
    ax.set_title("Anomaly Detection using Autoencoder")
    ax.legend()

    # Return the figure and numeric summary
    total_count = len(df)
    below_threshold = sum(df["ReconstructionError"] <= threshold)
    above_threshold = sum(df["ReconstructionError"] > threshold)

    # Calculate percentiles for interpretation
    percentiles = {
        "25th": np.percentile(df["ReconstructionError"], 25),
        "50th (median)": np.percentile(df["ReconstructionError"], 50),
        "75th": np.percentile(df["ReconstructionError"], 75),
        "90th": np.percentile(df["ReconstructionError"], 90),
        "95th": np.percentile(df["ReconstructionError"], 95),
        "99th": np.percentile(df["ReconstructionError"], 99),
    }

    stats = {
        "min": df["ReconstructionError"].min(),
        "max": df["ReconstructionError"].max(),
        "mean": df["ReconstructionError"].mean(),
        "std": df["ReconstructionError"].std(),
        "threshold": threshold,
        "normal_points": below_threshold,
        "anomalies": above_threshold,
        "normal_percent": (below_threshold / total_count) * 100,
        "anomaly_percent": (above_threshold / total_count) * 100,
        "percentiles": percentiles,
    }

    return fig, stats


def interpret_time_series(df, threshold):
    """Analyze patterns in time series anomalies"""
    # Get basic stats
    total_points = len(df)
    anomaly_points = sum(df["Anomaly"])
    normal_points = total_points - anomaly_points

    # Look for consecutive anomalies (potential incident periods)
    df = df.sort_values(by="Anomaly", ascending=False).reset_index(drop=True)
    df["consecutive_group"] = (df["Anomaly"] != df["Anomaly"].shift()).cumsum()
    anomaly_groups = df[df["Anomaly"]].groupby("consecutive_group").size()

    # Calculate stats on groups
    if len(anomaly_groups) > 0:
        max_consecutive = anomaly_groups.max()
        avg_consecutive = anomaly_groups.mean()
        total_groups = len(anomaly_groups)
    else:
        max_consecutive = 0
        avg_consecutive = 0
        total_groups = 0

    stats = {
        "total_points": total_points,
        "normal_points": normal_points,
        "anomaly_points": anomaly_points,
        "anomaly_percent": (
            (anomaly_points / total_points) * 100 if total_points > 0 else 0
        ),
        "distinct_anomaly_periods": total_groups,
        "longest_anomaly_sequence": max_consecutive,
        "avg_anomaly_sequence": avg_consecutive,
    }

    return stats


def interpret_feature_importance(normal_data, anomaly_data):
    """Analyze and interpret feature importance"""
    if normal_data.empty or anomaly_data.empty:
        return None

    # Calculate mean for normal and anomaly data
    normal_mean = normal_data.mean()
    anomaly_mean = anomaly_data.mean()

    # Calculate absolute difference
    feature_importance = abs(normal_mean - anomaly_mean)

    # Handle zero max value
    if feature_importance.max() > 0:
        # Normalize to 0-1 scale
        normalized_importance = feature_importance / feature_importance.max()

        # Create interpretation data
        importance_df = pd.DataFrame(
            {
                "feature": feature_importance.index,
                "normal_mean": normal_mean,
                "anomaly_mean": anomaly_mean,
                "abs_difference": feature_importance,
                "relative_importance": normalized_importance,
            }
        ).sort_values("relative_importance", ascending=False)

        return importance_df

    return None


def main():
    st.title("🔍 Anomaly Detection System")
    st.write(
        "Upload your CSV data file and model files to detect anomalies (no file size restrictions)"
    )

    # First check if local model is available
    local_model, local_scaler = load_local_model()
    if local_model is not None and local_scaler is not None:
        st.success("✅ Local model and scaler loaded successfully")
        model_source = "local"
    else:
        model_source = "upload"

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(
        ["Upload & Detect", "Visualization & Interpretation", "About"]
    )

    with tab1:
        st.header("Upload Files")

        # Data upload
        data_file = st.file_uploader("Upload CSV data file", type=["csv"])

        # Model files upload if local model not available
        if model_source == "upload":
            st.subheader("Model Files")
            model_file = st.file_uploader("Upload model file (.h5)", type=["h5"])
            scaler_file = st.file_uploader("Upload scaler file (.npy)", type=["npy"])

        # Detection parameters
        st.subheader("Detection Parameters")
        threshold_factor = st.slider(
            "Threshold Factor (σ)",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Number of standard deviations above mean for anomaly threshold",
        )
        batch_size = st.slider(
            "Batch Size",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="Number of rows to process at once (lower for less memory usage)",
        )

        # Process files when ready
        process_ready = False
        if data_file:
            if model_source == "local":
                process_ready = True
                model, scaler = local_model, local_scaler
            elif model_file and scaler_file:
                process_ready = True
                # Save model and scaler to temporary files
                with tempfile.NamedTemporaryFile(
                    suffix=".h5", delete=False
                ) as tmp_model_file:
                    tmp_model_file.write(model_file.getvalue())
                    tmp_model_path = tmp_model_file.name

                with tempfile.NamedTemporaryFile(
                    suffix=".npy", delete=False
                ) as tmp_scaler_file:
                    tmp_scaler_file.write(scaler_file.getvalue())
                    tmp_scaler_path = tmp_scaler_file.name

                # Load model and scaler
                model, scaler = load_model_and_scaler(tmp_model_path, tmp_scaler_path)

                # Clean up if loading fails
                if model is None or scaler is None:
                    os.unlink(tmp_model_path)
                    os.unlink(tmp_scaler_path)
                    process_ready = False

        if process_ready and st.button("Detect Anomalies"):
            with st.spinner("Processing data..."):
                # Process CSV in chunks
                st.info(
                    "Processing CSV file in chunks. This may take some time for very large files..."
                )
                result_df, threshold = process_csv_in_chunks(
                    data_file, model, scaler, threshold_factor, batch_size
                )

                # Clean up temporary files if they were created
                if model_source == "upload":
                    os.unlink(tmp_model_path)
                    os.unlink(tmp_scaler_path)

                if result_df is not None:
                    # Store original data for visualization
                    original_cols = [
                        col
                        for col in result_df.columns
                        if col not in ["ReconstructionError", "Anomaly"]
                    ]
                    st.session_state.original_sample = result_df[original_cols].copy()

                    # Save full results to session state
                    st.session_state.df = result_df
                    st.session_state.threshold = threshold

                    # Calculate anomaly stats
                    anomaly_count = result_df["Anomaly"].sum()
                    total_count = len(result_df)
                    anomaly_percent = (anomaly_count / total_count) * 100

                    # Show success and summary
                    st.success("Processing complete!")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Records", f"{total_count:,}")
                    col2.metric("Anomalies Detected", f"{anomaly_count:,}")
                    col3.metric("Anomaly Rate", f"{anomaly_percent:.2f}%")

                    # Preview results
                    st.subheader("Data Preview (First 100 rows)")
                    st.dataframe(result_df.head(100))

                    # Preview anomalies
                    if anomaly_count > 0:
                        st.subheader("Anomaly Preview (First 100 anomalies)")
                        st.dataframe(result_df[result_df["Anomaly"]].head(100))

                    # Download options
                    st.subheader("Download Options")
                    col1, col2 = st.columns(2)

                    with col1:
                        csv_all = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download All Results",
                            data=csv_all,
                            file_name="anomaly_detection_full_results.csv",
                            mime="text/csv",
                        )

                    with col2:
                        if anomaly_count > 0:
                            csv_anomalies = (
                                result_df[result_df["Anomaly"]]
                                .to_csv(index=False)
                                .encode("utf-8")
                            )
                            st.download_button(
                                label="Download Anomalies Only",
                                data=csv_anomalies,
                                file_name="anomaly_detection_anomalies_only.csv",
                                mime="text/csv",
                            )
                else:
                    st.error(
                        "Failed to process data. Please check your files and try again."
                    )

    with tab2:
        st.header("Visualizations with Numerical Interpretation")

        if "df" in st.session_state:
            # Create visualization based on a sample if dataset is large
            sample_size = min(
                len(st.session_state.df), 50000
            )  # Limit to 50k rows for visualization
            sample_df = (
                st.session_state.df.sample(sample_size, random_state=42)
                if len(st.session_state.df) > sample_size
                else st.session_state.df
            )

            # Plot reconstruction error histogram with interpretation
            st.subheader("1. Reconstruction Error Distribution")
            st.caption(
                f"Based on a sample of {sample_size:,} records"
                if len(st.session_state.df) > sample_size
                else "Based on all records"
            )

            fig, hist_stats = plot_reconstruction_histogram(
                sample_df, st.session_state.threshold, sample_size
            )
            st.pyplot(fig)

            # Display numerical interpretation
            st.subheader("Numerical Interpretation of Error Distribution")
            col1, col2, col3 = st.columns(3)

            col1.metric("Threshold Value", f"{hist_stats['threshold']:.4f}")
            col2.metric("Mean Error", f"{hist_stats['mean']:.4f}")
            col3.metric("Standard Deviation", f"{hist_stats['std']:.4f}")

            st.markdown("#### Error Distribution Statistics")
            st.markdown(
                f"- **Range**: {hist_stats['min']:.4f} to {hist_stats['max']:.4f}"
            )
            st.markdown(
                f"- **Normal points**: {hist_stats['normal_points']:,} ({hist_stats['normal_percent']:.2f}%)"
            )
            st.markdown(
                f"- **Anomalies**: {hist_stats['anomalies']:,} ({hist_stats['anomaly_percent']:.2f}%)"
            )

            st.markdown("#### Percentiles")
            percentile_cols = st.columns(3)
            percentile_cols[0].markdown(
                f"- **25th**: {hist_stats['percentiles']['25th']:.4f}"
            )
            percentile_cols[0].markdown(
                f"- **50th (median)**: {hist_stats['percentiles']['50th (median)']:.4f}"
            )
            percentile_cols[1].markdown(
                f"- **75th**: {hist_stats['percentiles']['75th']:.4f}"
            )
            percentile_cols[1].markdown(
                f"- **90th**: {hist_stats['percentiles']['90th']:.4f}"
            )
            percentile_cols[2].markdown(
                f"- **95th**: {hist_stats['percentiles']['95th']:.4f}"
            )
            percentile_cols[2].markdown(
                f"- **99th**: {hist_stats['percentiles']['99th']:.4f}"
            )

            # Time series visualization if time column exists
            time_cols = [
                col
                for col in st.session_state.df.columns
                if any(
                    time_indicator in col.lower()
                    for time_indicator in ["time", "date", "timestamp", "datetime"]
                )
            ]

            if time_cols:
                st.subheader("2. Anomalies Over Time")
                st.caption(
                    f"Based on a sample of {sample_size:,} records"
                    if len(st.session_state.df) > sample_size
                    else "Based on all records"
                )

                time_col = time_cols[0]

                # Create figure
                fig, ax = plt.subplots(figsize=(12, 6))

                # Plot normal points
                normal_points = sample_df[~sample_df["Anomaly"]]
                ax.scatter(
                    normal_points[time_col],
                    normal_points["ReconstructionError"],
                    color="blue",
                    alpha=0.3,
                    s=10,
                    label="Normal",
                )

                # Plot anomalies
                anomaly_points = sample_df[sample_df["Anomaly"]]
                if not anomaly_points.empty:
                    ax.scatter(
                        anomaly_points[time_col],
                        anomaly_points["ReconstructionError"],
                        color="red",
                        alpha=0.6,
                        s=20,
                        label="Anomaly",
                    )

                ax.axhline(
                    st.session_state.threshold,
                    color="red",
                    linestyle="dashed",
                    linewidth=2,
                    label="Threshold",
                )
                ax.set_xlabel(time_col)
                ax.set_ylabel("Reconstruction Error")
                ax.set_title("Anomalies Over Time")
                ax.legend()
                st.pyplot(fig)

                # Time series interpretation
                st.subheader("Numerical Interpretation of Time Series")
                time_stats = interpret_time_series(
                    sample_df, st.session_state.threshold
                )

                col1, col2 = st.columns(2)
                col1.metric("Total Points", f"{time_stats['total_points']:,}")
                col2.metric("Anomaly Rate", f"{time_stats['anomaly_percent']:.2f}%")

                st.markdown("#### Anomaly Pattern Analysis")
                st.markdown(
                    f"- **Distinct anomaly periods**: {time_stats['distinct_anomaly_periods']}"
                )
                st.markdown(
                    f"- **Longest sequence of anomalies**: {time_stats['longest_anomaly_sequence']:.1f} points"
                )
                st.markdown(
                    f"- **Average anomaly sequence length**: {time_stats['avg_anomaly_sequence']:.1f} points"
                )

            # Feature importance visualization and interpretation
            st.subheader("3. Feature Contribution to Anomalies")

            if "original_sample" in st.session_state:
                # Get numerical columns
                numerical_cols = st.session_state.original_sample.select_dtypes(
                    include=["number"]
                ).columns

                # Use our sample for analysis
                normal_data = sample_df[~sample_df["Anomaly"]][numerical_cols]
                anomaly_data = sample_df[sample_df["Anomaly"]][numerical_cols]

                if not anomaly_data.empty and not normal_data.empty:
                    # Get feature importance analysis
                    importance_df = interpret_feature_importance(
                        normal_data, anomaly_data
                    )

                    if importance_df is not None and not importance_df.empty:
                        # Plot top features
                        top_n = min(15, len(importance_df))
                        top_features = importance_df.head(top_n)

                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.barh(
                            top_features["feature"],
                            top_features["relative_importance"],
                            color="teal",
                        )
                        ax.set_xlabel("Relative Importance")
                        ax.set_title(f"Top {top_n} Features Contributing to Anomalies")
                        st.pyplot(fig)

                        # Numerical interpretation
                        st.subheader("Feature Importance Analysis")
                        st.markdown(
                            f"#### Top {min(5, len(importance_df))} Contributing Features"
                        )

                        for i, row in importance_df.head(5).iterrows():
                            st.markdown(
                                f"**{row['feature']}** (Importance: {row['relative_importance']:.2f})"
                            )
                            st.markdown(f"- Normal mean: {row['normal_mean']:.4f}")
                            st.markdown(f"- Anomaly mean: {row['anomaly_mean']:.4f}")
                            st.markdown(f"- Difference: {row['abs_difference']:.4f}")

                        # Show full feature table
                        st.subheader("Complete Feature Importance Table")
                        st.dataframe(importance_df)
                    else:
                        st.info(
                            "Not enough data to calculate meaningful feature importance"
                        )
                else:
                    st.info(
                        "Not enough anomalies detected to calculate feature importance"
                    )
            else:
                st.info(
                    "Original feature data not available for feature importance analysis"
                )
        else:
            st.info(
                "Please upload data and detect anomalies in the 'Upload & Detect' tab first to view visualizations"
            )

    with tab3:
        st.header("About This Application")
        st.markdown(
            """
        ## Anomaly Detection System
        
        This application uses an autoencoder neural network to detect anomalies in your CSV data. The autoencoder learns the normal patterns in your data and identifies records that deviate significantly from these patterns.
        
        ### How It Works
        
        1. **Upload your data and model files**: The application requires a CSV data file, a trained Keras model (.h5), and a fitted scaler (.npy).
        
        2. **Detection process**: The system scales your data, passes it through the autoencoder, and calculates the reconstruction error for each record.
        
        3. **Anomaly detection**: Records with reconstruction errors above the threshold (mean + threshold_factor × standard deviation) are flagged as anomalies.
        
        ### Features
        
        - **No file size restrictions**: Process CSV files of any size
        - **Memory-efficient**: Uses chunk-based processing to handle large datasets
        - **Detailed visualizations**: See the distribution of anomalies with numerical interpretations
        - **Feature importance analysis**: Understand which variables contribute most to anomalies
        - **Export options**: Download all results or just the anomalies
        
        ### Tips for Best Results
        
        - Ensure your CSV data has the same columns/features that the model was trained on
        - Adjust the threshold factor to control sensitivity (higher = less sensitive)
        - For very large datasets, use a smaller batch size to avoid memory issues
        - The model works best when it was trained on normal data (not containing anomalies)
        
        ### Output
        
        The application adds two columns to your data:
        - **ReconstructionError**: The mean squared error between the original and reconstructed data
        - **Anomaly**: Boolean flag (True/False) indicating whether the record is an anomaly
        """
        )


if __name__ == "__main__":
    main()
