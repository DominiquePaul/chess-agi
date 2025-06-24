# Example: Using the Chess Piece Detection Model
import os
from pathlib import Path

from dotenv import load_dotenv

from src.chess_piece_detection.model import ChessModel
from src.utils import is_notebook

load_dotenv()

# Configure matplotlib for better notebook support
if is_notebook():
    try:
        import matplotlib

        matplotlib.use("inline")
        from IPython.core.getipython import get_ipython

        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("matplotlib", "inline")
    except Exception:
        pass

# Set up paths
single_eval_img_path = Path(os.environ["DATA_FOLDER_PATH"]) / "eval_images/chess_4.jpeg"
data_merged_yaml_path = Path(os.environ["DATA_FOLDER_PATH"]) / "chess_pieces_merged" / "data.yaml"

# Create artifacts folder for saving results
artifacts_folder = Path("artifacts")
artifacts_folder.mkdir(exist_ok=True)

print("🏆 Loading the improved chess piece detection model...")

# Load the latest improved model (71% mAP50!)
model_v2 = ChessModel.from_huggingface("dopaul/chess-piece-detector-merged-v2")

print("✅ Model loaded successfully!")

# List available model files in the repository
print("\n📋 Available model files:")
available_files = ChessModel.list_huggingface_files("dopaul/chess-piece-detector-merged-v2")


def count_detections(results):
    """Helper function to safely count detections, handling None cases."""
    if results is not None and results.boxes is not None:
        return len(results.boxes)
    return 0


# === INFERENCE EXAMPLES ===

print("\n🔍 Running inference on test image...")
# Predict on a single image with high confidence threshold
results = model_v2.predict(single_eval_img_path, conf=0.4, save=True)
detection_count = count_detections(results)
print(f"📊 Detected {detection_count} chess pieces")

print("\n🔍 Running inference with lower confidence threshold...")
# Predict with lower confidence to catch more pieces
results_low_conf = model_v2.predict(single_eval_img_path, conf=0.25, save=True)
detection_count_low = count_detections(results_low_conf)
print(f"📊 Detected {detection_count_low} chess pieces (lower threshold)")

# === EVALUATION EXAMPLES ===

print("\n📊 Evaluating model performance on test dataset...")
# Evaluate on the merged dataset (this will show the 71% mAP50 performance)
if data_merged_yaml_path.exists():
    model_v2.evaluate(data_merged_yaml_path)
else:
    print("⚠️ Merged dataset not found, skipping evaluation")

# === PLOTTING EXAMPLES ===

print("\n📈 Creating detection visualization...")
# Plot detection results with bounding boxes
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    model_v2.plot_eval(single_eval_img_path, ax=ax, conf=0.4)
    plt.title("Chess Piece Detection Results (mAP50: 71.0%)")
    plt.tight_layout()

    # Save to artifacts folder
    output_path = artifacts_folder / "chess_detection_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Visualization saved as '{output_path}'")

    if is_notebook():
        plt.show()
    else:
        plt.close()

except Exception as e:
    print(f"⚠️ Could not create visualization: {e}")

# === COMPARISON WITH OLDER MODEL ===

print("\n🔄 Comparing with previous model version...")
try:
    # Load the original model for comparison
    model_v1 = ChessModel.from_huggingface("dopaul/chess-piece-detector-merged")

    print("🆚 Performance Comparison:")
    print("   v1 (Original): ~39% mAP50")
    print("   v2 (Enhanced): ~71% mAP50")
    print("   🚀 Improvement: +81%")

    # Test multiple confidence thresholds
    confidence_levels = [0.3, 0.4, 0.5]

    print("\n📊 Detection Comparison at Different Confidence Thresholds:")
    print("=" * 60)
    print(f"{'Confidence':<12} {'v1 Model':<12} {'v2 Model':<12} {'Improvement':<12}")
    print("-" * 60)

    # Store results for visualization
    comparison_results = []

    for conf in confidence_levels:
        # Run inference with both models
        results_v1 = model_v1.predict(single_eval_img_path, conf=conf)
        results_v2 = model_v2.predict(single_eval_img_path, conf=conf)

        v1_count = count_detections(results_v1)
        v2_count = count_detections(results_v2)

        # Calculate improvement
        if v1_count > 0:
            improvement = f"+{((v2_count - v1_count) / v1_count * 100):.1f}%"
        else:
            improvement = "N/A" if v2_count == 0 else "+∞"

        print(f"{conf:<12.1f} {v1_count:<12} {v2_count:<12} {improvement:<12}")

        # Store for visualization
        comparison_results.append(
            {
                "conf": conf,
                "v1_count": v1_count,
                "v2_count": v2_count,
                "results_v1": results_v1,
                "results_v2": results_v2,
            }
        )

    print("=" * 60)

    # Create comprehensive comparison visualization
    try:
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle(
            "Model Comparison: v1 vs v2 at Different Confidence Levels",
            fontsize=16,
            fontweight="bold",
        )

        for i, result in enumerate(comparison_results):
            conf = result["conf"]
            v1_count = result["v1_count"]
            v2_count = result["v2_count"]

            # Top row: v1 model results
            ax_v1 = axes[0, i]
            model_v1.plot_eval(single_eval_img_path, ax=ax_v1, conf=conf)
            ax_v1.set_title(
                f"v1 Model (conf={conf}): {v1_count} detections",
                fontsize=12,
                fontweight="bold",
            )

            # Bottom row: v2 model results
            ax_v2 = axes[1, i]
            model_v2.plot_eval(single_eval_img_path, ax=ax_v2, conf=conf)
            ax_v2.set_title(
                f"v2 Model (conf={conf}): {v2_count} detections",
                fontsize=12,
                fontweight="bold",
            )

        plt.tight_layout()

        # Save comprehensive comparison to artifacts folder
        comprehensive_comparison_path = artifacts_folder / "comprehensive_model_comparison.png"
        plt.savefig(comprehensive_comparison_path, dpi=150, bbox_inches="tight")
        print(f"✅ Comprehensive model comparison saved as '{comprehensive_comparison_path}'")

        if is_notebook():
            plt.show()
        else:
            plt.close()

    except Exception as viz_e:
        print(f"⚠️ Could not create comprehensive comparison visualization: {viz_e}")

    # Create a simple bar chart comparison
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        conf_labels = [f"conf={c}" for c in confidence_levels]
        v1_counts = [r["v1_count"] for r in comparison_results]
        v2_counts = [r["v2_count"] for r in comparison_results]

        x = range(len(confidence_levels))
        width = 0.35

        bars1 = ax.bar(
            [i - width / 2 for i in x],
            v1_counts,
            width,
            label="v1 Model (39% mAP50)",
            color="lightcoral",
            alpha=0.8,
        )
        bars2 = ax.bar(
            [i + width / 2 for i in x],
            v2_counts,
            width,
            label="v2 Model (71% mAP50)",
            color="lightblue",
            alpha=0.8,
        )

        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Number of Detections")
        ax.set_title("Detection Count Comparison: v1 vs v2 Models")
        ax.set_xticks(x)
        ax.set_xticklabels(conf_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()

        # Save bar chart comparison
        bar_chart_path = artifacts_folder / "detection_count_comparison.png"
        plt.savefig(bar_chart_path, dpi=150, bbox_inches="tight")
        print(f"✅ Detection count comparison chart saved as '{bar_chart_path}'")

        if is_notebook():
            plt.show()
        else:
            plt.close()

    except Exception as chart_e:
        print(f"⚠️ Could not create bar chart comparison: {chart_e}")

except Exception as e:
    print(f"⚠️ Could not load v1 model for comparison: {e}")

# === USAGE TIPS ===

print("\n💡 Usage Tips:")
print("   • conf=0.5: High precision, may miss some pieces")
print("   • conf=0.25: Higher recall, may have false positives")
print("   • conf=0.4: Good balance (recommended)")
print("   • save=True: Saves annotated images to runs/ folder")
print("   • Model achieves 71% mAP50 with 77.7% precision")
print(f"   • Results saved to: {artifacts_folder.absolute()}")

print("\n🎉 Chess piece detection model ready for use!")
