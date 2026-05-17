"""
=============================================================================
Engineering Data Systems Pipeline — AIP-03: Confidence Calibration
Course: Computer Programming 1 | Academic Year: 2026
Student: John Carl Misalang | ID: 25-0478
Dataset: Hiring Decision Dataset (Algorithmic Bias in Recruitment)
File Used: data.csv (renamed to dataset_original.csv)
Unique Filter: 
  Cohort A (Younger): Age < 30
  Cohort B (Older/High-Exp): Age > 40 & ExperienceYears > 10
  Comparing hiring rates between cohorts to analyze confidence calibration bias
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.express as px
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)


class ConfidenceCalibrationPipeline:
    """
    OOP Pipeline for AIP-03: Confidence Calibration Analysis.
    5 Modules: ingest_data, clean_data, analyze_data,
               visualize_static, visualize_animated
    """

    def __init__(self, filepath: str):
        self.filepath  = filepath
        self.raw_df    = None
        self.clean_df  = None
        self.cohort_a  = None
        self.cohort_b  = None
        self.stats     = {}
        print("=" * 65)
        print("  AIP-03 Confidence Calibration Pipeline")
        print("  Student: John Carl Misalang | ID: 25-0478")
        print("=" * 65)

    # =========================================================================
    # MODULE 1: DATA INGESTION
    # =========================================================================
    def ingest_data(self):
        print("\n[1/5] INGESTING DATA...")
        try:
            df = pd.read_csv(self.filepath)
            print(f"  Loaded {len(df):,} total records.")
            print(f"  Columns: {list(df.columns)}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found: {self.filepath}\n"
                "Make sure dataset_original.csv is inside the data/ folder."
            )
        except Exception as e:
            raise RuntimeError(f"Could not load file: {e}")

        # Standardize column names
        df.columns = df.columns.str.strip()

        # UNIQUE FILTER (Misalang 25-0478):
        # Cohort A: Younger candidates (Age < 30)
        # Cohort B: Older high-experience candidates (Age > 40 & ExperienceYears > 10)
        # This isolates two distinct demographic groups for hiring bias analysis
        try:
            df["Age"]             = pd.to_numeric(df["Age"],             errors="coerce")
            df["ExperienceYears"] = pd.to_numeric(df["ExperienceYears"], errors="coerce")

            cohort_a = df[df["Age"] < 30].copy()
            cohort_b = df[(df["Age"] > 40) & (df["ExperienceYears"] > 10)].copy()

            cohort_a["Cohort"] = "Cohort A (Age < 30)"
            cohort_b["Cohort"] = "Cohort B (Age > 40 & Exp > 10)"

            self.raw_df = pd.concat([cohort_a, cohort_b]).reset_index(drop=True)
            print(f"  Unique Filter Applied:")
            print(f"    Cohort A (Age < 30):              {len(cohort_a):,} records")
            print(f"    Cohort B (Age > 40 & Exp > 10):   {len(cohort_b):,} records")
            print(f"    Combined:                          {len(self.raw_df):,} records")

        except Exception as e:
            raise RuntimeError(f"Filter failed: {e}")

        self.raw_df.to_csv("data/dataset_original.csv", index=False)
        print("  Saved to data/dataset_original.csv")
        return self.raw_df

    # =========================================================================
    # MODULE 2: DATA CLEANING
    # =========================================================================
    def clean_data(self):
        print("\n[2/5] CLEANING DATA...")
        df = self.raw_df.copy()

        try:
            before = len(df)
            df.drop_duplicates(inplace=True)
            print(f"  Duplicates removed: {before - len(df)}")

            # Select key columns for analysis
            key_cols = [
                "Age", "ExperienceYears", "InterviewScore",
                "SkillScore", "PersonalityScore", "HiringDecision",
                "Cohort"
            ]
            key_cols = [c for c in key_cols if c in df.columns]
            df = df[key_cols].copy()

            # Coerce numeric columns
            numeric_cols = [c for c in key_cols if c != "Cohort"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            before_null = len(df)
            df.dropna(subset=numeric_cols, inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"  Null rows removed: {before_null - len(df)}")
            print(f"  Clean records remaining: {len(df):,}")

        except Exception as e:
            raise RuntimeError(f"Cleaning failed: {e}")

        # Compute Confidence Calibration Score
        # Higher interview + skill + personality = higher model confidence
        # We compare this against actual hiring decision to measure calibration
        if all(c in df.columns for c in ["InterviewScore", "SkillScore", "PersonalityScore"]):
            int_arr  = np.array(df["InterviewScore"])
            sk_arr   = np.array(df["SkillScore"])
            per_arr  = np.array(df["PersonalityScore"])
            df["Confidence_Score"] = (
                (int_arr / 100) * 0.4 +
                (sk_arr  / 100) * 0.35 +
                (per_arr / 100) * 0.25
            )
            print("  Computed: Confidence_Score column added")

        # Calibration Error = |Confidence_Score - HiringDecision|
        if "Confidence_Score" in df.columns and "HiringDecision" in df.columns:
            df["Calibration_Error"] = np.abs(
                df["Confidence_Score"] - df["HiringDecision"]
            )
            print("  Computed: Calibration_Error column added")

        self.clean_df = df
        self.cohort_a = df[df["Cohort"] == "Cohort A (Age < 30)"]
        self.cohort_b = df[df["Cohort"] == "Cohort B (Age > 40 & Exp > 10)"]
        self.clean_df.to_csv("data/dataset_cleaned.csv", index=False)
        print("  Saved to data/dataset_cleaned.csv")
        return self.clean_df

    # =========================================================================
    # MODULE 3: STATISTICAL ANALYSIS (NumPy)
    # =========================================================================
    def analyze_data(self):
        print("\n[3/5] PERFORMING STATISTICAL ANALYSIS...")
        df = self.clean_df
        stats = {}

        try:
            target_cols = [c for c in [
                "Confidence_Score", "Calibration_Error",
                "InterviewScore", "SkillScore",
                "PersonalityScore", "HiringDecision"
            ] if c in df.columns]

            # Descriptive Statistics
            desc = {}
            for col in target_cols:
                arr = np.array(df[col])
                desc[col] = {
                    "mean"    : np.mean(arr),
                    "median"  : np.median(arr),
                    "std"     : np.std(arr, ddof=1),
                    "variance": np.var(arr, ddof=1),
                    "min"     : np.min(arr),
                    "max"     : np.max(arr),
                }
            stats["descriptive"] = desc

            # Skewness
            def skewness(arr):
                n = len(arr)
                m = np.mean(arr)
                s = np.std(arr, ddof=1)
                if s == 0:
                    return 0
                return (n / ((n-1)*(n-2))) * np.sum(((arr - m) / s) ** 3)

            stats["skewness"] = {
                col: skewness(np.array(df[col])) for col in target_cols
            }

            # Outlier Detection (IQR)
            outliers = {}
            for col in target_cols:
                arr = np.array(df[col])
                Q1  = np.percentile(arr, 25)
                Q3  = np.percentile(arr, 75)
                IQR = Q3 - Q1
                low = Q1 - 1.5 * IQR
                up  = Q3 + 1.5 * IQR
                outliers[col] = {
                    "Q1": Q1, "Q3": Q3, "IQR": IQR,
                    "lower_fence": low, "upper_fence": up,
                    "n_outliers": int(np.sum((arr < low) | (arr > up)))
                }
            stats["outliers"] = outliers

            # Pearson Correlation
            corr_matrix = np.corrcoef(df[target_cols].values.T)
            stats["correlation_matrix"] = corr_matrix
            stats["correlation_cols"]   = target_cols

            # Cohort Comparative Analysis
            comp = {}
            for col in ["Confidence_Score", "Calibration_Error",
                        "InterviewScore", "SkillScore", "HiringDecision"]:
                if col not in df.columns:
                    continue
                arr_a = np.array(self.cohort_a[col].dropna())
                arr_b = np.array(self.cohort_b[col].dropna())
                comp[col] = {
                    "cohort_a_mean": np.mean(arr_a),
                    "cohort_b_mean": np.mean(arr_b),
                    "cohort_a_std" : np.std(arr_a, ddof=1),
                    "cohort_b_std" : np.std(arr_b, ddof=1),
                }
            stats["comparative"] = comp

            # Hiring rates per cohort
            hr_a = np.mean(np.array(self.cohort_a["HiringDecision"]))
            hr_b = np.mean(np.array(self.cohort_b["HiringDecision"]))
            stats["hiring_rate_a"] = hr_a
            stats["hiring_rate_b"] = hr_b
            print(f"  Cohort A Hiring Rate: {hr_a:.4f} ({hr_a*100:.1f}%)")
            print(f"  Cohort B Hiring Rate: {hr_b:.4f} ({hr_b*100:.1f}%)")

        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}")

        self.stats = stats
        self._print_summary()
        return stats

    def _print_summary(self):
        print("\n  ── DESCRIPTIVE STATISTICS ───────────────────────────────────")
        for col, vals in self.stats["descriptive"].items():
            print(f"\n  [{col}]")
            for k, v in vals.items():
                print(f"      {k:>10}: {v:>12.4f}")
        print("\n  ── SKEWNESS ─────────────────────────────────────────────────")
        for col, s in self.stats["skewness"].items():
            print(f"      {col:>22}: {s:>10.4f}")
        print("\n  ── OUTLIERS (IQR method) ────────────────────────────────────")
        for col, o in self.stats["outliers"].items():
            print(f"      {col:>22}: {o['n_outliers']} outliers | IQR = {o['IQR']:.4f}")
        print("  ─────────────────────────────────────────────────────────────\n")

    # =========================================================================
    # MODULE 4: STATIC VISUALIZATIONS (3 charts)
    # =========================================================================
    def visualize_static(self):
        print("[4/5] GENERATING STATIC VISUALIZATIONS...")
        df  = self.clean_df
        BG  = "#0d1117"
        PAN = "#161b22"
        C1  = "#58a6ff"
        C2  = "#f78166"
        C3  = "#3fb950"
        C4  = "#d2a8ff"
        TXT = "#e6edf3"
        BDR = "#30363d"

        def style_ax(ax, fig):
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(PAN)
            ax.tick_params(colors=TXT)
            for spine in ax.spines.values():
                spine.set_edgecolor(BDR)

        # Plot 1: Histogram of Confidence Score by Cohort
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        style_ax(ax1, fig1)
        arr_a = np.array(self.cohort_a["Confidence_Score"].dropna())
        arr_b = np.array(self.cohort_b["Confidence_Score"].dropna())
        ax1.hist(arr_a, bins=20, color=C1, alpha=0.7, label="Cohort A (Age < 30)")
        ax1.hist(arr_b, bins=20, color=C2, alpha=0.7, label="Cohort B (Age > 40 & Exp > 10)")
        ax1.axvline(np.mean(arr_a), color=C1, linestyle="--", linewidth=2)
        ax1.axvline(np.mean(arr_b), color=C2, linestyle="--", linewidth=2)
        ax1.set_title("Distribution of Confidence Score by Cohort\nMisalang | 25-0478",
                      color=TXT, fontsize=13, fontweight="bold")
        ax1.set_xlabel("Confidence Score", color=TXT)
        ax1.set_ylabel("Frequency", color=TXT)
        ax1.legend(facecolor=PAN, labelcolor=TXT)
        plt.tight_layout()
        fig1.savefig("outputs/plot1_histogram.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig1)
        print("  Saved: outputs/plot1_histogram.png")

        # Plot 2: Boxplot — Hiring Rate Cohort A vs B
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        style_ax(ax2, fig2)
        data_a = np.array(self.cohort_a["Confidence_Score"].dropna())
        data_b = np.array(self.cohort_b["Confidence_Score"].dropna())
        bp = ax2.boxplot(
            [data_a, data_b],
            labels=["Cohort A\n(Age < 30)", "Cohort B\n(Age > 40 & Exp > 10)"],
            patch_artist=True,
            medianprops=dict(color=C3, linewidth=2.5),
            whiskerprops=dict(color=TXT),
            capprops=dict(color=TXT),
            flierprops=dict(marker="o", color=C4, alpha=0.5, markersize=4)
        )
        bp["boxes"][0].set_facecolor("#1e3a5f")
        bp["boxes"][1].set_facecolor("#3b1f2b")
        for box in bp["boxes"]:
            box.set_edgecolor(C1)
        ax2.set_title("Boxplot: Confidence Score — Cohort A vs Cohort B\nMisalang | 25-0478",
                      color=TXT, fontsize=13, fontweight="bold")
        ax2.set_ylabel("Confidence Score", color=TXT)
        plt.tight_layout()
        fig2.savefig("outputs/plot2_boxplot.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
        print("  Saved: outputs/plot2_boxplot.png")

        # Plot 3: Correlation Heatmap
        corr_cols   = self.stats["correlation_cols"]
        corr_matrix = self.stats["correlation_matrix"]
        fig3, ax3 = plt.subplots(figsize=(9, 7))
        style_ax(ax3, fig3)
        im = ax3.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        cbar = plt.colorbar(im, ax=ax3)
        cbar.ax.yaxis.set_tick_params(color=TXT)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TXT)
        ax3.set_xticks(range(len(corr_cols)))
        ax3.set_yticks(range(len(corr_cols)))
        ax3.set_xticklabels(corr_cols, rotation=35, ha="right",
                             color=TXT, fontsize=8)
        ax3.set_yticklabels(corr_cols, color=TXT, fontsize=8)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr_matrix[i, j]
                ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if abs(val) > 0.4 else "#888", fontsize=8)
        ax3.set_title("Pearson Correlation Heatmap\nMisalang | 25-0478",
                      color=TXT, fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig3.savefig("outputs/plot3_heatmap.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig3)
        print("  Saved: outputs/plot3_heatmap.png")

    # =========================================================================
    # MODULE 5: ANIMATED VISUALIZATIONS (2 animations)
    # =========================================================================
    def visualize_animated(self):
        print("\n[5/5] GENERATING ANIMATED VISUALIZATIONS...")
        df  = self.clean_df
        BG  = "#0d1117"
        PAN = "#161b22"
        C1  = "#58a6ff"
        C2  = "#f78166"
        TXT = "#e6edf3"
        BDR = "#30363d"

        # Animation 1: Rolling Mean of Calibration Error (Matplotlib GIF)
        arr    = np.array(df["Calibration_Error"].dropna())[:400]
        window = 20
        roll   = np.convolve(arr, np.ones(window) / window, mode="valid")
        x_raw  = np.arange(len(arr))
        x_roll = np.arange(window - 1, len(arr))

        fig_a, ax_a = plt.subplots(figsize=(11, 5))
        fig_a.patch.set_facecolor(BG)
        ax_a.set_facecolor(PAN)
        ax_a.tick_params(colors=TXT)
        for sp in ax_a.spines.values():
            sp.set_edgecolor(BDR)
        ax_a.plot(x_raw, arr, color="#30363d", linewidth=0.8,
                  alpha=0.5, label="Raw Calibration Error")
        line_m, = ax_a.plot([], [], color=C1, linewidth=2.2,
                             label="Rolling Mean (20-window)")
        ax_a.set_xlim(0, len(arr))
        ax_a.set_ylim(0, arr.max() + 0.05)
        ax_a.set_title("Animated Rolling Mean — Calibration Error\nMisalang | 25-0478",
                        color=TXT, fontsize=13, fontweight="bold")
        ax_a.set_xlabel("Record Index", color=TXT)
        ax_a.set_ylabel("Calibration Error", color=TXT)
        ax_a.legend(facecolor=PAN, labelcolor=TXT)

        def init():
            line_m.set_data([], [])
            return (line_m,)

        def update(frame):
            line_m.set_data(x_roll[:frame+1], roll[:frame+1])
            return (line_m,)

        anim1 = animation.FuncAnimation(
            fig_a, update, frames=min(len(roll), 180),
            init_func=init, blit=True, interval=30
        )
        anim1.save("outputs/animation1_rolling_mean.gif",
                   writer="pillow", fps=25, dpi=110)
        plt.close(fig_a)
        print("  Saved: outputs/animation1_rolling_mean.gif")

        # Animation 2: Scatter — Confidence Score vs Hiring Decision (Plotly)
        plot_df = df[["Confidence_Score", "HiringDecision",
                      "Cohort", "InterviewScore"]].copy().dropna()
        plot_df["frame"] = (
            np.arange(len(plot_df)) // max(1, len(plot_df) // 25)
        )

        fig_b = px.scatter(
            plot_df,
            x="Confidence_Score",
            y="HiringDecision",
            color="Cohort",
            animation_frame="frame",
            size="InterviewScore",
            size_max=15,
            color_discrete_map={
                "Cohort A (Age < 30)":          "#58a6ff",
                "Cohort B (Age > 40 & Exp > 10)": "#f78166"
            },
            title="Animated Confidence Score vs Hiring Decision — Cohort Comparison<br>"
                  "<sup>AIP-03: Misalang 25-0478</sup>",
            labels={
                "Confidence_Score": "Confidence Score",
                "HiringDecision":   "Hiring Decision (0=No, 1=Yes)"
            },
            template="plotly_dark",
            opacity=0.75,
        )
        fig_b.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font_color="#e6edf3",
            title_font_size=14,
        )
        fig_b.write_html("outputs/animation2_scatter_cluster.html")
        print("  Saved: outputs/animation2_scatter_cluster.html")

    # =========================================================================
    # RUN FULL PIPELINE
    # =========================================================================
    def run(self):
        self.ingest_data()
        self.clean_data()
        self.analyze_data()
        self.visualize_static()
        self.visualize_animated()
        print("\n" + "=" * 65)
        print("  PIPELINE COMPLETE — check your outputs/ folder!")
        print("=" * 65)


if __name__ == "__main__":
    pipeline = ConfidenceCalibrationPipeline(filepath="data/dataset_original.csv")
    pipeline.run()
