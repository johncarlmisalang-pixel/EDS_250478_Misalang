"""
=============================================================================
Engineering Data Systems Pipeline — ENV-02: Aerosol Optical Depth vs. Humidity
Course: Computer Programming 1 | Academic Year: 2026
Student: Marv Andrew S. Inocencio | ID: TUPM-25-0351
Dataset: Global Air Quality Data (Kaggle)
File Used: global_air_quality_data_10000.csv (renamed to dataset_original.csv)
Unique Filter: Country == "India" (540 records — India-specific air quality slice)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)


class AerosolHumidityPipeline:
    """
    OOP Pipeline for ENV-02: Aerosol Optical Depth vs. Humidity Analysis.
    5 Modules: ingest_data, clean_data, analyze_data,
               visualize_static, visualize_animated
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_df   = None
        self.clean_df = None
        self.stats    = {}
        print("=" * 65)
        print("  ENV-02 Aerosol Optical Depth vs. Humidity Pipeline")
        print("  Student: Marv Andrew S. Inocencio | ID: TUPM-25-0351")
        print("=" * 65)

    # =========================================================================
    # MODULE 1: DATA INGESTION
    # =========================================================================
    def ingest_data(self):
        print("\n[1/5] INGESTING DATA...")
        try:
            df = pd.read_csv(self.filepath)
            print(f"  Loaded {len(df):,} total records.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found: {self.filepath}\n"
                "Make sure dataset_original.csv is inside the data/ folder."
            )
        except Exception as e:
            raise RuntimeError(f"Could not load file: {e}")

        # UNIQUE FILTER (Inocencio TUPM-25-0351):
        # Keep only rows where Country == "India"
        # India has the highest aerosol pollution load in the world,
        # making it the ideal geographic slice for ENV-02 analysis.
        try:
            self.raw_df = df[df["Country"] == "India"].reset_index(drop=True)
            print(f"  Unique Filter: Country == 'India'")
            print(f"  Records after filter: {len(self.raw_df):,}")
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

            key_cols = ["City", "Date", "PM2.5", "PM10", "NO2",
                        "SO2", "CO", "O3", "Temperature", "Humidity", "Wind Speed"]
            key_cols = [c for c in key_cols if c in df.columns]
            df = df[key_cols].copy()

            numeric_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO",
                            "O3", "Temperature", "Humidity", "Wind Speed"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            before_null = len(df)
            df.dropna(subset=numeric_cols, inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"  Null rows removed: {before_null - len(df)}")
            print(f"  Clean records remaining: {len(df):,}")

        except Exception as e:
            raise RuntimeError(f"Cleaning failed: {e}")

        # Aerosol Proxy Index (API):
        # PM2.5 and PM10 are ground-level proxies for Aerosol Optical Depth.
        # Normalized weighted sum — higher value = denser aerosol loading.
        df["Aerosol_Proxy"] = (0.6 * df["PM2.5"] + 0.4 * df["PM10"])

        # Humidity-Aerosol Interaction Score:
        # Hygroscopic growth — aerosols absorb moisture and swell,
        # increasing their optical depth. Higher humidity amplifies aerosol effect.
        df["Hygro_Score"] = df["Aerosol_Proxy"] * (df["Humidity"] / 100.0)

        self.clean_df = df
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
            target_cols = ["PM2.5", "PM10", "Humidity",
                           "Aerosol_Proxy", "Hygro_Score", "Temperature"]
            target_cols = [c for c in target_cols if c in df.columns]

            # --- Descriptive Statistics ---
            desc = {}
            for col in target_cols:
                arr = np.array(df[col].dropna())
                desc[col] = {
                    "mean"    : np.mean(arr),
                    "median"  : np.median(arr),
                    "std"     : np.std(arr, ddof=1),
                    "variance": np.var(arr, ddof=1),
                    "min"     : np.min(arr),
                    "max"     : np.max(arr),
                }
            stats["descriptive"] = desc

            # --- Skewness (Fisher-Pearson adjusted) ---
            def skewness(arr):
                n = len(arr)
                m = np.mean(arr)
                s = np.std(arr, ddof=1)
                if s == 0 or n < 3:
                    return 0.0
                return (n / ((n - 1) * (n - 2))) * np.sum(((arr - m) / s) ** 3)

            stats["skewness"] = {
                col: skewness(np.array(df[col].dropna())) for col in target_cols
            }

            # --- Outlier Detection (IQR Method) ---
            outliers = {}
            for col in target_cols:
                arr = np.array(df[col].dropna())
                Q1  = np.percentile(arr, 25)
                Q3  = np.percentile(arr, 75)
                IQR = Q3 - Q1
                low = Q1 - 1.5 * IQR
                up  = Q3 + 1.5 * IQR
                outliers[col] = {
                    "Q1"          : Q1,
                    "Q3"          : Q3,
                    "IQR"         : IQR,
                    "lower_fence" : low,
                    "upper_fence" : up,
                    "n_outliers"  : int(np.sum((arr < low) | (arr > up)))
                }
            stats["outliers"] = outliers

            # --- Pearson Correlation Matrix ---
            clean_num   = df[target_cols].dropna()
            corr_matrix = np.corrcoef(clean_num.values.T)
            stats["correlation_matrix"] = corr_matrix
            stats["correlation_cols"]   = target_cols

            # --- Comparative Group Analysis ---
            # High Humidity (>=55%) vs Low Humidity (<55%) groups
            # split at approximate median humidity
            hum_arr      = np.array(df["Humidity"])
            median_hum   = np.median(hum_arr)
            group_low    = df[df["Humidity"] <  median_hum]
            group_high   = df[df["Humidity"] >= median_hum]

            comp = {}
            for col in ["PM2.5", "PM10", "Aerosol_Proxy", "Hygro_Score"]:
                if col not in df.columns:
                    continue
                comp[col] = {
                    "low_hum_mean" : np.mean(np.array(group_low[col].dropna())),
                    "high_hum_mean": np.mean(np.array(group_high[col].dropna())),
                    "low_hum_std"  : np.std(np.array(group_low[col].dropna()),  ddof=1),
                    "high_hum_std" : np.std(np.array(group_high[col].dropna()), ddof=1),
                }
            stats["comparative"] = comp
            stats["median_hum"]  = median_hum

        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}")

        self.stats = stats
        self._print_summary()
        return stats

    def _print_summary(self):
        print("\n  ── DESCRIPTIVE STATISTICS ───────────────────────────────────")
        for col, vals in self.stats["descriptive"].items():
            print(f"  {col:<20}  mean={vals['mean']:>9.4f}  "
                  f"std={vals['std']:>9.4f}  var={vals['variance']:>12.4f}")
        print("\n  ── SKEWNESS ─────────────────────────────────────────────────")
        for col, val in self.stats["skewness"].items():
            print(f"  {col:<20}  skew={val:.4f}")
        print("\n  ── OUTLIER DETECTION (IQR) ──────────────────────────────────")
        for col, vals in self.stats["outliers"].items():
            print(f"  {col:<20}  IQR={vals['IQR']:>8.2f}  "
                  f"outliers={vals['n_outliers']}")
        print(f"\n  Humidity split (median): {self.stats['median_hum']:.2f}%")
        print("\n  ── COMPARATIVE: LOW vs HIGH HUMIDITY GROUP ──────────────────")
        for col, vals in self.stats["comparative"].items():
            print(f"  {col:<20}  "
                  f"Low Hum Mean={vals['low_hum_mean']:>8.3f}  "
                  f"High Hum Mean={vals['high_hum_mean']:>8.3f}")

    # =========================================================================
    # MODULE 4: STATIC VISUALIZATIONS (3 plots)
    # =========================================================================
    def visualize_static(self):
        print("\n[4/5] GENERATING STATIC VISUALIZATIONS...")
        df  = self.clean_df

        # Color palette
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

        # ── Plot 1: Histogram of Aerosol Proxy Index ──────────────────────────
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        style_ax(ax1, fig1)
        arr = np.array(df["Aerosol_Proxy"].dropna())
        ax1.hist(arr, bins=35, color=C1, edgecolor=BG, alpha=0.85)
        ax1.axvline(np.mean(arr),   color=C2, linestyle="--", linewidth=2,
                    label=f"Mean = {np.mean(arr):.2f}")
        ax1.axvline(np.median(arr), color=C3, linestyle=":",  linewidth=2,
                    label=f"Median = {np.median(arr):.2f}")
        ax1.set_title(
            "Distribution of Aerosol Proxy Index (India)\nInocencio | TUPM-25-0351",
            color=TXT, fontsize=13, fontweight="bold"
        )
        ax1.set_xlabel("Aerosol Proxy Index (weighted PM2.5 + PM10)", color=TXT)
        ax1.set_ylabel("Frequency", color=TXT)
        ax1.legend(facecolor=PAN, labelcolor=TXT)
        plt.tight_layout()
        fig1.savefig("outputs/plot1_histogram.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig1)
        print("  Saved: outputs/plot1_histogram.png")

        # ── Plot 2: Boxplot — Low Humidity vs High Humidity Groups ────────────
        median_hum = self.stats["median_hum"]
        low_grp    = df[df["Humidity"] <  median_hum]["Aerosol_Proxy"].values
        high_grp   = df[df["Humidity"] >= median_hum]["Aerosol_Proxy"].values

        fig2, ax2 = plt.subplots(figsize=(9, 6))
        style_ax(ax2, fig2)
        bp = ax2.boxplot(
            [low_grp, high_grp],
            labels=[f"Low Humidity\n(<{median_hum:.1f}%)",
                    f"High Humidity\n(≥{median_hum:.1f}%)"],
            patch_artist=True,
            medianprops=dict(color=C2, linewidth=2.5),
            whiskerprops=dict(color=TXT),
            capprops=dict(color=TXT),
            flierprops=dict(marker="o", color=C3, alpha=0.5, markersize=4)
        )
        bp["boxes"][0].set_facecolor("#1e3a5f")
        bp["boxes"][1].set_facecolor("#3b1f2b")
        for box in bp["boxes"]:
            box.set_edgecolor(C1)
        ax2.set_title(
            "Aerosol Proxy: Low Humidity vs High Humidity Groups (India)\n"
            "Inocencio | TUPM-25-0351",
            color=TXT, fontsize=13, fontweight="bold"
        )
        ax2.set_ylabel("Aerosol Proxy Index", color=TXT)
        plt.tight_layout()
        fig2.savefig("outputs/plot2_boxplot.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
        print("  Saved: outputs/plot2_boxplot.png")

        # ── Plot 3: Pearson Correlation Heatmap ───────────────────────────────
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
        ax3.set_title(
            "Pearson Correlation Heatmap — Air Quality Variables (India)\n"
            "Inocencio | TUPM-25-0351",
            color=TXT, fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        fig3.savefig("outputs/plot3_heatmap.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig3)
        print("  Saved: outputs/plot3_heatmap.png")

        # ── Plot 4 (bonus): Scatter — Humidity vs Aerosol Proxy ──────────────
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        style_ax(ax4, fig4)
        hum = np.array(df["Humidity"].dropna())
        aer = np.array(df["Aerosol_Proxy"].dropna())
        sc  = ax4.scatter(hum, aer, c=np.array(df["PM2.5"].dropna()),
                          cmap="plasma", alpha=0.6, s=25, edgecolors="none")
        cbar4 = plt.colorbar(sc, ax=ax4)
        cbar4.set_label("PM2.5 (µg/m³)", color=TXT)
        cbar4.ax.yaxis.set_tick_params(color=TXT)
        plt.setp(cbar4.ax.yaxis.get_ticklabels(), color=TXT)

        # Linear trend line
        m, b = np.polyfit(hum, aer, 1)
        x_line = np.linspace(hum.min(), hum.max(), 200)
        ax4.plot(x_line, m * x_line + b, color=C2, linewidth=2,
                 linestyle="--", label=f"Trend: y={m:.2f}x+{b:.1f}")
        ax4.set_title(
            "Humidity vs Aerosol Proxy Index (India)\nInocencio | TUPM-25-0351",
            color=TXT, fontsize=13, fontweight="bold"
        )
        ax4.set_xlabel("Relative Humidity (%)", color=TXT)
        ax4.set_ylabel("Aerosol Proxy Index", color=TXT)
        ax4.legend(facecolor=PAN, labelcolor=TXT)
        plt.tight_layout()
        fig4.savefig("outputs/plot4_scatter_humidity_aerosol.png", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig4)
        print("  Saved: outputs/plot4_scatter_humidity_aerosol.png")

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

        # ── Animation 1: Rolling Mean of Aerosol Proxy (Matplotlib GIF) ──────
        arr    = np.array(df["Aerosol_Proxy"].dropna())
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
                  alpha=0.5, label="Raw Aerosol Proxy")
        line_m, = ax_a.plot([], [], color=C1, linewidth=2.2,
                             label=f"Rolling Mean ({window}-window)")
        ax_a.set_xlim(0, len(arr))
        ax_a.set_ylim(max(0, arr.min() - 10), arr.max() + 10)
        ax_a.set_title(
            "Animated Rolling Mean — Aerosol Proxy Index (India)\n"
            "Inocencio | TUPM-25-0351",
            color=TXT, fontsize=13, fontweight="bold"
        )
        ax_a.set_xlabel("Record Index", color=TXT)
        ax_a.set_ylabel("Aerosol Proxy Index", color=TXT)
        ax_a.legend(facecolor=PAN, labelcolor=TXT)

        def init_a():
            line_m.set_data([], [])
            return (line_m,)

        def update_a(frame):
            line_m.set_data(x_roll[:frame + 1], roll[:frame + 1])
            return (line_m,)

        anim1 = animation.FuncAnimation(
            fig_a, update_a, frames=min(len(roll), 180),
            init_func=init_a, blit=True, interval=30
        )
        anim1.save("outputs/animation1_rolling_mean.gif",
                   writer="pillow", fps=25, dpi=110)
        plt.close(fig_a)
        print("  Saved: outputs/animation1_rolling_mean.gif")

        # ── Animation 2: Scatter reveal — Humidity vs Hygro Score ─────────────
        from matplotlib.patches import Patch

        plot_df = df[["Humidity", "Hygro_Score"]].dropna().reset_index(drop=True)
        median_hum = self.stats["median_hum"]
        hum_vals   = plot_df["Humidity"].values
        hsc_vals   = plot_df["Hygro_Score"].values
        n          = len(hum_vals)
        step       = max(1, n // 60)
        colors_arr = np.where(hum_vals >= median_hum, C2, C1)

        fig_b, ax_b = plt.subplots(figsize=(10, 6))
        fig_b.patch.set_facecolor(BG)
        ax_b.set_facecolor(PAN)
        ax_b.tick_params(colors=TXT)
        for sp in ax_b.spines.values():
            sp.set_edgecolor(BDR)
        ax_b.set_xlim(hum_vals.min() - 2, hum_vals.max() + 2)
        ax_b.set_ylim(hsc_vals.min() - 2, hsc_vals.max() + 2)
        ax_b.set_title(
            "Animated Humidity vs Hygroscopic Aerosol Score — Cluster Analysis\n"
            "Inocencio | TUPM-25-0351",
            color=TXT, fontsize=13, fontweight="bold"
        )
        ax_b.set_xlabel("Relative Humidity (%)", color=TXT)
        ax_b.set_ylabel("Hygroscopic Aerosol Score", color=TXT)
        legend_elements = [
            Patch(facecolor=C1, label=f"Low Humidity (<{median_hum:.1f}%)"),
            Patch(facecolor=C2, label=f"High Humidity (≥{median_hum:.1f}%)"),
        ]
        ax_b.legend(handles=legend_elements, facecolor=PAN, labelcolor=TXT)

        scat       = ax_b.scatter([], [], s=35, alpha=0.7)
        frame_text = ax_b.text(0.02, 0.96, "", transform=ax_b.transAxes,
                               color=TXT, fontsize=9, va="top")

        frames_b = list(range(step, n + 1, step))
        if not frames_b or frames_b[-1] < n:
            frames_b.append(n)

        def init_b():
            scat.set_offsets(np.empty((0, 2)))
            scat.set_color([])
            frame_text.set_text("")
            return scat, frame_text

        def update_b(i):
            idx = frames_b[i]
            scat.set_offsets(np.column_stack([hum_vals[:idx], hsc_vals[:idx]]))
            scat.set_color(colors_arr[:idx])
            frame_text.set_text(f"Records: {idx}/{n}")
            return scat, frame_text

        anim2 = animation.FuncAnimation(
            fig_b, update_b, frames=len(frames_b),
            init_func=init_b, blit=True, interval=60
        )
        anim2.save("outputs/animation2_scatter_cluster.gif",
                   writer="pillow", fps=15, dpi=110)
        plt.close(fig_b)
        print("  Saved: outputs/animation2_scatter_cluster.gif")

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
    pipeline = AerosolHumidityPipeline(filepath="data/dataset_original.csv")
    pipeline.run()
