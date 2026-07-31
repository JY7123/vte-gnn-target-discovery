@echo off
REM reproduce.bat — Windows one-click reproduction for VTE GNN Target Discovery
setlocal enabledelayedexpansion

echo ============================================
echo VTE GNN Target Discovery — Reproduction
echo ============================================

REM ── 1. Verify environment ──
echo.
echo [1/5] Verifying environment...
python -c "import torch; print(f'  PyTorch {torch.__version__}')"
python -c "import torch_geometric; print(f'  PyG {torch_geometric.__version__}')"

REM ── 2. Verify data ──
echo.
echo [2/5] Verifying data files...
for %%f in (data\processed\heterodata.pt data\processed\train_edges.pt data\processed\negative_edges.pt) do (
    if exist "%%f" (
        echo   [OK] %%f
    ) else (
        echo   [MISSING] %%f — please download from Zenodo [DOI]
        exit /b 1
    )
)
if exist "checkpoints\pca_features\features_128d.pt" (
    echo   [OK] PCA features found
) else (
    echo   [INFO] PCA features not found — will use random 128d features
)

REM ── 3. Run tests ──
echo.
echo [3/5] Running tests...
python -m pytest tests/ -x -q --tb=short

REM ── 4. Train model ──
echo.
echo [4/5] Training Tempered HGT (PCA 128d, 93 epochs)...
python train_full_v2.py

REM ── 5. Generate figures ──
echo.
echo [5/5] Generating paper figures...
where Rscript >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    Rscript scripts/render_figure1_kg_temporal.R
    Rscript scripts/render_figure2_benchmark.R
    Rscript scripts/render_figure3_target_ranking.R
    Rscript scripts/render_figure4_scRNA_mapping.R
    Rscript scripts/render_figure5_cross_species.R
    echo   All Figures 1-5 complete.
) else (
    echo   [SKIP] Rscript not found. Figures can be generated separately with:
    echo     Rscript scripts/render_figure1_kg_temporal.R
    echo     Rscript scripts/render_figure2_benchmark.R
    echo     Rscript scripts/render_figure3_target_ranking.R
    echo     Rscript scripts/render_figure4_scRNA_mapping.R
    echo     Rscript scripts/render_figure5_cross_species.R
)

echo.
echo ============================================
echo Reproduction complete.
echo Figures saved to figures\paper_figures\
echo ============================================
