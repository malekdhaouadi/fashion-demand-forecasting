# Fashion demand forecasting with image-based style features and MLOps

![CI](https://github.com/<your-user>/<repo>/actions/workflows/ci.yml/badge.svg)

Can product images improve weekly demand forecasts for fashion categories, and can the
pipeline around the model be kept reliable over time?

- **Modelling question:** does visual information help?
- **Engineering question:** can the system validate its data, track every run, and
  promote a model only when it is really better?

## Data: what is real and what is simulated

| Part | Source | Status |
|---|---|---|
| Product images, category, colour, season, gender | [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) (Kaggle, MIT licence) | **Real** |
| Weekly sales per product | `src/data/simulate.py` | **Simulated** |

No public dataset links fashion images to weekly sales, so sales are simulated for
5,000 products (500 for each of 10 article types) over 156 weeks:

```
sales = base level x season x holidays x price effect x style trend x noise
```

- seasonality follows each product's real `season` label; holiday bumps; promotions with
  a price elasticity of -1.5; gamma-Poisson noise (overdispersed counts)
- **style trend:** products are grouped into style groups (k-means on ResNet18
  embeddings, within each category), and each group gets a random yearly growth or
  decline. Its size is a parameter (`--style-effect`), and **0 switches it off** (null
  experiment).

The images are not redistributed: download them yourself (see below).

### Important limitation

The simulated style groups come from the same embeddings that the nearest-neighbour search
uses. A visual gain here shows that the pipeline can exploit visual similarity **when demand
really depends on it**. It does not show that real demand follows ResNet similarity.

## Method

1. **Embeddings:** ResNet18 (ImageNet weights), 512-d vector per image.
2. **Style groups:** PCA (20 components), then k-means per category (k=4 defines the
   simulated styles; a different clustering, k=6 with another seed, is what the model sees).
3. **Forecast unit:** category x style group = 40 weekly series (category totals hide the
   style trends, because groups growing and declining cancel out).
4. **Models:** seasonal naive baseline, then one global XGBoost with quantile objectives
   (10th, 50th, 67th and 90th percentiles) on lags, rolling means, calendar, price and
   promo features.
5. **Evaluation:** rolling-origin backtest (13 folds, train on the first 104 weeks and
   up, 4-week horizon), never a random split. Metrics: MAE, RMSE, and a weighted error
   where a stockout costs 2x an overstock. The cost-optimal forecast for that ratio is the
   2/3 quantile, hence the q67 model.
6. **Controls:** every visual variant is compared with a `base+series_id` control (the
   visual features are constant per series, so a tree can use them as an identifier), and
   with a null run where the style effect is 0.

## Results

Mean over folds, with the standard deviation across folds in brackets.

### 1. Products with sales history (40 series, 13 folds)

| Model | MAE, style effect 0.3 | MAE, null run (0.0) |
|---|---|---|
| Seasonal naive | 38.2 (7.0) | 24.6 (4.9) |
| XGBoost, base | 18.2 (3.6) | 14.9 (1.8) |
| XGBoost, base + series id (control) | 18.4 (3.6) | 14.6 (1.6) |
| XGBoost, base + visual | 18.1 (3.7) | 14.7 (1.8) |

- XGBoost roughly halves the error of seasonal naive.
- **Visual features add nothing measurable** when history is available: the paired
  difference against `base` is -0.07 MAE with a fold std of 0.91. The trend is already in
  the series' own lags.
- Forecasting the 67th percentile lowers the weighted error from 28.5 (median) to 25.2.
- 10-90 interval coverage is 0.69 (target 0.80): the intervals are too narrow.

### 2. New products without history (cold start)

20% of products are held out and never seen. A new product borrows recent sales from
other products. Series-level WAPE, 3 random holdouts x 13 folds = 39 runs:

| Borrows from | WAPE, style effect 0.3 | WAPE, null run |
|---|---|---|
| all products of the category | 37.3% | 16.5% |
| 20 random products of the category (control) | 37.4% | 16.4% |
| **20 visually nearest products** | **24.2%** | 16.3% |
| products of the true style group (upper bound) | 18.9% | 16.6% |

- Visual neighbours fall in the true style group 74-75% of the time (random: 31%).
- `visual_knn` beats the category average by 13.1 points in 39 of 39 runs, and beats the
  random-neighbour control by 13.2 points. The gain is not just averaging fewer products.
- In the null run, all four variants are within 0.3 points: the pipeline does not invent a
  visual effect.

**Summary:** visual features do not help when history exists, and cut the error by about a
third for products with no history.

## MLOps layer

| Part | Status | Where |
|---|---|---|
| Data validation (pandera): schema, ranges, no missing labels, consecutive weeks, no duplicates; a failed check stops the pipeline (exit code 1) | done | `src/data/validate.py` |
| Experiment tracking (MLflow): parameters, metrics per fold, git commit, content hash of each input file, model artefacts | done | `src/tracking.py` |
| Model registry with a promotion rule | done | `src/models/promote.py`, `src/models/rules.py` |
| Unit tests (data checks, promotion rule) | done | `tests/` |
| CI (GitHub Actions): lint, tests, validation, smoke training | done | `.github/workflows/ci.yml` |
| Weekly scheduled retrain | done (synthetic data, see below) | `.github/workflows/retrain.yml` |
| Conformal calibration of the intervals | planned | |
| FastAPI service + Docker | planned | |
| Drift monitoring (Evidently / KS test) | planned | |
| DVC data versioning | planned | |

### Promotion rule

A candidate is trained on all weeks except the **last 12**, which are scoring-only for
every model. The production model is re-scored on the same window, and the candidate is
promoted to `production` only if it lowers the q67 weighted error by at least 1%.
A leakage guard refuses the comparison (exit code 2) if the production model was trained on
weeks inside the holdout.

| Run | Candidate | Production | Gain | Decision |
|---|---|---|---|---|
| 1 | depth-1 model (weak) | none | n/a | promote (first model) |
| 2 | default model | depth-1 model | +51.6% | promote |
| 3 | same default model | default model | 0.0% | keep production |
| 4 | depth-1 model | default model | -106.4% | keep production |
| 5 | holdout moved earlier | default model | n/a | refused (leakage) |

The rule is simple and explicit, not a significance test: the window is 12 weeks x 40
series = 480 rows.

### About CI

The dataset cannot be redistributed, so `data/` is not in git. CI and the weekly retrain
therefore run on a **small synthetic fixture** (`src/data/fixture.py`: fake products,
random embeddings, simulated sales). They prove that the automation works, not that the
real model is good. The numbers above come from the real images and the simulation run
locally. Retraining on real data needs the files in a remote store (DVC, planned).

## Reproduce

Python 3.11. No GPU needed.

```bash
python -m pip install -r requirements-ci.txt
python -m pip install torch torchvision matplotlib      # only for embeddings and cluster grids

# 1. download the dataset from Kaggle, extract images/ and styles.csv into data/raw/
python -m src.data.load                                  # 5,000-product subset
python -m src.features.embed                             # ResNet18 embeddings
python -m src.features.cluster --k 4                     # styles that drive the simulation
python -m src.features.cluster --k 6 --seed 7 --out clusters_model.csv   # clusters the model sees

python -m src.data.simulate --style-col cluster --style-effect 0.3
python -m src.data.simulate --style-col cluster --style-effect 0.0      # null run
python -m src.features.build --tag cluster_style0.3 --model-clusters clusters_model.csv
python -m src.data.validate --stage all --tag cluster_style0.3

python -m src.models.baseline cluster_style0.3 series    # seasonal naive
python -m src.models.xgb --tag cluster_style0.3          # ablation (needs several minutes)
python -m src.models.coldstart --tag cluster_style0.3 --n-splits 3
python -m src.models.promote                             # registry + promotion rule

python -m pytest -q
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`reports/clusters_sim/` holds the k=4 style grids that define the simulation, and
`reports/clusters/` the k=6 grids the model sees.

## Repository layout

```
src/
  data/        load, simulate, validate, fixture
  features/    embed, cluster, build
  models/      evaluate, baseline, xgb, coldstart, rules, promote
  tracking.py  MLflow helper
tests/         validation and promotion-rule tests
.github/workflows/   ci.yml, retrain.yml
```

## Known limitations

- Sales are simulated, so the results show what the pipeline can recover, not how real
  demand behaves.
- Clusters from a pretrained CNN partly encode how a photo was taken (on a model, flat,
  small in the frame) and colour, not only garment style.
- Cluster ids (`vis_share_c0`, ...) mean different things in different categories.
- 10-90 interval coverage is 0.66-0.70 instead of 0.80; calibration is planned.
- Folds of a rolling-origin backtest overlap, so "better in 9 of 13 folds" is
  indicative only; I report mean differences and their spread.

## Credits

Images and labels: Param Aggarwal, *Fashion Product Images (Small)*, Kaggle (MIT licence).