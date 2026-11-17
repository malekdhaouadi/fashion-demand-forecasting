# Fashion demand forecasting from product images

![CI](https://github.com/malekdhaouadi/fashion-demand-forecasting/actions/workflows/ci.yml/badge.svg)

Fashion shops must decide months ahead what to buy. Too much stock means discounts,
too little means lost sales. This project asks whether the look of a product helps
predict its demand, and whether the system around the forecast can be kept reliable.

## Data: what is real, what is simulated

The product images and their labels (category, colour, season) are real. They come from
the public Kaggle dataset *Fashion Product Images (Small)*, which is not redistributed here.
No public dataset links fashion images to weekly sales, so the sales are simulated, for
5,000 products over three years, with seasons, holidays, promotions, noise and a
style trend that can be switched on or off.

Important limitation: the simulated trends follow groups of visually similar products.
So the results show that the pipeline can use visual similarity when demand depends on
it. They do not prove that real demand follows it.

## How it works

Each image is turned into a numeric description by a pretrained neural network (ResNet18).
Products of the same category are grouped by visual similarity into style groups. Each
category and style group is forecast week by week, and the forecast is a range rather
than a single number. Models are always tested on weeks that come after the ones they
learned from, never on a random split.

## Results

When a product has sales history, a gradient boosting model roughly halves the error of
the simple "same week last year" baseline. Visual features add nothing measurable in
that case, because the trend is already visible in the past sales.

For new products with no history, the picture changes. A new product borrows the
recent sales of the 20 products that look most like it. This lowers the forecast error
from 37% to 24%, over 39 runs, and the gain disappears when the style trend is switched
off. A control that borrows from 20 random products gives no gain, so the improvement
comes from visual similarity and not from averaging.

In short: images do not help when history exists, and they help a lot when it does not.

Forecasting the 67th percentile instead of the median lowers the business cost, since
a missed sale is treated as twice as costly as an unsold item. One weakness: the
80% forecast ranges contain the true value only about 70% of the time, so they are
too narrow. Calibrating them is planned.

## Reliability

- **Data checks:** the data is validated before training. A failed check stops the
  pipeline.
- **Experiment tracking:** every run is logged with its parameters, metrics, code version
  and a fingerprint of its input data.
- **Model registry:** a new model replaces the live one only if it lowers the business
  error by at least 1% on the last 12 weeks, which no model has trained on. If the live
  model has seen those weeks, the comparison is refused.
- **Tests and automation:** unit tests and a GitHub Actions workflow run on every push.
  They run on a small synthetic dataset, because the real data cannot be shared. They show
  that the automation works, not that the real model is good. A weekly retrain is
  scheduled the same way.
- **API:** a small FastAPI service returns the forecast range for a category and a week,
  up to four weeks beyond the data.

Not done yet: Docker packaging, drift monitoring, data versioning, calibrated ranges.

## Run it

Python 3.11, no GPU needed.

```bash
python -m pip install -r requirements-ci.txt
python -m pip install torch torchvision matplotlib

python -m src.data.load
python -m src.features.embed
python -m src.features.cluster --k 4
python -m src.features.cluster --k 6 --seed 7 --out clusters_model.csv
python -m src.data.simulate --style-col cluster --style-effect 0.3
python -m src.features.build --tag cluster_style0.3 --model-clusters clusters_model.csv
python -m src.data.validate --stage all --tag cluster_style0.3
python -m src.models.xgb --tag cluster_style0.3
python -m src.models.coldstart --tag cluster_style0.3 --n-splits 3
python -m src.models.promote
python -m src.serving.export --tag cluster_style0.3
python -m uvicorn src.serving.app:app --port 8000
```

Tests: `python -m pytest -q`. Experiment viewer: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Limitations

- Sales are simulated.
- The image groups partly reflect how a photo was taken (on a model, flat, small in the
  frame) and its colour, not only the style of the garment.
- Results come from one simulation setup.

## Credits

Images and labels: Param Aggarwal, *Fashion Product Images (Small)*, Kaggle, MIT licence.