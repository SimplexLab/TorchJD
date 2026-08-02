---
title: TorchJD Interactive Plotter
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.21.0
app_file: app.py
pinned: false
license: mit
---

# TorchJD Interactive Plotter

Interactive visualization of gradient aggregation methods from [TorchJD](https://torchjd.org).

Adjust the angle and length of each gradient vector and select aggregators to see how they combine
the gradients. The green region shows the dual cone — any descent direction must lie inside it.

## URL parameters

The app accepts query parameters so you can link to a specific configuration or embed it in
documentation with an aggregator pre-selected:

| Parameter | Format | Example |
|-----------|--------|---------|
| `agg` | Comma-separated aggregator names | `?agg=Mean,MGDA` |
| `g1`, `g2`, `g3` | `angle_radians,length` | `?g1=1.5708,2.0` |
| `seed` | Integer | `?seed=42` |
