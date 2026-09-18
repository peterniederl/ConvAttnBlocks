import argparse
import json
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt


def load_runs(results_dir, selected):
    root = Path(results_dir)
    names = selected or sorted(path.name for path in root.iterdir() if path.is_dir())
    runs = {}
    for name in names:
        run_dir = root / name
        history_path = run_dir / "history.json"
        evaluation_path = run_dir / "evaluation.json"
        settings_path = run_dir / "settings.json"
        if history_path.exists() and evaluation_path.exists():
            payload = {
                "history": json.loads(history_path.read_text()),
                "evaluation": json.loads(evaluation_path.read_text()),
            }
            if settings_path.exists():
                payload["settings"] = json.loads(settings_path.read_text())
            runs[name] = payload
    if not runs:
        raise FileNotFoundError("No completed runs with history.json and evaluation.json were found.")
    return runs


def _model_name_for_run(run_name):
    if "_run_" not in run_name:
        return run_name
    return run_name.rsplit("_run_", 1)[0]


def _best_run_by_model(runs):
    best_by_model = {}
    for run_name, run in runs.items():
        model_name = _model_name_for_run(run_name)
        history = run["history"]
        val_accuracy = history.get("val_accuracy") or []
        current_best = best_by_model.get(model_name)
        score = max(val_accuracy) if val_accuracy else float("-inf")
        if current_best is None or score > current_best["score"]:
            best_by_model[model_name] = {"run_name": run_name, "score": score, "run": run}
    return {model_name: info["run"] for model_name, info in best_by_model.items()}


def _run_summary_rows(runs):
    rows = []
    for run_name in sorted(runs):
        run = runs[run_name]
        history = run["history"]
        evaluation = run["evaluation"]
        val_accuracy = history.get("val_accuracy") or []
        val_loss = history.get("val_loss") or []
        best_index = max(range(len(val_accuracy)), key=lambda idx: val_accuracy[idx]) if val_accuracy else None
        rows.append({
            "model": _model_name_for_run(run_name),
            "run": run_name,
            "seed": run.get("settings", {}).get("seed", "-"),
            "best_val_accuracy": round(val_accuracy[best_index], 4) if best_index is not None else None,
            "best_val_loss": round(val_loss[best_index], 4) if best_index is not None else None,
            "final_val_accuracy": round(val_accuracy[-1], 4) if val_accuracy else None,
            "final_val_loss": round(val_loss[-1], 4) if val_loss else None,
            "evaluation_accuracy": round(evaluation.get("accuracy", 0), 4),
            "evaluation_loss": round(evaluation.get("loss", 0), 4),
        })
    return rows


def _plotly_series_for_best_runs(best_runs, metric, last_n=None):
    trace_data = []
    for model_name in sorted(best_runs):
        run = best_runs[model_name]
        values = run["history"].get(metric) or []
        if not values:
            continue
        if last_n is not None:
            values = values[-last_n:]
            start_epoch = max(len(run["history"].get(metric) or []) - last_n + 1, 1)
            x_values = list(range(start_epoch, start_epoch + len(values)))
        else:
            x_values = list(range(1, len(values) + 1))
        trace_data.append({
            "type": "scatter",
            "mode": "lines",
            "name": model_name,
            "x": x_values,
            "y": [float(v) for v in values],
            "line": {"width": 2.5},
        })
    return trace_data


def _average_model_series(runs, metric, last_n=None):
    by_model = {}
    for run_name, run in runs.items():
        model_name = _model_name_for_run(run_name)
        values = run["history"].get(metric) or []
        if not values:
            continue
        by_model.setdefault(model_name, []).append(values)

    trace_data = []
    for model_name in sorted(by_model):
        values_by_run = by_model[model_name]
        if last_n is not None:
            values_by_run = [values[-last_n:] for values in values_by_run]
            max_len = max(len(values) for values in values_by_run)
            start_epoch = max(1, len(values_by_run[0]) - last_n + 1) if values_by_run else 1
            x_values = list(range(start_epoch, start_epoch + max_len))
            averaged = []
            for offset in range(max_len):
                sample = [
                    values[offset]
                    for values in values_by_run
                    if len(values) > offset
                ]
                if sample:
                    averaged.append(sum(sample) / len(sample))
            x_values = x_values[:len(averaged)]
        else:
            max_len = max(len(values) for values in values_by_run)
            x_values = list(range(1, max_len + 1))
            averaged = []
            for epoch_idx in range(max_len):
                sample = [
                    values[epoch_idx]
                    for values in values_by_run
                    if len(values) > epoch_idx
                ]
                if sample:
                    averaged.append(sum(sample) / len(sample))
        if not averaged:
            continue
        trace_data.append({
            "type": "scatter",
            "mode": "lines",
            "name": model_name,
            "x": x_values,
            "y": [float(v) for v in averaged],
            "line": {"width": 2.5},
        })
    return trace_data


def _write_html_report(runs, output_path):
    best_runs = _best_run_by_model(runs)
    rows = _run_summary_rows(runs)

    table_rows_html = []
    for row in rows:
        rendered = "".join(
            f'<td data-key="{key}">{row.get(key, "-")}</td>'
            for key in [
                "model",
                "run",
                "seed",
                "best_val_accuracy",
                "best_val_loss",
                "final_val_accuracy",
                "final_val_loss",
                "evaluation_accuracy",
                "evaluation_loss",
            ]
        )
        table_rows_html.append(f"<tr>{rendered}</tr>")

    accuracy_series = _plotly_series_for_best_runs(best_runs, "val_accuracy")
    loss_series = _plotly_series_for_best_runs(best_runs, "val_loss")
    recent_accuracy_series = _plotly_series_for_best_runs(best_runs, "val_accuracy", last_n=25)
    average_recent_accuracy_series = _average_model_series(runs, "val_accuracy", last_n=20)

    header_pairs = [
        ("model", "Model"),
        ("run", "Run"),
        ("seed", "Seed"),
        ("best_val_accuracy", "Best val accuracy"),
        ("best_val_loss", "Best val loss"),
        ("final_val_accuracy", "Final val accuracy"),
        ("final_val_loss", "Final val loss"),
        ("evaluation_accuracy", "Evaluation accuracy"),
        ("evaluation_loss", "Evaluation loss"),
    ]
    header_html = "".join(
        f'<th data-key="{key}">{label}<span class="sort-indicator"></span></th>'
        for key, label in header_pairs
    )

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Experiment Results</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background: #f5f7fb;
      font-family: Arial, sans-serif;
      color: #1f2937;
    }}
    .container {{
      max-width: 1600px;
      margin: 0 auto;
    }}
    .chart-card, .table-card {{
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 12px rgba(15, 23, 42, 0.08);
      padding: 14px 14px 10px;
      margin-bottom: 20px;
    }}
    .chart {{
      width: 100%;
      height: 440px;
    }}
    h2 {{
      margin: 0 0 12px 0;
      font-size: 1.25rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      border: 1px solid #dfe3eb;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef4ff;
      font-weight: 600;
      cursor: pointer;
      user-select: none;
    }}
    tbody tr:nth-child(even) {{
      background: #fafbff;
    }}
    .sort-indicator {{
      display: inline-block;
      width: 10px;
      margin-left: 6px;
      color: #4b5563;
    }}
    @media (max-width: 900px) {{
      body {{ padding: 12px; }}
      th, td {{ padding: 6px 8px; }}
      .chart {{ height: 360px; }}
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"chart-card\">
      <h2>Validation accuracy</h2>
      <div id=\"chart-accuracy\" class=\"chart\"></div>
    </div>
    <div class=\"chart-card\">
      <h2>Validation loss</h2>
      <div id=\"chart-loss\" class=\"chart\"></div>
    </div>
    <div class=\"chart-card\">
      <h2>Validation accuracy in the last 25 epochs</h2>
      <div id=\"chart-last25-accuracy\" class=\"chart\"></div>
    </div>
    <div class=\"chart-card\">
      <h2>Average validation accuracy over all runs (last 20 epochs)</h2>
      <div id=\"chart-average-last20-accuracy\" class=\"chart\"></div>
    </div>
    <div class=\"table-card\">
      <h2>All run results</h2>
      <table id=\"results-table\">
        <thead>
          <tr>{header_html}</tr>
        </thead>
        <tbody>
          {''.join(table_rows_html)}
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const accuracyData = {json.dumps(accuracy_series)};
    const lossData = {json.dumps(loss_series)};
    const recentAccuracyData = {json.dumps(recent_accuracy_series)};
    const averageRecentAccuracyData = {json.dumps(average_recent_accuracy_series)};

    const commonLayout = {{
      legend: {{
        orientation: 'h',
        x: 0.5,
        y: 1.12,
        xanchor: 'center',
        yanchor: 'bottom',
        font: {{size: 11}}
      }},
      hovermode: 'closest',
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      dragmode: 'zoom',
      modebar: {{orientation: 'v'}}
    }};

    const accuracyLayout = {{
      ...commonLayout,
      xaxis: {{title: 'Epoch'}},
      yaxis: {{title: 'Accuracy'}},
      margin: {{l: 60, r: 20, t: 10, b: 90}}
    }};

    const lossLayout = {{
      ...commonLayout,
      xaxis: {{title: 'Epoch'}},
      yaxis: {{title: 'Loss'}},
      margin: {{l: 60, r: 20, t: 10, b: 90}}
    }};

    const recentAccuracyLayout = {{
      ...commonLayout,
      xaxis: {{title: 'Epoch'}},
      yaxis: {{title: 'Accuracy'}},
      margin: {{l: 60, r: 20, t: 10, b: 90}}
    }};

    const averageRecentAccuracyLayout = {{
      ...commonLayout,
      xaxis: {{title: 'Epoch'}},
      yaxis: {{title: 'Average validation accuracy'}},
      margin: {{l: 70, r: 20, t: 10, b: 90}}
    }};

    const chartConfigs = [
      ['chart-accuracy', accuracyData, accuracyLayout],
      ['chart-loss', lossData, lossLayout],
      ['chart-last25-accuracy', recentAccuracyData, recentAccuracyLayout],
      ['chart-average-last20-accuracy', averageRecentAccuracyData, averageRecentAccuracyLayout]
    ];

    chartConfigs.forEach(([chartId, traces, layout]) => {{
      Plotly.newPlot(chartId, traces, layout, {{responsive: true, displaylogo: false, scrollZoom: true}});
      const graph = document.getElementById(chartId);
      graph.on('plotly_legendclick', function(event) {{
        const traceIndex = event.curveNumber;
        const traceStates = this.data.map((trace) => trace.visible);
        const isolate = traceStates[traceIndex] !== false && traceStates[traceIndex] !== 'legendonly';

        if (isolate) {{
          Plotly.restyle(this, 'visible', 'legendonly', Array.from({{length: this.data.length}}, (_, i) => i).filter((i) => i !== traceIndex));
          Plotly.restyle(this, 'visible', true, [traceIndex]);
        }} else {{
          Plotly.restyle(this, 'visible', true, Array.from({{length: this.data.length}}, (_, i) => i));
        }}
        return false;
      }});
    }});

    window.addEventListener('resize', () => {{
      chartConfigs.forEach(([chartId]) => Plotly.Plots.resize(chartId));
    }});

    const table = document.getElementById('results-table');
    const headers = table.querySelectorAll('th[data-key]');
    let currentSort = {{key: null, dir: 1}};

    function parseCellValue(value) {{
      if (value === '-' || value === '') return Number.NEGATIVE_INFINITY;
      const num = Number(value);
      return Number.isFinite(num) ? num : String(value).toLowerCase();
    }}

    function sortTable(key) {{
      const rows = [...table.querySelectorAll('tbody tr')];
      const dir = currentSort.key === key && currentSort.dir === 1 ? -1 : 1;
      currentSort = {{key, dir}};

      rows.sort((rowA, rowB) => {{
        const a = parseCellValue(rowA.querySelector(`td[data-key=\"${{key}}\"]`)?.textContent.trim() ?? '');
        const b = parseCellValue(rowB.querySelector(`td[data-key=\"${{key}}\"]`)?.textContent.trim() ?? '');
        if (a < b) return -1 * dir;
        if (a > b) return 1 * dir;
        return 0;
      }});

      rows.forEach(row => table.querySelector('tbody').appendChild(row));
      headers.forEach(header => {{
        const indicator = header.querySelector('.sort-indicator');
        if (header.dataset.key !== key) {{
          indicator.textContent = '';
          return;
        }}
        indicator.textContent = dir === 1 ? '▲' : '▼';
      }});
    }}

    headers.forEach(header => {{
      header.addEventListener('click', () => sortTable(header.dataset.key));
    }});
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def plot_runs(runs, output_path=None):
    if output_path is not None and Path(output_path).suffix.lower() == ".html":
        return _write_html_report(runs, Path(output_path))

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, run in runs.items():
        history = run["history"]
        epochs = range(1, len(history["accuracy"]) + 1)
        axes[0].plot(epochs, history["accuracy"], label=name)
        axes[0].plot(epochs, history["val_accuracy"], linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, history["loss"], label=name)
        axes[1].plot(epochs, history["val_loss"], linestyle="--", label=f"{name} val")
        axes[2].bar(name, run["evaluation"]["accuracy"])
    axes[0].set_title("Accuracy")
    axes[1].set_title("Loss")
    axes[2].set_title("Final evaluation accuracy")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    if output_path:
        figure.savefig(output_path, dpi=150)
    else:
        plt.show()
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize one, several, or all ResNet experiment runs.")
    parser.add_argument("runs", nargs="*", help="Run directory names. Omit to include every completed run.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", help="Optional HTML or image path. Defaults to results/comparison.html.")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(args.results_dir) / "comparison.html"
    output = plot_runs(load_runs(args.results_dir, args.runs), str(output_path))
    if output and output.suffix.lower() == ".html":
        print(f"HTML report written to {output}")
        if not args.output:
            webbrowser.open(output.resolve().as_uri())
