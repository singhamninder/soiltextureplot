# CLI Reference

`soiltextureplot` ships a command-line interface for classifying CSV samples and saving ternary texture plots.

## Commands

| Command | Purpose |
|---------|---------|
| `list-systems` | List registered texture systems |
| `classify` | Classify a CSV and write `texture_class` |
| `plot` | Plot samples and save a figure |
| `run` | Classify to CSV and save a figure in one step |

## Shared options

These options apply to `classify`, `plot`, and `run`:

| Option | Default | Description |
|--------|---------|-------------|
| `INPUT` | (required) | Path to the input CSV |
| `--system` | `USDA` | Texture system (`USDA` or `HYPRES`) |
| `--sand-col` | `sand` | Sand percentage column |
| `--silt-col` | `silt` | Silt percentage column |
| `--clay-col` | `clay` | Clay percentage column |

## classify

Write a classified CSV. Omit `--output` / `-o` (or pass `-`) to print to stdout.

```bash
soiltextureplot classify data.csv -o classified.csv
soiltextureplot classify data.csv --system HYPRES
```

## plot

Save a ternary figure. Default output is `texture_triangle.png`.

| Option | Default | Description |
|--------|---------|-------------|
| `--output` / `-o` | `texture_triangle.png` | Figure path (format from suffix) |
| `--size-by` | unset | Column for point size scaling |
| `--cmap` | `Set3_r` | Colormap for texture classes |
| `--color-points` | `black` | Sample point color |
| `--show-labels` / `--no-show-labels` | show labels | Label points with `sample_id` when present |
| `--dpi` | `150` | Figure DPI |

```bash
soiltextureplot plot data.csv --size-by BD -o triangle.png
```

## run

Classify and plot together.

| Option | Default | Description |
|--------|---------|-------------|
| `--output` / `-o` | `classified.csv` | Classified CSV path |
| `--figure` / `-f` | `texture_triangle.png` | Figure path |
| (plus all plot styling options above) | | |

```bash
soiltextureplot run data.csv -o classified.csv -f triangle.png --system HYPRES
```

## list-systems

```bash
soiltextureplot list-systems
```

## Module entry point

```bash
python -m soiltextureplot list-systems
```
