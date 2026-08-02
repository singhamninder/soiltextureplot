import matplotlib

matplotlib.use("Agg")

from pathlib import Path

from typer.testing import CliRunner

from soiltextureplot.cli import app

runner = CliRunner()


def _write_sample_csv(path: Path, *, include_texture_cols: bool = True) -> Path:
    if include_texture_cols:
        path.write_text(
            "sample_id,sand,silt,clay,BD\n"
            "S1,65,20,15,1.35\n"
            "S2,70,24,6,1.42\n"
            "S3,35,45,20,1.20\n",
            encoding="utf-8",
        )
    else:
        path.write_text(
            "sample_id,BD\nS1,1.35\nS2,1.42\n",
            encoding="utf-8",
        )
    return path


def test_list_systems() -> None:
    result = runner.invoke(app, ["list-systems"])
    assert result.exit_code == 0
    assert "USDA" in result.stdout
    assert "HYPRES" in result.stdout


def test_classify_writes_texture_class(tmp_path: Path) -> None:
    input_csv = _write_sample_csv(tmp_path / "in.csv")
    output_csv = tmp_path / "out.csv"
    result = runner.invoke(app, ["classify", str(input_csv), "-o", str(output_csv)])
    assert result.exit_code == 0
    content = output_csv.read_text(encoding="utf-8")
    assert "texture_class" in content
    assert content.count("\n") >= 4


def test_classify_stdout(tmp_path: Path) -> None:
    input_csv = _write_sample_csv(tmp_path / "in.csv")
    result = runner.invoke(app, ["classify", str(input_csv)])
    assert result.exit_code == 0
    assert "texture_class" in result.stdout


def test_plot_writes_image(tmp_path: Path) -> None:
    input_csv = _write_sample_csv(tmp_path / "in.csv")
    output_png = tmp_path / "triangle.png"
    result = runner.invoke(
        app,
        ["plot", str(input_csv), "-o", str(output_png), "--size-by", "BD"],
    )
    assert result.exit_code == 0
    assert output_png.is_file()
    assert output_png.stat().st_size > 0


def test_run_writes_csv_and_figure(tmp_path: Path) -> None:
    input_csv = _write_sample_csv(tmp_path / "in.csv")
    output_csv = tmp_path / "classified.csv"
    output_png = tmp_path / "triangle.png"
    result = runner.invoke(
        app,
        [
            "run",
            str(input_csv),
            "-o",
            str(output_csv),
            "-f",
            str(output_png),
            "--system",
            "HYPRES",
        ],
    )
    assert result.exit_code == 0
    assert "texture_class" in output_csv.read_text(encoding="utf-8")
    assert output_png.is_file()
    assert output_png.stat().st_size > 0


def test_unknown_system_exits_nonzero(tmp_path: Path) -> None:
    input_csv = _write_sample_csv(tmp_path / "in.csv")
    result = runner.invoke(
        app, ["classify", str(input_csv), "--system", "NOT_A_SYSTEM"]
    )
    assert result.exit_code != 0
    assert (
        "Unknown texture system" in result.stderr
        or "Unknown texture system" in result.stdout
    )


def test_missing_columns_exits_nonzero(tmp_path: Path) -> None:
    input_csv = _write_sample_csv(tmp_path / "bad.csv", include_texture_cols=False)
    result = runner.invoke(app, ["classify", str(input_csv)])
    assert result.exit_code != 0
