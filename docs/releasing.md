# Releasing Guide (Maintainers)

This guide covers trusted publishing setup and tag-based release workflows for TestPyPI and PyPI.

## Trusted Publishing Prerequisites

Before pushing release tags, configure trusted publishing:

1. In GitHub, create environments named `pypi` and `testpypi` under repository settings.
2. In PyPI project settings, add a trusted publisher matching this repository and the `publish-pypi.yml` workflow.
3. In TestPyPI project settings, add a trusted publisher matching this repository and the `publish-testpypi.yml` workflow.

## Release Automation Summary

Publishing is automated via tag-triggered workflows:

- `v*rc*` tags publish to TestPyPI
- `v*.*.*` tags are evaluated for PyPI; tags containing `rc` are skipped by workflow condition

Both workflows verify that the git tag version matches `uv version --short` before publishing.
Both workflows also run isolated wheel and source-distribution smoke tests (`import soiltextureplot`) before upload.

## Prerelease to TestPyPI

```bash
# Example: bump to prerelease version
uv version 0.1.2rc1
git add pyproject.toml uv.lock
git commit -m "Bump version to 0.1.2rc1"
git push origin dev

# After merge to main, create and push prerelease tag from main
git checkout main
git pull
git tag -a v0.1.2rc1 -m "Release v0.1.2rc1"
git push origin v0.1.2rc1
```

Smoke test from TestPyPI:

```bash
uv run --with "soiltextureplot==0.1.2rc1" --no-project -- python -c "import soiltextureplot"
```

## Stable Release to PyPI

```bash
# Example: bump stable version
uv version --bump patch
git add pyproject.toml uv.lock
git commit -m "Bump version"
git push origin dev

# After merge to main, create and push stable tag from main
git checkout main
git pull
git tag -a v0.1.2 -m "Release v0.1.2"
git push origin v0.1.2
```

Smoke test from PyPI:

```bash
uv run --with soiltextureplot --no-project -- python -c "import soiltextureplot"
```
