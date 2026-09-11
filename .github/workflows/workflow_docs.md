# GitHub Workflows

## Contributions

Run **linting** and **functional test** checks against PRs to `develop`.

## Release Steps

1. Run **Prepare Release** (`release.yaml`) from `main`, choosing a `major`, `minor`, or `patch` bump. It runs checks unless `skip_tests` is selected, fast-forwards `develop`, updates the version files and changelog, then pushes a `release/vX.Y.Z` branch. The workflow log prints the compare URL, title, and body for a maintainer to open the pull request manually.

2. Open the printed compare URL, create the pull request from `release/vX.Y.Z` into `main`, and merge it after review.

3. Merging the release PR triggers **Publish Reviewed Release** (`publish-release.yaml`). It tags and publishes the reviewed `main` commit to PyPI, then sets `develop` to the next patch development version, such as `1.2.4-dev` after `1.2.3`.