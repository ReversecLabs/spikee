# GitHub Workflows

## Contributions

Run **linting** and **functional test** checks against PRs to `develop`.

## Release Steps

1. Run **Prepare Release** (`release.yaml`) from `main`, choosing a `major`, `minor`, or `patch` bump. It runs checks unless `skip_tests` is selected, fast-forwards `develop`, updates the version files and changelog, then pushes a `release/vX.Y.Z` branch. The workflow log prints the compare URL, title, and body for a maintainer to open the pull request manually.

2. Open the printed compare URL, create the pull request from `release/vX.Y.Z` into `main`, and merge it after review.

3. Merging the release PR triggers **Publish Reviewed Release** (`publish-release.yaml`). It publishes the exact release PR merge commit to PyPI and tags it. A separate synchronization job merges that release into `develop` and sets the next patch development version, such as `1.2.4-dev` after `1.2.3`.
## Retrying development synchronization

If publication and tagging succeed but synchronization fails, rerun only the
failed jobs. Alternatively, run **Synchronize Published Release** from `main`
with the published tag (for example `v0.9.3`). This workflow never builds or
publishes a package. It verifies that the release commit is on `main` and matches
its version tag before updating `develop`.

Synchronization handles version-only conflicts after a merge, rebase, or squash
of the release PR. It preserves other development edits and newer development
versions. Repeating a completed synchronization makes no commit. Other merge
conflicts stop the job for review; a concurrent update to `develop` rejects the
push rather than overwriting it.
