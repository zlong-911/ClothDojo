# ClothDojo website preview

Website-only snapshot of the local project page, including 24 full trajectory examples and the selected presentation films. The project code on `master` is unchanged.

## Preview locally

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory website
```

## GitHub Pages

In repository **Settings → Pages**, choose **Deploy from a branch**, select **codex/website-preview-20261030** and **/(root)**, then save. The root entry redirects to `website/`. `.nojekyll` keeps this a plain static deployment.

Pages is not yet enabled as of this upload. Private-repository Pages availability depends on the account plan; the repository has not been made public. Published Pages may be publicly accessible even when the source repository is private.

Only the website is present in this branch's current tree. No project datasets, model weights, credentials, or server configuration are included. Video-source paths in metadata document provenance; they are not downloadable trajectory datasets.

Source assets retain their original licenses, including applicable GenieSimAssets CC BY-NC-SA 4.0 conditions. See the project license notes in [paper/README.md on master](https://github.com/zlong-911/ClothDojo/blob/master/paper/README.md) before redistributing assets.

See [website/README.md](website/README.md) for media scope and validation limitations.
