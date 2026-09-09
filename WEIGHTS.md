# Weight handoff

Weights are not committed to Git. The primary release checkpoint is:

```text
runs/cifar10-generation-only-lowfreq-ot-ce-5k/model-1000000.pt
```

It is approximately 133 MB. Upload that file to a cloud drive, then download
it into `weights/model-1000000.pt` (or pass its absolute path to the commands
below). The frozen reference classifier used for FID/accuracy is:

```text
runs/reference-cifar10-classifier/cifar10-classifier-best.pt
```

That classifier is also excluded from Git and is only needed for the external
generation-accuracy and quality evaluation commands.

| artifact | bytes | SHA256 |
| --- | ---: | --- |
| `model-1000000.pt` | 132856063 | `fbd3a87ef31f3daf0aa862386a61d874129e18bf5bc79b82cf6a7d8b58ba682b` |
| `cifar10-classifier-best.pt` | 11288939 | `cf98ff68b904a1720e4fa7eb084680a7c714c2b6ec50b0b70753443c26e76fcb` |

## Current download location

The 1M model is available from Quark Drive:

<https://pan.quark.cn/s/67b5254c9276>

Download the file named `model-1000000.pt`, verify the SHA256 above, and place
it at `weights/model-1000000.pt`. The reference classifier is not included in
that link; upload/download it separately if generation accuracy or FID is to be
reproduced.

For a public release, record the cloud URL, SHA256 checksum, and license here
after uploading. Generate the checksum with:

```bash
sha256sum weights/model-1000000.pt
```
