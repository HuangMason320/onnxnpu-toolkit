# ONNXNPU Toolkit

[![PyPI version]][pypi-url] [![License]][license-url]

`onnxnpu` is a lightweight CLI toolkit that **inspects, optimizes, and modifies ONNX models *before* you convert or deploy them to NPUs**.

> "Catch unsupported operators early before they derail your model."
> — *Mason Huang*

## ✨ Features (v0.1 — released)

| Feature               | Description                                                                                                           |
| --------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Operator scan**     | Fast, dependency‑free static analysis of `.onnx` files                                                                 |
| **Hardware profiles** | Built‑in JSON compatibility tables for common NPUs (KL520 / 530 / 630 / 720 / 730 …) with an override mechanism        |
| **Clear report**      | CLI table ＋ optional JSON / Markdown export; highlights unsupported ops and optional‑feature gaps                      |
| **Actionable hints**  | Suggestions and links to official docs for each unsupported operator                                                  |
| **Opset update**      | Upgrade model opset (12 – 18) to match target hardware                                                                |

---

<details>
<summary>🧭 Roadmap</summary>

| Version | Target Date* | Major Items                                                                                           | Notes / Dependencies                                                    |
| ------- | ------------ | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| **0.2 – Validation & Reporting** | May 2025 | • Shape checker enforcing 4‑D `(1, C, H, W)` constraint<br>• Rich Markdown / JSON report templates for CI badges | Uses ONNX shape‑inference to avoid manual parsing                       |
| **0.3 – Graph Simplification & Slimming** | Jun 2025 | • Integrate **onnx‑sim** (`--simplify`)<br>• Model slimming (`--prune`, `--quantize`)<br>• Bundle Kneron **optimizer_scripts** (BN‑Conv fuse, Dropout removal …) | Requires onnx‑sim ≥ 0.4; quantization via ONNX QOps                     |
| **0.4 – Automatic Op Replacement** | Jul 2025 | • `--replace` mapping table (e.g., `Reshape → Flatten`)<br>• Fallback to custom kernels / plugin stubs           | Needs rule set ＋ regression tests                                       |
| **0.5 – Interactive Viewer**      | Aug 2025 | • `onnxnpu view` drag‑and‑drop web UI<br>• Highlight unsupported nodes directly on the graph<br>• Downloadable HTML report | Likely React + ONNX‑JS; demo hosted on GitHub Pages                     |
| **0.6 – Extensibility & Ecosystem** | Sep 2025 | • Plugin system via Python entry‑points<br>• Community hardware‑profile submission flow<br>• Freeze stable API v1.0 | Plan to publish on conda‑forge after API stabilisation                  |

\* Dates are tentative and may shift based on resources.
</details>

## 🚀 Quick start

You can use two different CLI commands: `onnxnpu` or `onpu` to check, optimize, and modify your ONNX models for NPU deployment. Both commands provide identical functionality with the same syntax.

| Command                                       | Description                                           |
|-----------------------------------------------|-------------------------------------------------------|
| `pip install onnxnpu`                         | Install latest package from PyPI                      |
| `onnxnpu check my_model.onnx -p KL720`           | Check `my_model.onnx` for the KL720 hardware profile  |
| `onnxnpu check my_model.onnx`                    | Check `my_model.onnx` for all built-in profiles       |
| `onnxnpu opt my_model.onnx --opset 13`           | Update model to opset 13                              |
| `onnxnpu -V`, `onnxnpu --version`                   | Show version number and exit                          |

### Sample output

```
Model summary – my_model.onnx
IR version : 6    Opset : 11
Inputs  : input  float32  [1, 3, 112, 112]
Outputs : output  float32  [1, 512]
Dynamic axes : None detected ✓

my_model.onnx · IR 6 · KL520
╭──────────────────────────────────────────────────────────────╮
│  Status  Operator   Count   Notes                            │
├──────────────────────────────────────────────────────────────┤
│   ✓      Conv        27                                      │
│   ✓      Relu        27                                      │
│   ✗      Elu          5     Not supported on KL520           │
│   ✓      MaxPool      5                                      │
│   ✗      Resize       2     Only linear/nearest modes OK     │
╰──────────────────────────────────────────────────────────────╯
⚠  2 unsupported operator(s) detected.
```

<!-- ## 🛠️ Development & Publishing

透過 Poetry 一鍵建置與發佈：

```bash
# 1. build wheel and sdist
poetry build

# 2. （Test version）publish to TestPyPI
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry config pypi-token.testpypi <YOUR_TEST_PYPI_TOKEN>
poetry publish -r testpypi

# 3. Publish official version to PyPI
poetry config pypi-token.pypi <YOUR_PYPI_TOKEN>
poetry publish
``` -->

## 🧑‍💻 API usage

```python
from onnxnpu import Checker, load_profile

checker = Checker("my_model.onnx", profile=load_profile("kl720"))
report = checker.run()
print(report.to_markdown())

# Update opset version
from onnxnpu import update_opset_version
update_opset_version("my_model.onnx", target_version=13)
```

## 📖 Hardware profiles

Profiles live under `onnxnpu/profiles/*.json`.
Each profile declares the operators, attributes, and constraints supported by a particular accelerator.
See [`docs/PROFILE_SCHEMA.md`](docs/PROFILE_SCHEMA.md) for the JSON schema.

Contributions for new hardware are very welcome!

<!-- ## 🤝 Contributing

We love pull requests! Please read `CONTRIBUTING.md` and open an issue before you start a large refactor so we can align on design.

Coding conventions follow **PEP 8** with the Black formatter. -->

### A note on language

The primary language of this README is **English** for wider community reach.  A Traditional Chinese translation will be added soon.


[PyPI version]: https://img.shields.io/pypi/v/onnxnpu
[pypi-url]: https://pypi.org/project/onnxnpu
[Build status]: https://img.shields.io/github/actions/workflow/status/HuangMason320/onnx-checker/ci.yml?branch=main
[ci-url]: https://github.com/HuangMason320/onnxnpu-toolkit/actions
[License]: https://img.shields.io/github/license/HuangMason320/onnxnpu-toolkit
[license-url]: https://pypi.org/project/onnxnpu-toolkit/
