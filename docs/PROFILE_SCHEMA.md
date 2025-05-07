# Hardware Profile JSON Schema

This document defines the structure of the `*.json` files located in `onnx_op_check/profiles/`.  A **hardware profile** describes which ONNX operators (and which attribute combinations) are supported by a specific accelerator, NPU, or runtime.

> The schema is intentionally simple — valid JSON that can be edited by hand yet expressive enough for automated validation.

---

## 1  Top‑level fields

| Key         | Type            | Required | Description                                               |
| ----------- | --------------- | -------- | --------------------------------------------------------- |
| `name`      | `string`        | ✔        | Human‑readable name of the hardware target (e.g. "KL720") |
| `vendor`    | `string`        | ❌        | Manufacturer or tool‑chain (e.g. "Kneron")                |
| `version`   | `string`        | ❌        | Silicon / SDK version this profile refers to              |
| `opset`     | `integer`       | ✔        | Highest ONNX opset version fully covered by the profile   |
| `notes`     | `string`/`null` | ❌        | Free‑form remarks (markdown OK)                           |
| `operators` | `object`        | ✔        | Map of *operator name* → *operator spec* (see §2)         |

---

## 2  Operator spec

Each key under `operators` represents a single ONNX operator.  The value is an **object** that can contain any of the following properties:

| Property      | Type                                        | Required | Description                                                                                                                                                          |
| ------------- | ------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `status`      | `"supported" \| "partial" \| "unsupported"` | ✔        | Quick compatibility flag. (<span style="color:green">supported</span>, <span style="color:orange">partial</span>, <span style="color:red">unsupported</span>)        |
| `since_opset` | `integer`                                   | ❌        | First opset where this operator is usable on the hardware                                                                                                            |
| `attrs`       | `object`                                    | ❌        | Allowed attribute *values*. Keys are attr names; values can be:<br>  • literal primitive (exact match)<br>  • array (allowed set)<br>  • string regex (prefix `re:`) |
| `dtypes`      | `array< string >`                           | ❌        | Supported tensor data types (e.g. `["float16", "int8"]`)                                                                                                             |
| `modes`       | `array< string >`                           | ❌        | For ops such as `Resize` or `Pad`, list of supported modes                                                                                                           |
| `constraints` | `string`                                    | ❌        | Human‑readable limitations (e.g. "input rank ≤ 4", "kernel size must be odd")                                                                                        |
| `aliases`     | `array< string >`                           | ❌        | Alternate operator names used by vendor forks, if any                                                                                                                |

### Attribute value syntax

```jsonc
"attrs": {
  "mode": ["linear", "nearest"],          // only these modes accepted
  "activation_alpha": {"min": 0.0, "max": 1.0},
  "coordinate_transformation_mode": "re:.*_align_corners$"   // regex match (prefix "re:")
}
```

---

## 3  Example profile (KL720 revA)

```json
{
  "name": "KL720",
  "vendor": "Kneron",
  "version": "SDK 1.5.0",
  "opset": 13,
  "operators": {
    "Conv": {
      "status": "supported",
      "dtypes": ["float32", "float16"],
      "constraints": "kernel_h == kernel_w; stride ≤ 4"
    },
    "Relu": { "status": "supported" },
    "Elu":  { "status": "unsupported" },
    "Resize": {
      "status": "partial",
      "attrs": {
        "mode": ["linear", "nearest"],
        "coordinate_transformation_mode": ["half_pixel", "align_corners"]
      }
    }
  }
}
```

---

## 4  Validation

`onnx-op-check` runs a light JSON Schema validation at load time.  Developers can also call:

```bash
python -m onnx_op_check.validate profiles/KL720.json
```

and receive detailed error messages if a profile deviates from this specification.

---

## 5  Versioning guidelines

* **Non‑breaking additions** (e.g. new operators, extra dtypes) → **patch bump** (`1.0.0` → `1.0.1`).
* **Breaking changes** (renaming fields, removing ops) → **minor bump** (`1.0.0` → `1.1.0`).
* Major version will track fundamental schema overhauls.

---

## 6  FAQ

> **Q : 必須列出所有運算子嗎？**
> A : 建議列出常用且硬體特化的運算子。不在表中的運算子預設視為 **unsupported**。

> **Q : 若同一硬體有不同 SDK 版號？**
> A : 建議以檔名或 `version` 欄位加以區分，例如 `KL720_sdk1.5.json`。

---

[⬆ 返回頂端](#hardware-profile-json-schema)
