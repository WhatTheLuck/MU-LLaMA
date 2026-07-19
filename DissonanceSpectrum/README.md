# Visual Dissonance Spectrum

从音频文件计算 Visual Dissonance 不协和度谱，并将谱矩阵进一步压缩为逐时刻的不协和度强度曲线。

## 安装依赖

在当前目录下执行：

```bash
pip install -r requirements.txt
```

依赖包括 NumPy、librosa 和 PyTorch。默认优先使用 PyTorch；没有可用 CUDA 时自动使用 CPU。

## 基本使用

```python
from DissonanceSpectrum.dissonance_spectrum import (
    calculate_dissonance_spectrum,
    calculate_dissonance_intensity,
)

# 默认以 4 FPS、10 秒窗口自动检测调性，并计算 tonic+self 不协和度谱。
spectrum = calculate_dissonance_spectrum("audio.wav")

# 一行计算各时刻的不协和度强度。
intensity = calculate_dissonance_intensity(spectrum)
```

`spectrum` 是形状为 `(pitch_bins, time_frames)` 的 NumPy 矩阵；`intensity` 是长度为 `time_frames` 的一维数组。

指定 A 小调并使用 A 小三和弦参考：

```python
spectrum = calculate_dissonance_spectrum(
    "audio.wav",
    reference_mode="tonic",
    reference_tonic="A",
    reference_tonality="minor",
    tonal_chord_ref=True,
)
```

自动分段检测主音与大/小调，并选择对应主和弦参考：

```python
spectrum = calculate_dissonance_spectrum(
    "audio.wav",
    reference_mode="tonic",
    tonal_chord_ref=True,
    auto_key_detect=True,
)
```

同时计算主音参考与自身参考并相加，可使用 `tonic+self`（也兼容内部名称 `tonic_self`）：

```python
spectrum = calculate_dissonance_spectrum(
    "audio.wav",
    reference_mode="tonic+self",
    reference_tonic="A",
    reference_tonality="minor",
    tonal_chord_ref=True,
)
```

如需时间轴、音高轴、处理后的 CQT 和实际计算后端，可使用：

```python
result = calculate_dissonance_spectrum("audio.wav", return_details=True)

spectrum = result["dissonance_spectrum"]
times = result["times"]
pitch = result["pitch"]
print(result["backend"])
```

## 主程序测试

直接运行文件会使用附带的 C4 音频执行一个低分辨率端到端测试：

```bash
python DissonanceSpectrum/dissonance_spectrum.py
```

也可以指定音频，或强制使用 NumPy 卷积：

```bash
python DissonanceSpectrum/dissonance_spectrum.py "audio.wav"
python DissonanceSpectrum/dissonance_spectrum.py "audio.wav" --numpy
```

## 不协和度谱参数

### 输入与参考谱

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `audio_path` | 必填 | 待计算音频路径，支持 librosa 能读取的音频格式。 |
| `reference_mode` | `"tonic+self"` | 参考谱模式：`specified` 指定音频、`tonic` 主音、`tonic+self` 主音与自身相加（兼容 `tonic_self`）、`self` 自身、`context_window` 前置时间窗口。 |
| `reference_audio_path` | `None` | `specified` 模式的参考音频；为 `None` 时使用附带的 C4 音频。 |
| `reference_frame` | `0` | 从参考音频中选择的 CQT 帧序号。 |
| `reference_tonic` | `"C"` | `tonic` 和 `tonic_self` 模式使用的主音，可取 C 至 B 的十二个音名。 |
| `reference_tonality` | `"major"` | 指定调式，可选 `major` 或 `minor`；启用和弦参考时分别选择 `maj.wav` 或 `min.wav`。 |
| `tonal_chord_ref` | `False` | 为 `True` 时根据主音和调式使用对应主和弦，否则使用主音单音。 |
| `auto_key_detect` | `True` | 自动分段检测主音及 major/minor 调式，并拼接对应的主音或主和弦参考谱。 |
| `auto_key_window_seconds` | `10.0` | 自动主音检测的分段时长，单位为秒。 |
| `average_frame` | `False` | 使用参考音频所有 CQT 帧的平均值，而不是 `reference_frame`。 |
| `context_window_seconds` | `1.0` | `context_window` 模式回看窗口的时长；当前帧不包含在参考窗口中。 |

### CQT 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `sr` | `22050` | 音频重采样率。 |
| `fps` | `4.0` | 每秒 CQT 帧数；大于 0 时据此自动计算 `hop_length`。 |
| `hop_length` | `512` | 相邻 CQT 帧的采样点间隔；`fps > 0` 时会被自动值覆盖。 |
| `n_octaves` | `8` | CQT 覆盖的八度数。 |
| `bins_per_octave` | `72` | 每个八度的频率 bin 数，越大则音高分辨率越高、计算量也越大。 |
| `fmin` | `"C1"` | CQT 最低频率，可传音名（如 `C1`）或 Hz 数值。 |
| `db_amp` | `False` | 与网页边栏兼容；最终计算与网页一致，始终使用线性 CQT 幅值。 |

输出的音高 bin 数为 `n_octaves * bins_per_octave`，时间分辨率约为 `sr / hop_length` 帧/秒。

### 噪声门参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `noise_filter` | `True` | 是否移除 CQT 噪声底。 |
| `noise_filter_mode` | `"relative_frame"` | 阈值模式，支持 `absolute`、`relative_global`、`relative_frame`、`percentile_global`、`percentile_frame`、`adaptive_frame`、`median_mad_frame`、`soft_relative_frame`、`soft_adaptive_frame`。 |
| `noise_thr` | `0.1` | 固定阈值或相对最大幅值的比例，具体含义由模式决定。 |
| `noise_percentile` | `20.0` | 百分位模式使用的百分位数。 |
| `noise_mad_k` | `2.5` | MAD 模式的阈值系数。 |
| `noise_soft` | `True` | 软门限会从幅值中减去噪声底；关闭时直接将阈值以下的值置零。 |

### 归一化参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `normalize_mode` | `"global"` | `frame` 逐帧最大值、`global` 全局最大值、`context_window` 窗口最大值或 `none` 不归一化。 |
| `normalize_independent` | `False` | 为 `False` 时计算谱和参考谱共享较大的归一化分母；为 `True` 时分别归一化。 |
| `normalize_context_window_seconds` | `1.0` | `context_window` 归一化模式的窗口时长。 |

### 峰值替换参数

峰值替换会将每帧 CQT 变为仅保留峰值的稀疏谱。将所有筛选参数设为 `None` 且关闭 `peak_normalize`，即可跳过峰值替换。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `peak_height` | `None` | 峰值最低高度。 |
| `peak_threshold` | `None` | 峰值相对相邻 bin 的最小高度差。 |
| `peak_distance` | `None` | 相邻峰值之间的最小 bin 距离。 |
| `peak_prominence` | `0.015` | 峰值最低显著性。 |
| `peak_width` | `None` | 峰值最小宽度。 |
| `peak_top_n` | `None` | 每帧最多保留幅值最高的 N 个峰。 |
| `peak_normalize` | `False` | 搜索峰值前是否将每帧幅值归一化到 0–1。 |

### 不协和曲线参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `curve_path` | `None` | 手动指定曲线 `.npz` 文件或包含 `dissonance_curve.npz`、`params.json` 的目录。 |
| `curve_max_den` | `60` | 近似有理数允许的最大分母。 |
| `curve_err_mode` | `"ratio_percentage"` | 邻近规则：固定误差 `ratio_err` 或按频率比缩放的 `ratio_percentage`。 |
| `curve_err_parameter` | `0.01` | 邻近有理数判定的误差参数。 |
| `curve_ignore_octave` | `True` | 是否将八度等价关系折叠后计算曲线。 |
| `use_precomputed_curve` | `True` | 优先读取 `dissonance_curves` 中参数匹配的预存曲线。 |
| `save_generated_curve` | `True` | 没有匹配曲线时，将新生成曲线保存到缓存目录。 |

### 缓存与加速参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `cache_cqt` | `True` | 在当前 Python 进程中缓存相同音频和参数的 CQT。 |
| `use_torch` | `True` | 使用 PyTorch 执行不协和矩阵卷积；导入失败时自动回退至 NumPy。 |
| `device` | `None` | Torch 设备，如 `"cpu"`、`"cuda"` 或 `"cuda:0"`；`None` 时自动选择 CUDA/CPU。 |
| `dtype` | `"float32"` | 计算精度，可选 `float32` 或 `float64`。 |
| `batch_frames` | `None` | 分批处理的时间帧数；长音频或显存不足时可设置为正整数。 |
| `return_details` | `False` | 是否返回包含矩阵、坐标轴、曲线和后端信息的字典。 |

## 不协和度强度参数

```python
intensity = calculate_dissonance_intensity(
    spectrum,
    reduction="sum",
    nonnegative=True,
    normalize=False,
    pitch_weights=None,
)
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `dissonance_spectrum` | 必填 | 二维不协和度谱，形状为 `(pitch_bins, time_frames)`。 |
| `reduction` | `"sum"` | 沿音高轴使用 `sum`、`mean` 或 `rms` 聚合。`sum` 与网页强度曲线一致。 |
| `nonnegative` | `True` | 聚合前是否将负值截断为 0。 |
| `normalize` | `False` | 是否将最终强度曲线按最大绝对值归一化。 |
| `pitch_weights` | `None` | 可选的一维音高权重，长度必须等于谱的音高 bin 数。 |

## 参考资源目录

- `ref_audio/notes`：十二个单音参考音频。
- `ref_audio/tonal_chords`：十二个大三和弦及十二个小三和弦参考音频。
- `dissonance_curves`：预存及运行时生成的不协和曲线。
