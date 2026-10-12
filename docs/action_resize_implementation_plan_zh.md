# Action 训练与推理统一 Resize 实施计划（Issue #30）

日期：2026-10-12。实施分支：`feat/30-action-resize`。关联：[Issue #30](https://github.com/petrichorz/cosmos-framework/issues/30)。

本文记录约定方案、代码职责、提交顺序和验收标准。各提交的验证项是验收要求；已执行的工作单独记录在末尾，不将计划项视为已经完成。

## 1. 目标与范围

目标是去掉 action 输入中拼接后 resize 和补边的逻辑，让送入 VAE 的 RGB 图像由有效相机画面直接组成。训练与推理同时适配，共用尺寸选择、相机缩放、RGB 拼接和空间元数据规则。

用户已确认不保留旧的“先 concat，再 resize / padding”行为。最终实现不增加兼容开关，不维护新旧两套空间处理流程。

新链路：

```text
读取 head 原始宽高 → 在 ACTION_RES_SIZE_INFO 中选择目标尺寸
    ↓
head resize 到 W×H，left/right 分别 resize 到 (W//2)×(H//2)
    ↓
拼接 RGB：上方 head，下方 left + right
    ↓
transform 保留空间尺寸，填写元数据并处理时间、action、文本
    ↓
整张 RGB 图像送入 VAE
```

本文所有尺寸写作 **宽×高**；PyTorch 空间维度和插值参数仍使用 **高、宽**。

- `concat_view`：最终图像为 `W×(H+H//2)`。
- `ego_view`：只读取 head，最终图像为 `W×H`。
- 按 head 原始宽高选择目标尺寸，不按拼接图宽高选桶；腕部跟随 head 目标尺寸，不独立选桶。
- 每个相机直接缩放到指定尺寸，允许放大和轻量宽高比变化；不保留比例后补边，不裁剪画面。
- 保持先拼接 RGB、再统一 VAE 编码的顺序。

按用户最新确认，本次只迁移当前 `causal_action_factory` 接入的 AgiBot / EgoSuite，以及 `inference/action.py` 中对应布局的通用 action 推理入口。旧 DROID / LIBERO 等 Dataset 的空间处理不在本次范围内；共享 adapter 删除参数时，仅同步其调用签名，不改旧 Reader 或扩展来源布局。旧工厂的空间输出不作为本阶段验收对象。

LIBERO 和 Robolab policy 服务端由用户后续重写，本次不修改其预处理、请求接口和测试，不要求对其旧行为保持兼容。本文的训推一致性验收针对当前训练链路与通用 action 推理。

不修改全局 `IMAGE_RES_SIZE_INFO`、`VIDEO_RES_SIZE_INFO`。片段索引、subtask、状态/action/mask、统计归一化、采样权重、`Pool(8)` 和 VAE 架构均保持现有职责。

## 2. 当前代码与改造位置

| 位置                                                                                                                              | 当前职责                                              | 本次调整                                                            |
| --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------- |
| [utils.py](../cosmos_framework/data/generator/utils.py#L76)                                                                       | 已有用户编写的 `ACTION_RES_SIZE_INFO`                 | 将此表接入 action 读取端，保留当前配置值                            |
| [causal_action_factory.py](../cosmos_framework/data/generator/action/datasets/causal_action_factory.py#L55)                       | 持有训练 resolution，构建来源 Reader 与共用 transform | 构造来源级 video_view，将 resolution 经已有参数传入；统一启用新链路 |
| [agibot_segment_lerobot_dataset.py](../cosmos_framework/data/generator/action/datasets/agibot_segment_lerobot_dataset.py#L31)     | 定义 AgiBot 相机映射与读取默认项                      | 暴露默认 video_view，方便工厂按来源配置                             |
| [egosuite_segment_lerobot_dataset.py](../cosmos_framework/data/generator/action/datasets/egosuite_segment_lerobot_dataset.py#L28) | 定义 EgoSuite 相机映射与读取默认项                    | 同上                                                                |
| [video_view.py](../cosmos_framework/data/generator/action/video_view.py#L93)                                                      | 选择相机、准备尺寸、拼接布局                          | 提交 1 增加 resolution；提交 2 去掉过渡的 None，按原始 head 选桶    |
| [segment_lerobot_dataset.py](../cosmos_framework/data/generator/action/datasets/segment_lerobot_dataset.py#L308)                  | 将元数据、时间戳交给 video_view 读取                  | 继续转发，不新增独立 resolution 字段或选桶逻辑                      |
| [transforms.py](../cosmos_framework/data/generator/action/transforms.py#L645)                                                     | 拼接后再 resize、补边并填写 image_size                | 删除空间变换阶段，始终记录实际尺寸                                  |
| [json_formatter.py](../cosmos_framework/data/generator/action/json_formatter.py#L115)                                             | 从 image_size 生成 prompt 分辨率与宽高比              | 新链路使用最终图像实际宽高比                                        |
| [inference/action.py](../cosmos_framework/inference/action.py#L78)                                                                | 推理 batch 构建时对整张媒体选桶、resize 和补边        | 接入共用相机预处理，batch 构建只组装数据                            |
| [inference/inference.py](../cosmos_framework/inference/inference.py#L652)                                                         | 转发 action 推理参数                                  | 传递明确的 action 档位和相机布局，不将拼接尺寸当相机尺寸            |
| [action_sft_dataset.py](../cosmos_framework/data/generator/action/datasets/action_sft_dataset.py#L139)                            | 共用 adapter 将 Reader 样本交给 transform             | 删除 transform resolution 传递；旧工厂只同步调用签名                |

### 配置归属

训练 resolution 由工厂持有，通用推理由采样配置明确提供；二者均表示 `ACTION_RES_SIZE_INFO` 的相机目标档位。`VideoViewConfig` 负责解释这个值并执行相机尺寸准备。`SegmentLeRobotDataset` 只调用已有 `video_view.read(...)` 接口，不需要知道 `ACTION_RES_SIZE_INFO`。

工厂通过已有 `video_view` 参数传递配置，不在 Reader 构建后逐个修改对象。建议沿用当前 `default_read_options` 的组织方式，在 AgiBot / EgoSuite 类上提供 `default_video_view`，工厂使用 `dataclasses.replace` 生成当前来源的配置。

来源配置不保存当前子集的 meta、episode 或解析后的目标尺寸。同一来源下各子集的原始尺寸仍可能不同，读取时根据传入的 meta 计算目标尺寸；相机映射按只读配置使用。跨进程序列化允许产生配置副本，不要求实现跨进程对象共享。

训练和推理要求明确的有效档位，支持字符串或整数形式的 `256`、`480`。`None`、`720` 等不支持值在入口初始化时明确报错，不根据整张拼接图自动推断或回退到通用视频分辨率表。直接构造 Reader 的调用按需使用明确的默认档位或传入值；统计工具不依赖空间补边，需确认调用接口仍有效。

通用推理现有 `image_size` 若继续保留为外部参数名，其 action 语义需明确为档位，而不是最终拼接图高度。

推理启动配置应与 checkpoint 对应的训练 recipe 使用同一档位和来源布局；能读取训练配置时从中获取，不能获取时要求明确提供。不得让训练默认 `480`、推理独立默认 `256` 而不提示配置差异。档位覆盖是用户主动选择的输入变化，需显式记录。

## 3. 尺寸选择和两种解码路径

在指定档位内，以 head 原始 `H/W` 和候选尺寸实际 `H/W` 的绝对差选最近尺寸。比较实际数值，不从 `"4,3"` 等配置键推导候选尺寸；相同候选尺寸和距离并列时按配置顺序稳定选择。

当前 `256` 中 `4,3` / `5,4` 都是 `320×256`，`3,4` / `4,5` 都是 `256×320`，因此是 7 个键、5 种实际尺寸。当前 `480` 是 5 个键、5 种实际尺寸，没有独立的 `5,4` / `4,5` 桶。此实现不自行增加桶或修改用户给定的值。

两种后端使用同一目标尺寸规则：

- `pyav_resize`：复用已加载的 LeRobot meta 中 head 视频的命名宽高；将各相机目标尺寸传给现有解码接口，在 RGB 拼接前完成 resize。
- 其他现有后端：从实际解码得到的 head 宽高选桶，然后将 head 和腕部缩放到目标尺寸，再复用现有拼接函数。

不为选桶额外扫描视频或重复读取 `info.json`。保持已有时间戳偏移、容差、相机顺序、dtype / 数值范围约定。

### 训练与推理共用的空间处理

复用现有 `VideoViewConfig`：`read` 负责读取，`compose` 接收已解码的 RGB，二者使用同一个 `_VideoLayout.sizes_to_resize` 计算目标，再交给原有 `_compose_prepared` 拼接。推理可以构造相机映射并调用 `compose`，不需要 LeRobot Reader 或训练数据。不新增 `prepare_action_video`、独立 resize 入口或第二套布局查询。训练模块不反向依赖 inference 模块，LeRobot 读取依赖继续按需导入。

同一组相机输入，在相同档位和布局下，训练与推理使用一致的选桶规则、相机目标尺寸、相机顺序和最终拼接尺寸，并保持正确的颜色通道、dtype 和数值范围。

按用户确认，各路径保留已有插值方式，不将 bicubic / bilinear 的差异作为改造或验收项。直接复用现有解码和 RGB 准备方法，禁止在 PyAV 已完成目标 resize 后再做一次 Tensor resize。

### 推理输入的两种明确形式

1. **待处理输入：** ego 媒体作为单相机输入；标准 concat 媒体按明确的来源布局拆出视图，再逐相机 resize 和拼接。AgiBot 使用 head / left / right，EgoSuite ego 只需要 head。
2. **已经按目标规则处理的输入：** 输入明确标记为已准备，附带档位和布局；校验尺寸符合该布局的候选目标后直接透传。不能再次对整图选桶、resize 或补边。

现有单个 `vision_path` 可以继续使用，不强制增加多相机路径接口。对于已知、无补边的 AgiBot 三视图布局，上方 `2/3` 高度为 head，下方 `1/3` 的左右两半为腕部。先按布局拆分，用 head 宽高选桶，再调用共用预处理。腕部此前可能已经缩小，但选桶不依赖腕部原始分辨率，无需恢复它。

不能从任意 concat 图的整体宽高比猜来源或视图边界，也不能把它当作 ego 输入悄悄重缩放。布局未知或包含不明确补边的输入不纳入标准拼接输入支持范围。拆分适配不允许复用旧整图 resize 路径。

批请求逐项执行相同规则；不同样本可能选择不同宽高比桶。沿用模型支持的变长空间输入组装方式，或按目标尺寸分组执行，不能为了拼 batch 再对整图 resize / padding。单请求和批请求中同一样本的空间输入应一致。

### 当前配置产生的尺寸

下表中的 latent 宽高按 VAE 空间压缩因子 16 计算。`ego_view` 的所有现有候选尺寸均不会触发空间 `2×2` patch 补齐。

| 档位 | 桶键      | Head / ego_view | 单个腕部 | concat_view | Concat latent 宽×高 | 2×2 patch 补齐     |
| ---- | --------- | --------------- | -------- | ----------- | ------------------- | ------------------ |
| 256  | 1,1       | 256×256         | 128×128  | 256×384     | 16×24               | 无                 |
| 256  | 4,3 / 5,4 | 320×256         | 160×128  | 320×384     | 20×24               | 无                 |
| 256  | 3,4 / 4,5 | 256×320         | 128×160  | 256×480     | 16×30               | 无                 |
| 256  | 16,9      | 448×256         | 224×128  | 448×384     | 28×24               | 无                 |
| 256  | 9,16      | 256×448         | 128×224  | 256×672     | 16×42               | 无                 |
| 480  | 1,1       | 512×512         | 256×256  | 512×768     | 32×48               | 无                 |
| 480  | 4,3       | 640×480         | 320×240  | 640×720     | 40×45               | latent 高度补 1 行 |
| 480  | 3,4       | 480×640         | 240×320  | 480×960     | 30×60               | 无                 |
| 480  | 16,9      | 640×352         | 320×176  | 640×528     | 40×33               | latent 高度补 1 行 |
| 480  | 9,16      | 352×640         | 176×320  | 352×960     | 22×60               | 无                 |

对 concat 布局，head 的 `W` 为 16 倍数、`H` 为 32 倍数足以保证整张图符合 VAE 空间对齐。当前表全部满足。若要进一步避免 `2×2` latent patch 补齐，head 的 `W` 需为 32 倍数、`H` 为 64 倍数；这不作为本次调整分辨率表的要求。

现有网络对奇数 latent 高度补零，再在输出端恢复原始 latent 尺寸。它发生在 VAE 编码之后，区别于本次移除的 RGB 补边，不需要修改网络补齐逻辑。

## 4. Transform 与下游元数据

### 删除空间变换阶段

从 `ActionTransformPipeline` 删除 `VideoResize` 调用，以及仅服务该阶段的 `pad_keys`、`keep_aspect_ratio` 和 resolution 参数。同步更新 `ActionSFTDataset`、相关训练工厂和通用推理调用接口，不增加 `resize_video` 开关。

空间尺寸由读取端或推理观测准备阶段负责。Transform 始终根据实际输入记录空间元数据，其他时间处理、prompt、文本分词、action 处理和 `sequence_plan` 构造保持原有流程。推理 batch 构建采用同样的元数据约定。

拼接图为 `H_final×W_final` 时填写：

```python
image_size = torch.tensor(
    [H_final, W_final, H_final, W_final],
    dtype=torch.float32,
)
```

保持现有 `(4,)` 形状、字段顺序和 dtype。推理 batch 根据现有接口再加 batch 维度；不能用 `pad_keys=[]` 代替删除空间阶段，因为现有函数可能留下为零的有效内容尺寸。

### VAE 后裁剪

现有模型通过 `image_size` 的前两个值记录整图尺寸，后两个值识别有效内容，并在 `_remove_padding_from_latent` 中裁剪。当前训练与通用推理的整图和有效内容尺寸都相等，因此不会裁掉有效画面。模型共享函数仍服务其他模态，本次不删除；通用推理生成的 RGB 保持目标布局尺寸。

### Prompt 与训练档位

- Prompt 的分辨率写最终拼接图尺寸，宽高比按实际宽高约分，不能写 head 桶标签。例如 `640×720` 的最终宽高比是 `8:9`。
- 共用 formatter 始终按实际尺寸生成宽高比，不保留查通用视频表的旧 action 分支。训练与推理使用相同的布局描述和空间字段。
- 不将 action resize 档位直接改写为模型噪声调度的 resolution。现有模型根据最终 `image_size` 推断训练档位，继续保持该行为。

这里有两个不同概念：入口的 `resolution="256"` 选择相机目标尺寸；模型的档位取决于最终图像。例如 `256` 档的 `320×384`、`448×384` 拼接图，在现有尺寸推断规则下会归为模型 `480` 档。训练与推理必须使用同一个最终尺寸推断规则，检查推理 sampler 的 shift 等尺寸相关设置是否另有固定覆盖；发现差异要明确对齐当前训练配置，不直接把相机档位当作噪声调度档位。

## 5. 三个核心提交

提交顺序为 1 → 2 → 3。每个提交均应通过自身检查；训练与推理在同一提交中切换，不允许中间提交只启用一侧。提交 1 尚未启用新链路，不等于最终实现保留兼容模式。

### 提交 1：扩展现有相机准备链路的尺寸规则

建议标题：`feat(action): 支持按 action 分辨率准备相机尺寸`

#### 修改范围

- 接入已有 `ACTION_RES_SIZE_INFO`，包含当前已确认的用户配置。
- 原位置扩展 `find_closest_target_size`，增加可指定尺寸表的参数，不移动或重写选桶算法。
- `VideoViewConfig` 增加 resolution 配置；在现有 `_VideoLayout.sizes_to_resize` 中先选择 head 目标，再复用已有腕部半尺寸规则。
- `_prepare_pyav_resized` 把目标尺寸直接传给现有解码接口；`_prepare_legacy` 使用实际 RGB head 尺寸选桶，在原 resize 循环中处理 head 和腕部。
- `_get_layout`、相机映射、`_compose_prepared` 保持原实现。同尺寸 RGB 在配置档位时跳过插值。
- 不新增 resize / 准备 / 已准备图像校验函数，不提前增加 DROID / LIBERO 专用布局。AgiBot / EgoSuite 接入及推理输入校验随提交 2 实现；旧来源不迁移。
- 在已有 `video_view_test.py` 中验证读取与 RGB 准备，不修改已有插值方式。

**提交边界：** 只扩展已有准备能力，训练工厂和推理入口尚未切换。本阶段配置的默认 `resolution=None` 让尚未接入的调用方保持现状；提交 2 接入明确档位后去掉此过渡默认，不作为最终的新旧行为开关。

**Review 重点：** 宽高顺序、原始 head 尺寸来源、半尺寸规则、相机顺序、允许放大、配置无子集状态污染、PyAV 无二次 resize、现有 dtype 约定。

#### 验证与验收

1. 覆盖当前两档及已有宽高比的实际候选尺寸，按原始 head 选桶，允许放大和缩小。
2. 检查现有 `read` 与 `compose` 的 ego / concat 输出尺寸、相机顺序及 float32 / uint8 dtype。
3. 检查 PyAV 直接收到 head / 腕部目标尺寸，时间戳偏移与容差不变，不再对输出 Tensor 插值。
4. 已处于各相机目标尺寸的 RGB 保持像素；重复调用或切换后端不污染来源配置。
5. 不支持的档位报错；现有相机缺失、尺寸、帧数和布局检查继续通过。训练与推理入口的成对验证属于提交 2。

### 提交 2：训练与推理原子切换到新空间链路

建议标题：`feat(action): 统一训练推理的读取阶段 resize 并移除拼接后补边`

#### 修改范围

- AgiBot / EgoSuite 暴露来源默认 video_view；工厂将明确的 resolution 经已有 video_view 参数传入，不给 Segment Reader 新增独立字段。
- 旧 Dataset 不迁移；共享 adapter 签名变化所涉及的旧工厂只同步移除无效参数，不新增布局或 resize。
- 删除 `ActionTransformPipeline` 的空间变换，统一填写 `image_size=[H,W,H,W]`，同步调整 adapter / 训练工厂 / 通用推理调用签名。
- 通用 action 推理适配 ego、标准 concat 拆分和已准备输入；更新顶层参数转发及使用示例。
- 共用 formatter 使用实际尺寸宽高比；训练与推理统一空间字段、布局描述及基于最终尺寸的模型档位处理。
- 更新本次范围内受影响的已有功能测试与通用推理测试。

**原子性要求：** 读取端 / 观测准备端的 resize、删除 transform 空间阶段、推理入口更新必须在同一提交内完成。该提交结束时生产链路只有新规则，不保留旧行为开关。

**Review 重点：** 训练与推理都调用提交 1 的共用能力；原始 concat 输入不能按整体比例选桶；batch 构建不二次缩放；时间 / action 处理未被跳过；单请求与批请求都适配；接口没有失效调用。

#### 验证与验收

1. AgiBot 原始 head `640×480`：`480` 档得到 `640×720`，`256` 档得到 `320×384`；EgoSuite ego 保持对应 head 目标尺寸。
2. 从同一组相机 RGB 构建训练样本和推理观测，检查 VAE 前相机顺序、目标尺寸、有效区域、dtype / 数值范围、`image_size`、prompt 空间字段和模型档位。
3. 标准 concat 按来源布局正确拆分，已准备输入没有二次插值；缺失视图和格式含义不明确的输入报错。已有缩小 / 编码损失的 concat 输入不声称与未损失的原始相机逐像素相同。
4. 检查通用 action 推理与当前 AgiBot / EgoSuite 训练的成对空间处理，不执行 LIBERO / Robolab 服务端验收。
5. 检查同一图像尺寸下训练噪声调度与推理 sampler 相关设置，显式覆盖不能导致档位处理不同。
6. 多进程 Reader 构造、统计工具和所有已调整调用接口通过相关检查。

### 提交 3：清理旧代码并记录完整链路验证

建议标题：`refactor(action): 清理旧 resize 辅助代码并补充训推验证记录`

#### 修改范围

- 删除无调用的 `VideoResize` 和旧空间处理代码；仍被本次范围外代码引用的辅助函数不作为死代码误删，待后续调用方重写再清理。当前训练与通用推理不再调用旧空间函数。
- 全仓搜索这些符号与旧参数，确认本次范围内没有剩余调用或文档继续指导用户执行旧空间链路，范围外引用明确记录。
- 保留其他模态仍需使用的模型通用裁剪和网络 patch 补齐逻辑。
- 整理当前 recipe、通用推理媒体输入示例和端到端验证记录。

**提交边界：** 生产行为已在提交 2 切换完成；本提交删除死代码和完善证据，不再引入另一轮空间语义变化。

**Review 重点：** 清理基于实际调用关系，避免误删其他模态能力；验证记录对应当前实现和真实命令，没有将未运行的检查写成通过。

#### 验证与验收

1. 用真实 AgiBot / EgoSuite 样本记录训练与推理 VAE 前图像；整幅图由有效视图组成，空间尺寸与预期一致。
2. 检查 VAE 输入 / latent、token 估算、prompt、模型档位、有效区域和 action 时间对齐。
3. 运行本地短训练，至少完成一个有效优化步骤，检查有限 loss 和反向传播；运行同配置 checkpoint 的 action 推理 smoke，检查有效输出及尺寸。
4. 对当前 recipe 覆盖 `256` / `480` 的预处理对照；至少对实际使用的档位执行训练和推理完整链路，未完整运行的档位明确记录。
5. 空间处理变化会使旧数值回归基线失效。只在实际执行相应回归后更新基线，不为让测试通过而放宽容差或抄写结果。

## 6. 可选后续提交：VAE 编译预热形状

现有 VAE AOT 预热形状来自通用 `VIDEO_RES_SIZE_INFO`。新 action 的 ego / concat 尺寸不一定在其中；没有匹配形状时，现有编码路径可以回退 eager。

此项不阻塞三个核心提交。如果实际训练启用了 VAE AOT，且验证发现需要新形状的编译覆盖，再独立提交：

`perf(action): 为 VAE 预热补充 action 输入尺寸`

只补充当前 action 表和保留布局派生的尺寸并去重，不替换全局视频表，不更改 resize 语义。训练与推理使用同一组空间形状，分别验证命中编译形状与 eager 回退。未执行时应在验证记录中说明，不声称新尺寸已完成编译覆盖。

## 7. 提交与最终 Review 要求

- 提交 1 已完成用户 Review，按授权提交本次修改；后续提交按各阶段范围继续实施。
- 每个代码提交说明具体变化、默认行为、验证命令与结果；未执行的验证明确写出原因。
- 功能验证优先修改已有相关测试文件，不为简单配置另建测试体系。
- Python 变更运行相关文件的 lint / format 和必要测试；真实训练使用已具备依赖的本地环境，涉及 torch 的 Python 命令按仓库要求清空 `LD_LIBRARY_PATH`。
- Markdown 修改执行 `pre-commit run rumdl-fmt --files <本次修改的 Markdown>`，包含格式化结果并复查。
- 提交前逐项检查 diff，保留用户已有改动，不混入无关格式化、重命名、Reader 清理或多进程重构。
- 最终验收要求训练与推理均按来源主要相机原始宽高选桶，各相机先 resize，RGB 拼接后空间尺寸不再变化，送入 VAE 无额外补边。
- 必须提供成对的训推预处理验证，覆盖选桶规则、目标尺寸、相机顺序、有效区域、dtype / 数值范围、`image_size` 和 prompt 空间字段；不要求不同插值后端逐像素一致。
- 本次不维护旧 checkpoint 的旧补边行为。更新后的训练和推理使用同一规则；旧 checkpoint 不保证输入分布或输出与旧版一致。

## 8. 提交 1 实施记录

本阶段扩展现有方法，经用户 Review 后提交，尚未接入训练工厂或通用推理。

- **选桶复用：** `find_closest_target_size` 保留在 `action/transforms.py` 原位置，保留原选择算法，只增加 `size_info` 参数。默认仍使用视频表；配置 action 档位时传入用户已有的 `ACTION_RES_SIZE_INFO`。
- **尺寸规则复用：** `_VideoLayout.sizes_to_resize` 先根据原始 head 宽高选目标，再调用既有 `_concat_resize_targets` 计算腕部半尺寸。`VideoViewConfig.resolution` 保存档位，不缓存子集的目标尺寸或元数据。
- **读取准备复用：** `_prepare_pyav_resized` 直接把目标尺寸传给解码器，输出不再插值；`_prepare_legacy` 在原循环中将已解码的 head / 腕部 resize 后拼接，已有目标尺寸跳过插值。没有新增准备链路。
- **布局与拼接复用：** `_get_layout`、`_compose_prepared` 和现有 AgiBot / EgoSuite 相机映射不变。删除本轮曾添加的独立尺寸计算、RGB resize、RGB 准备、已准备图像校验入口及来源专用布局。
- **阶段边界：** 现有生产调用方尚未传入档位，仍走原来的准备方式；没有仅切换训练。用户已有分辨率表和其他文档改动保留。

按用户确认，保留现有插值方式，不处理不同后端的 bicubic / bilinear 差异。提交 2 继续通过已有 `read` / `compose` 同时接入训练与推理，不增加重复预处理。

验证结果：

- 修改的 Python 文件 Ruff format 检查通过，功能 lint 检查通过；`transforms.py` 保留原有 import 排列，lint 单独忽略原有 `I001`。`git diff --check` 通过。
- `video_view_test.py` 与 `transforms_test.py` 共 **84 项通过**，覆盖两档选桶、不同 head 宽高、允许放大 / 缩小、相机顺序、head 与腕部尺寸、dtype、同尺寸不插值、时间戳偏移与后端切换。测试明确禁止 PyAV 路径再次调用 Tensor resize。
- 上述为隔离全局 NPU fixture 的 CPU 功能验证；本阶段没有执行完整训练或模型推理。

执行命令（仓库根目录，使用已有训练环境）：

```bash
LD_LIBRARY_PATH='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 \
  /mnt/sfs_turbo/public/apps/miniforge3/envs/cosmos-framework-py312/bin/python -m pytest \
  --confcutdir=cosmos_framework/data/generator/action \
  cosmos_framework/data/generator/action/video_view_test.py \
  cosmos_framework/data/generator/action/transforms_test.py -q --tb=short --disable-warnings
```

## 9. 提交 2 实施记录

本阶段经用户 Review 后授权提交，只接入 AgiBot / EgoSuite。复用提交 1 的选桶、读取 resize 和 RGB 拼接，没有新增空间准备函数或旧来源布局。

- 工厂按来源用 `replace(default_video_view, resolution=resolution)` 构造配置，经已有 `video_view` 参数和 `Pool(8)` 传入 Reader。直接构造来源 Reader 默认使用 `256`；训练工厂显式传入其配置档位（工厂默认 `480`）。不支持的档位在配置初始化时报错。
- `ActionTransformPipeline` 和推理 `build_action_batch` 不再对 RGB 做空间变换，只填写 `[H,W,H,W]`。Adapter 移除原来的 resolution 传递，时间处理、state / action / mask 和统计归一化不变。
- 通用推理按明确的 `ego_view` / 标准 AgiBot `concat_view` 准备相机，复用 `VideoViewConfig.compose`；`action_video_prepared=true` 校验目标尺寸后直接透传。该字段声明输入是否已准备，不选择旧的补边行为。
- action 推理要求显式提供 `image_size=256` 或 `480`，删除三个 action 默认 JSON 中独立的 `256` 默认值；顶层入口转发已准备标记。多个输出各自携带对应的 `image_size`。
- Prompt 使用最终图像实际宽高比和来源布局描述；例如 AgiBot `480` 的最终尺寸为 `640×720`，宽高比为 `8:9`。

### 推理输入示例

在现有 action 输入 JSON 中明确填写以下字段（其余模型、媒体路径和 action 参数沿用当前入口）：

```json
{
  "image_size": 480,
  "view_point": "concat_view",
  "action_video_prepared": false,
  "shift": 5.0
}
```

EgoSuite 使用 `view_point="ego_view"`。标准无补边 AgiBot concat 图作为待处理输入，上方 `2/3` 是 head、下方两半是 wrists；如果媒体已经按目标尺寸准备好，改为 `action_video_prepared=true`，空间像素直接透传。

`shift` 是独立的采样参数，本阶段不重写 sampler。当前 Edge 训练 recipe 对模型 `256` / `480` 档分别使用 `3` / `5`；上例使用相机 `480` 档，最终图像归为模型 `480` 档，因此显式填写 `5`，覆盖 action 默认的 `10`。相机 `256` 档的 concat 仍可能归为模型 `480` 档，需要按最终图像和 checkpoint recipe 选择 shift，而不能仅按相机档位设置。

### 已执行验证

- CPU 训练读取与 transform 相关测试 **199 项通过**，覆盖 VideoView、transform、causal block adapter 和 Segment Reader；action 推理及参数测试 **17 项通过**。
- 从本地真实 AgiBot / EgoSuite v3 数据构造 Reader，经真实 `Pool(8)` 返回到主进程，读取视频、应用 transform；将 PNG 媒体交给真实 `read_media_frames` 和通用 action 推理准备，核对最终尺寸、dtype 和完整有效区域。
- 另外直接调用 `get_causal_action_dataset`，两档都完成真实 `Pool(8)`、来源分组、`CausalActionSFTDataset` 取样，再与已准备 PNG 的推理空间元数据核对。该预处理 smoke 显式使用 mock placeholder 统计，不据此声称真实统计或完整模型验证通过。
- AgiBot 原始 head 与两腕均为 `640×480`：相机 `480` 档最终 `640×720`，`256` 档最终 `320×384`。EgoSuite 原始 head 为 `960×728`：最终分别为 `640×480` 和 `320×256`。
- 已准备输入透传、两档 ego / concat、视图顺序、时间补齐 / 截断、批输出空间元数据和 prompt 均有相关测试。不同解码插值方式没有要求逐像素一致。
- 验证图像保存在 `/tmp/action_resize_stage2/`，没有修改原始数据。本阶段没有运行 VAE、完整训练优化步骤或 checkpoint 推理；这些仍属于提交 3 的完整链路验证。

功能 lint 与 Markdown 格式检查通过，`git diff --check` 通过。保留三处既有 import 排列，相关文件 lint 忽略原有 `I001`；`args.py` / `args_test.py` 的既有无关格式差异未重写。

本地环境的可选 `torch_npu` 无法加载 `libhccl.so`，直接扩展运行整组 inference 测试会在原有模型导入处失败；这与 action 空间预处理测试结果分开记录，没有修改环境依赖或扩大本次实现范围。

执行 CPU 功能测试：

```bash
LD_LIBRARY_PATH='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 \
  /mnt/sfs_turbo/public/apps/miniforge3/envs/cosmos-framework-py312/bin/python -m pytest \
  --confcutdir=cosmos_framework/data/generator/action \
  cosmos_framework/data/generator/action/video_view_test.py \
  cosmos_framework/data/generator/action/transforms_test.py \
  cosmos_framework/data/generator/action/block_state_test.py \
  cosmos_framework/data/generator/action/datasets/segment_lerobot_dataset_test.py \
  -q --tb=short --disable-warnings

LD_LIBRARY_PATH='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 \
  /mnt/sfs_turbo/public/apps/miniforge3/envs/cosmos-framework-py312/bin/python -m pytest \
  --confcutdir=cosmos_framework/inference \
  cosmos_framework/inference/inference_test.py \
  cosmos_framework/inference/args_test.py \
  -k action -q --tb=short --disable-warnings
```
