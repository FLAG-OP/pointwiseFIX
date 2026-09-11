# pointwiseFIX — FlagGems 广播 pointwise 的 block pointer 零步长编译崩溃修复

修复 [FlagGems issue #5739](https://github.com/flagos-ai/FlagGems/issues/5739)：在昇腾 NPU 上
启用 flag_gems 后，广播形状的 pointwise 运算（`torch.add((128,1), (1,128))` 等）在
block pointer 路径下触发 Triton 编译错误：

```
[ConvertLinalgRToBinary] encounters error:  (issue 环境, triton 3.2)
loc(...kernel.ttadapter.mlir): error: strides must not be zero
[ConvertTritonIRToLinalgIR] encounters error:  (本机复现, triton-ascend 3.5.1)
```

一句话版本：**广播是用"步长为 0 的虚拟指针"实现的（零拷贝），而 block pointer
这种寻址描述方式表达不了零步长——编译器 pass 直接拒绝。修复 = 检测到广播时，
这次调用自动退回线性寻址（步长 0 只是乘法常数，天然合法），并用 config 拷贝
避免污染全局单例。**

```
修复前 (bptr 开启): 广播 add/sub/le/gt... 编译期必崩 (strides must not be zero)
修复后: 广播自动走线性寻址, 12/12 用例绿; 非广播的 bptr 快路径完全不受影响
```

## 问题是怎么回事

issue 的最小复现：

```python
import torch, torch_npu, flag_gems
a = torch.randn(128, 1, device='npu:0')
b = torch.randn(1, 128, device='npu:0')
c = torch.add(a, b)        # enable 前: OK
flag_gems.enable()
c = torch.add(a, b)        # bptr 路径下: 编译错误!
```

**为什么是"可能会"触发**：本仓库 Ascend 默认配置 `prefer_block_pointer=False`
（线性寻址），所以默认不崩。但 issue 环境（flagtree 0.6.0/triton 3.2）、
NVIDIA 后端（默认 True）、或用户显式开启 bptr 时必崩。我们在本机把
`prefer_block_pointer` 强制 True 完整复现并隔离了根因：

| 同样开 bptr | 结果 |
|---|---|
| 非广播 (128,128)+(128,128) | OK，数值正确 |
| 广播 (128,1)+(1,128) | MLIRCompilationError（**0 步长是唯一变量**） |

**根因链**：flag_gems 的 `pointwise_dynamic` 框架为零拷贝处理广播，用
`broadcasted_stride()` 把广播维的 stride 置 0，包成 `StridedBuffer` 传给 kernel。
线性寻址下 `地址 = base + i*0` 完全合法；但 block pointer（`tl.make_block_ptr`）
的 MLIR 表示（memref descriptor）**要求步长非零**，TritonToLinalg pipeline
（`ConvertTritonIRToLinalgIR` / `ConvertLinalgRToBinary`）直接拒绝编译。

交叉证据：上游未合并的 [PR #5668](https://github.com/flagos-ai/FlagGems/pull/5668)
（NVIDIA 侧）是同一根因的另一投影（报错文案 `Last dimension must be contiguous`），
并额外警告了一个次生 bug（见下）。

## 修复思路

改 `utils/pointwise_dynamic.py` 的 `prepare_args` / `_call_real_impl`，三件事：

1. **广播守卫**：慢路径检测到任何 operand 形状 ≠ task_shape（即被广播）时，
   本次调用禁用 block pointer，走线性寻址——两种寻址正确性等价，bptr 只是
   优化路径，不该为它牺牲合法性
2. **config 拷贝**：用 `dataclasses.replace` 生成 per-call config，不再原地改
   共享单例。原代码的 INT32_MAX 守卫（`self.config.prefer_block_pointer = False`）
   是**全局性污染**：一次大张量调用会静默关闭所有其他算子的 bptr（PR #5668
   也指出了这一点）——本修复顺带治好
3. **快路径不动**：`use_fast_path`（同形连续）本来就不产生 0 步长，零改动

```
广播调用 → 检测 stride 将含 0 → per-call config (bptr=False) → 线性寻址 kernel
非广播调用 → 原路径 (bptr 保持) → bptr kernel                    ← 不受影响
INT32_MAX 守卫 → 同样走 per-call config                          ← 不再污染全局
```

## 怎么用

前置：昇腾环境（Ascend 910 + CANN 8.5 / torch 2.10 / triton-ascend 3.5.1 /
flag_gems 5.3.5 验证；issue 报障环境 triton 3.2 同病）。

**部署**（一个文件）：

```bash
cp /path/to/site-packages/flag_gems/utils/pointwise_dynamic.py /path/to/backup/
cp src/pointwise_dynamic.py /path/to/site-packages/flag_gems/utils/
find /path/to/site-packages/flag_gems/utils -name __pycache__ -exec rm -rf {} +
```

或 `git apply pointwise_dynamic.patch`（91 行，含署名）。

**验证**：

```bash
ASCEND_LAUNCH_BLOCKING=1 python tests/repro_5739.py     # issue 原例 (默认配置, 应 OK)
ASCEND_LAUNCH_BLOCKING=1 python tests/test_red.py       # 强制 bptr 的 12 用例矩阵: 全 PASS
ASCEND_LAUNCH_BLOCKING=1 python tests/guard_sidecheck.py # 守卫副作用: 单例不污染/分流正确
ASCEND_LAUNCH_BLOCKING=1 python tests/repro_matrix.py   # 15 用例广播矩阵
ASCEND_LAUNCH_BLOCKING=1 python tests/perf_guard.py     # 性能护栏
```

**回归**：

```bash
cd /root/FlagGems
python -m pytest tests/test_pointwise_dynamic.py -q --ref cpu   # 1232 passed / 0 failed
python -m pytest tests/test_add.py -q --ref cpu                 # 1010 passed
# 注: tests/test_mul.py 的 ~44 失败为存量 (mul kernel inf 溢出, 原版同样 43 失败,
#     git stash 对照裁定), 与本修复无关; 必须带 --ref cpu (NPU fp64 参考污染问题,
#     见 geluFIX 报告 §5)
```

## 性能说明（实测，含机器噪声声明）

- **默认配置零影响**：Ascend 默认 `prefer_block_pointer=False`，广播守卫是
  no-op（检测后不改变任何行为）
- **bptr 环境下广播从"崩"变"可用"**：强制 bptr 时广播 (4096,1)+(1,4096) 走
  线性寻址 1081 µs，非广播同规模 bptr 路径 19404 µs——广播读的内存少
  （32KB vs 128MB），反而快是合理的
- 测量时 NPU 0 卡被外部 benchmark 进程占用（AICore 99%），绝对值有干扰，
  相对结论（守卫 no-op / 广播可用）不受影响
- 附带的手写 kernel 实验（`tests/handwrite_bench.py`）发现：**大张量（128MB）
  手写 Triton add 比原生快 38%**，中小张量差距来自 Python host launch 链路
  （~35µs/次）——这是框架性能路线的独立参考数据，详见报告 §6

## 后续优化路线与 stage-2 进展

本修复是四阶段路线的 stage-1（correctness）。stage-2（host 链路减负）已完成
定位与原型（`tests/stage2_*.py`，报告 §5.1）：

| host enqueue 分解（4MB add） | µs/次 | 占比 |
|---|---|---|
| 原生 torch.add（C++ + aclnn） | 8.9 | 8% |
| 裸 Triton launch | 33.4 | 29% |
| flag_gems add（框架 Python 层） | 123.1 | **63% ← 靶位** |

plan-cache 原型（缓存 prepare_args 的路径决策，仅重建 StridedBuffer）：
**127.2 → 84.6 µs（-34%）**，输出 `torch.equal` 逐位一致；直调 overload 的
收益上限 -52%。原型未合入 src（缓存失效策略需上游产品化决策）。
stage-3（形状感知 fallback 阈值）与 stage-4（AOT 预编译）未动。

## 目录结构

```
├── README.md                    # 本文
├── pointwise_dynamic.patch      # 91 行 diff（含署名），git apply 用
├── src/
│   ├── pointwise_dynamic.py     # 修复后完整文件
│   └── pointwise_dynamic.py.orig# v5.3.5 原版备份
├── tests/
│   ├── repro_5739.py            # issue 复现
│   ├── test_red.py              # TDD 红转绿: 强制 bptr 的 12 用例矩阵
│   ├── repro_matrix.py          # 15 用例广播矩阵 (默认配置)
│   ├── guard_sidecheck.py       # 守卫副作用验证 (单例/分流)
│   ├── perf_guard.py            # 性能护栏
│   ├── probe_force_bptr.py      # 根因复现: 强制 bptr 触发崩溃
│   ├── probe_isolate.py         # 根因隔离: bptr 下广播 vs 非广播
│   ├── handwrite_bench.py       # Triton vs 原生性能实验 (路线参考)
│   ├── host_overhead.py         # host launch 开销分解 (路线参考)
│   ├── stage2_locate.py         # stage-2: host 开销三层分解 (定位)
│   ├── stage2_profile.py        # stage-2: 框架层 cProfile 内部分解
│   ├── stage2_proto.py          # stage-2: 直调 overload 收益上限
│   └── stage2_plancache.py      # stage-2: plan-cache 原型 (-34%, 数值equal)
└── docs/
    └── POINTWISE_5739_REPORT.md # 完整报告: 根因链、复现矩阵、性能数据、修复路线
```

## 如果你想深究 bug 在哪一行

| 原版位置（src/*.orig） | 函数 | 与本次修复的关系 |
|---|---|---|
| L1533 | `prepare_args` 的 `self.config.prefer_block_pointer = False` | **单例污染点**（INT32_MAX 守卫写法错误） |
| L1595-1612 | `prepare_args` 慢路径的 `broadcasted_stride(...)` | **0 步长产生点**（行为正确，但下游 bptr 承接不住） |
| L1321 | `_call_real_impl` | 修复挂载点（per-call config 切换 + finally 恢复） |
| shape_utils.py L119 | `broadcasted_stride` 的 `0 if ... else stride[i]` | 0 步长源头（未改，语义正确） |

真正的拒绝点在上游 triton-ascend 的 TritonToLinalg pipeline（memref descriptor
不允许 0 步长），但 flag_gems 侧分流是更小的修复——上游若支持 0 步长（如
NVIDIA 的 expand 方式），分流自然失效也无害。

## 已知边界（不装完美）

- **守卫只覆盖通用 `utils/pointwise_dynamic.py`**：各硬件 backend 的独立副本
  （`_cambricon/`、`_sunrise/` 等）未改——它们有自己的 prepare_args 副本，
  若某 backend 默认开 bptr 且遇广播会仍崩（需各自修，模式相同）
- **仅 slow path 检测**：`use_fast_path`（同形连续）不产生 0 步长故无需守卫，
  但若未来 fast path 支持广播需重新评估
- issue 环境（triton 3.2 / CANN 9.0）未实机验证——复现是在本机 triton 3.5.1
  强制 bptr 得到的等价失败（报错 pass 名从 ConvertLinalgRToBinary 漂移到
  ConvertTritonIRToLinalgIR，同 pipeline 族）
- mul 的 44 个 pytest 失败为存量 inf 溢出（原版 43 失败对照），与本修复无关，
  未在本仓库处理
- 性能数据在共享机器上测得（外部负载），绝对值仅供相对比较

## 验证矩阵汇总

| 验证项 | 结果 |
|---|---|
| 根因隔离（bptr 下广播 vs 非广播） | 非广播 OK / 广播崩——0 步长是唯一变量 |
| TDD 红→绿（强制 bptr 12 用例: 5 位置广播 × 5 算子 + bptr 正常路径 + 数值） | 红 10 FAIL → **绿 12/12** |
| 守卫副作用（单例不污染 / 非广播仍走 bptr / 广播走线性） | 三项全过（kernel 缓存双变体实证） |
| 15 用例广播矩阵（默认配置） | 15/15（修复前后一致，no-op 验证） |
| 仓库 pytest test_pointwise_dynamic --ref cpu | **1232 passed / 0 failed** |
| 仓库 pytest test_add --ref cpu | 1010 passed |
| 回归裁定（test_mul 44f） | 原版 43f 对照——存量 inf 溢出，非本修复引入 |
| 性能护栏 | 默认配置 no-op；bptr 环境广播 1081µs（修复前不可用） |
