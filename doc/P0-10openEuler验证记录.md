# P0-10 openEuler CPU smoke 验证记录

- 任务：P0-10
- 主责：成员 B
- 复核：成员 A
- 状态：部分完成（B 已提交，待 A 复核）
- 日期：2026-07-17
- 验证提交：`91ae683106663d103c87f70db93ac6813d882dce`

## 1. 验证结论

TriSched 已在固定的 openEuler 24.03 LTS-SP4 x86_64 用户态镜像中完成一次性 CPU smoke。全新容器内创建独立虚拟环境，安装锁定依赖和非 editable 项目包后，完整测试、pipeline、checkpoint 无训练复评和结构化证据检查均通过。

本记录证明当前代码与 openEuler 24.03 LTS-SP4 用户态、glibc 2.38 和 Python 3.11.6 兼容。由于 Docker Desktop 使用 WSL2 内核，本次结果不等价于原生 openEuler 内核或物理机验证；P0-10 仍须由 A 独立复核，新增的远端 CI job 也须在推送后确认。

## 2. 固定环境

| 项目 | 实际值 |
| --- | --- |
| 镜像 | `openeuler/openeuler:24.03-lts-sp4` |
| manifest digest | `sha256:17c15554be2a5bc46023acb6e04d609d77642b8c20e236e88deb18e41ae4558e` |
| 本地 image ID | `sha256:9f5566b65d3ea860bade99330ded684abd87f89861b97b7b9e99d3a4c5b834ee` |
| 用户态发行版 | openEuler 24.03 LTS-SP4，x86_64 |
| 内核 | WSL2 `5.15.167.4-microsoft-standard-WSL2` |
| CPU | 12th Gen Intel Core i5-12500H |
| Docker | Docker Desktop Linux engine 28.0.4 |
| Python / pip | Python 3.11.6 / pip 23.3.1 |
| glibc | 2.38 |
| NumPy / pytest | NumPy 1.26.4 / pytest 7.4.0 |

镜像由 digest 而不是可变 tag 固定。NumPy manylinux wheel 的本地缓存不是 Git 交付物；其 SHA-256 为 `666dbfb6ec68962c033a450943ded891bed2d54e6755e35e5835d63f4f6931d5`，脚本会在使用前校验。CI 未携带该缓存时仍按锁文件从 Python 包索引安装。

## 3. 从无容器状态复现

确认 Docker Desktop 已启动后，从仓库根目录执行：

```powershell
$image = "openeuler/openeuler:24.03-lts-sp4@sha256:17c15554be2a5bc46023acb6e04d609d77642b8c20e236e88deb18e41ae4558e"
docker pull $image
.\scripts\run_openeuler_smoke.ps1
Get-Content outputs\p0-10-openeuler\validation.json
docker ps -a --filter name=trisched-openeuler-smoke
```

PowerShell 包装脚本先检查 Git commit、Docker engine 和固定镜像，再创建名为 `trisched-openeuler-smoke` 的临时容器。容器使用 `--rm`，结束后最后一条命令应不返回该容器。容器内脚本依次执行：

1. 记录发行版、内核、架构和 CPU；
2. 检查镜像自带的 Python、pip 和 venv 能力；
3. 在 `/opt/trisched-venv` 创建全新虚拟环境；
4. 安装锁定依赖与非 editable TriSched 包，并在仓库外验证导入路径；
5. 执行完整 pytest；
6. 执行安装包的 smoke pipeline；
7. 从 checkpoint 复评，不重新训练；
8. 检查样本数、合法率、数据隔离和 checkpoint hash。

## 4. 实际结果

| 检查项 | 结果 |
| --- | --- |
| 非 editable 导入 | `/opt/trisched-venv/lib64/python3.11/site-packages/trisched/__init__.py` |
| `pip check` | `No broken requirements found` |
| 完整测试 | `139 passed in 3.10s` |
| train / validation / test | 24 / 10 / 20 |
| pipeline test masked MLP `mean_ratio` | 1.0 |
| checkpoint 复评 `mean_ratio` | 1.0 |
| checkpoint 复评合法率 | 1.0 |
| train/validation、train/test、validation/test hash 交集 | 0 / 0 / 0 |
| checkpoint SHA-256 | `42f86f33b04329278f51a6b5b461a14c367bc3f6ac1f2b6308340bf204398926` |
| 结构化状态 | `pass` |
| 临时容器 | 运行结束后已删除 |

`mean_ratio = 1.0` 只表示当前 masked MLP 在这组 smoke 数据上与 HEFT 持平，不构成性能领先结论。

## 5. 证据与校验值

本地证据写入 `outputs/p0-10-openeuler/`，该目录被 Git 忽略：

```text
platform.txt                   发行版、内核、架构和 CPU
pip-freeze.txt                 容器虚拟环境依赖
pytest.xml                     139 项测试的 JUnit 证据
validation.json               最终结构化通过结论
sha256.txt                     锁文件、许可证和 checkpoint hash
smoke.log                      最终完整终端日志
pipeline/                      pipeline 结果与 checkpoint
evaluate/                      checkpoint 独立复评结果
wheelhouse/                    本地 wheel 缓存，不提交
smoke-attempt1-dnf-failure.log 第一次真实失败
smoke-attempt2-validation-parser.log 第二次真实失败
```

| 文件 | SHA-256 |
| --- | --- |
| `requirements-lock.txt` | `e94a62623dbf7f9702292c405fe14ed9d1b8829e93f620eaec404e1bc0c9208f` |
| `LICENSE` | `42aaec46bfd7037f5b52c9b79f430a73af4ab853ac3680b508f54f4c02979d69` |
| `pipeline/masked_mlp.npz` | `42f86f33b04329278f51a6b5b461a14c367bc3f6ac1f2b6308340bf204398926` |
| `scripts/openeuler_smoke.sh` | `7eaf8e782b6c86586a378832ba957e8c6e5aedd4f7e149455276f90acaf4046c` |
| `scripts/run_openeuler_smoke.ps1` | `d24466244ba4d8b453f209a17b53a5558e7adc439b658ed6ec8f53822752c7fe` |
| `.github/workflows/ci.yml` | `c06734f56959b274ebc99ec4cbc289e4c2c6401fb53df75509c9a6bcfabf6b91` |
| `.gitattributes` | `1d5d71dcc6ccd28ab46a335349ecc74742ca4b81f4fcad212f8d134f70f1f13e` |

运行输出不进入源码提交。新增 GitHub Actions job 会在远端重新生成并上传 `platform.txt`、`pytest.xml`、`validation.json`、pipeline 和 evaluate 摘要等证据，保留 14 天。

## 6. 两次失败与修复

### 失败 1：openEuler 软件源不可用

第一次尝试通过 `dnf` 安装运行时，openEuler `everything` 仓库出现 metalink 404/504、SSL 读取失败和低速超时，最终所有镜像均失败。失败日志已保留为 `smoke-attempt1-dnf-failure.log`。

处理决策：固定镜像已经提供 Python 3.11.6、pip、`ensurepip` 和 `venv`，而项目是纯 Python 包，因此不再把 `dnf` 作为 smoke 的非必要前置条件。依赖仍由 `requirements-lock.txt` 锁定，NumPy wheel 在本地复跑时另做 SHA-256 校验。该处理绕开的是外部仓库可用性，不是绕过项目依赖检查。

### 失败 2：JUnit 根节点假设错误

第二次尝试中 139 项测试、pipeline 和 checkpoint 复评均已通过，但证据脚本假设 JUnit XML 根节点直接包含 `tests` 属性，遇到 `testsuites/testsuite` 结构时抛出 `KeyError: 'tests'`。失败日志已保留为 `smoke-attempt2-validation-parser.log`。

修复：解析器现在同时支持根节点 `tests` 属性和对子 `testsuite` 的计数求和。修复后删除旧容器并从无容器状态重新运行，最终通过。

## 7. CI 与复核状态

CI workflow 已增加固定 digest 的 `openeuler-smoke` job，设计为拉取镜像、运行同一容器脚本、校验 `validation.json` 并上传结构化证据。该 job 尚未在远端执行，因为当前变更尚未由用户推送。

A 的独立复核至少应完成：

- [ ] 从已推送的不可变 commit 开始，不复用 B 的容器或虚拟环境；
- [ ] 核对镜像 digest、发行版、架构、Python、glibc 和 CPU 记录；
- [ ] 运行完整 smoke，确认 139 项测试、数据集计数、零 hash 交集和 checkpoint 复评；
- [ ] 确认运行后没有残留临时容器；
- [ ] 检查 GitHub Actions 的 openEuler job 和 artifact；
- [ ] 在任务日志记录“通过”或“退回”，再决定 P0-10 是否关闭。

## 8. 已知边界

- Docker 容器复用了 WSL2 内核，只证明 openEuler 用户态兼容；原生 openEuler VM/物理机是更强的后续证据。
- smoke 使用小规模确定性合成数据，只验证安装、正确性闭环和可复现性，不代表真实 benchmark 性能。
- 当前依赖仍包含网络安装路径；正式离线提交包需要完整 wheelhouse、许可证清单和离线安装演练。
- 远端 openEuler CI 在用户推送并成功运行前属于待验证项。
- P0-10 在 A 独立复核前保持“部分完成”。
