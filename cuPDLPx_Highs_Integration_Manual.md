# HiGHS - cuPDLPx GPU 求解器集成手册 (v1.1)

本手册基于 2026-02-04 的最新修改记录，指导如何在 HiGHS 中集成并使用 cuPDLPx（基于 Halpern PDHG 的 GPU 加速 LP 求解器）。

---

## 1. 项目概述

cuPDLPx 是对现有 cuPDLP-C 的增强，利用现代 Halpern 重启策略在 GPU 上实现了显著的加速。主要特性包括：
- **高性能**：相比 cuPDLP-C 提升 2.5x - 6.8x。
- **现代 CUDA 支持**：使用 `CUDAToolkit` 进行构建。
- **深度集成**：支持通过 HiGHS 命令行和 API 直接切换。
- **MIP 支持**：可作为 MIP 松弛问题的底层 LP 求解器。

---

## 2. 环境要求

- **CUDA Toolkit**: 建议版本 ≥ 12.0。
- **GPU**: NVIDIA GPU（支持架构需在 CMake 中配置，如 SM80/SM89）。
- **HiGHS 源码**: 已配置 Git 子模块。

---

## 3. 编译与运行

### 3.1 获取源码
首先克隆 HiGHS 仓库，并初始化 cuPDLPx 子模块：
```bash
# 克隆仓库
git clone https://github.com/ERGO-Code/HiGHS.git
cd HiGHS

# 添加并初始化 cuPDLPx 子模块 (如果尚未添加)
git submodule add https://github.com/MIT-Lu-Lab/cuPDLPx.git highs/pdlp/cuPDLPx
git submodule update --init --recursive
```

### 3.2 编译命令
使用 `HIGHS_WITH_CUPDLPX=ON` 开启集成：
```bash
cd ~/code/HiGHS_1.13.1/HiGHS/highs/pdlp/cuPDLPx

# 1) 解决 GitHub 不通导致 FetchContent 拉不到 PSLP（你这台机子需要）
git config --global url."https://gitclone.com/github.com/".insteadOf "https://github.com/"

# 2) 用 CUDA 12.6 工具链（避免 /usr/bin/nvcc 10.1 与 ptxas 冲突）
rm -rf build
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.6

# 3) 编译
cmake --build build -j
验证命令
mkdir -p test_output
./build/cupdlpx ~/code/HiGHS_1.13.1/HiGHS/check/instances/afiro.mps test_output
ls test_output

mkdir build && cd build
cmake .. -DHIGHS_WITH_CUPDLPX=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES=80  # 根据你的显卡调整 (如 A100=80, 4090=89)
make -j$(nproc)
```

### 3.3 运行示例
使用命令行执行测试：
```bash
./bin/highs --solver pdlp --pdlp_use_cupdlpx true ../example.lp
```

对于 MIP 问题使用 PDLP 作为子求解器：
```bash
./bin/highs --mip_lp_solver pdlp --pdlp_use_cupdlpx true ../example_mip.lp
```

---

## 4. 关键修改说明 (Modifications Checklist)

### 4.1 构建系统 (CMake)
- **文件**: [highs/CMakeLists.txt](highs/CMakeLists.txt)
- **修改**: 
  - 使用 `find_package(CUDAToolkit REQUIRED)`。
  - 添加 `pdlp/cuPDLPx` 子目录。
  - 定义宏 `HIGHS_HAS_CUPDLPX`。
  - 链接 `libcupdlpx.so` 并包含相关头文件路径。
  - **重要**: 必须在 `target_sources` 中包含 `highs/pdlp/CupdlpxWrapper.cpp`。

### 4.2 选项与控制 (Options)
- **文件**: [highs/lp_data/HighsOptions.h](highs/lp_data/HighsOptions.h), [highs/lp_data/HighsOptions.cpp](highs/lp_data/HighsOptions.cpp), [app/HighsRuntimeOptions.h](app/HighsRuntimeOptions.h)
- **修改**:
  - `pdlp_use_cupdlpx` (bool): 默认 `false`。设为 `true` 时启用 cuPDLPx。
  - `mip_lp_solver` (string): 支持 `"pdlp"` 选项。
  - 参数映射：`pdlp_optimality_tolerance` 直接对应 cuPDLPx 的 `eps_optimal_relative`。

### 4.3 求解逻辑集成
- **文件**: [highs/lp_data/HighsSolve.cpp](highs/lp_data/HighsSolve.cpp)
- **修改**: 在 `solveLp` 内部添加 `if (options.pdlp_use_cupdlpx)` 分支，调用 `solveLpCupdlpx`。
- **文件**: [highs/mip/HighsLpRelaxation.cpp](highs/mip/HighsLpRelaxation.cpp)
- **修改**: 在 `run` 方法中添加对 `kPdlpString` 的处理，同步 `pdlp_use_cupdlpx` 选项。

### 4.4 接口封装 (Wrapper)
- **文件**: [highs/pdlp/CupdlpxWrapper.cpp](highs/pdlp/CupdlpxWrapper.cpp)
- **逻辑**:
  - **CSR 转换**: cuPDLPx 目前更稳定地支持 CSR 格式。Wrapper 中使用 `ensureRowwise()` 确保数据布局。
  - **内存所有权**: `create_lp_problem` 会执行深拷贝，因此局部变量 `local_matrix` 在函数结束前有效即可。
  - **解提取**: 实现了从 `cupdlpx_result_t` 到 `HighsSolution` 的映射，包括 Dual 值的计算。

---

## 5. 参数对照表

| HiGHS 命令行参数 | HiGHS 选项名称 | cuPDLPx 内部参数 |
| :--- | :--- | :--- |
| `--pdlp_iteration_limit` | `pdlp_iteration_limit` | `iteration_limit` |
| `--pdlp_optimality_tolerance` | `pdlp_optimality_tolerance` | `eps_optimal_relative` |
| `--primal_feasibility_tolerance` | `primal_feasibility_tolerance` | `eps_feasible_relative` |
| `--time_limit` | `time_limit` | `time_sec_limit` |

---

## 6. 当前不足与改进空间 (Known Limitations & Future Work)

1. **Basis 支持**: cuPDLPx 作为一阶算法，暂不支持生成 `HighsBasis`（BasisValid = false）。对于需要热启动或交叉 (Crossover) 的场景需配合其他工具。
2. **绝对路径依赖**: 当前 CMake 脚本中存在部分 Hardcoded 路径（如 `libcupdlpx.so` 的位置），在不同机器部署时需检查。
3. **参数透传**: 更多 cuPDLPx 特有参数（如 `reflection_gamma`）尚未通过 HiGHS 选项完全暴露。
4. **CUDA 架构自动探测**: 目前需在 CMake 手动指定 `CMAKE_CUDA_ARCHITECTURES`，未来可加入自动探测机制。
5. **数值稳定性**: 大规模高度退化问题的收敛性仍需通过更多 Benchmark 验证。

---

## 7. 详细代码集成步骤 (Step-by-Step Integration)

如果你需要手动在新的 HiGHS 版本中重新实现此集成，请遵循以下顺序：

### 阶段一：定义选项 (Options)

1.  **修改 `highs/lp_data/HighsOptions.h`**:
    *   在 `struct HighsOptionsStruct` 成员变量最后（大约 380 行）添加：
    ```cpp
    // --- 直接复制 ---
    bool pdlp_use_cupdlpx;  // 新增：使用 cuPDLPx 替代 cuPDLP-C
    // ---------------
    ```
    *   在 `HighsOptionsStruct()` 构造函数初始化列表中添加：
    ```cpp
    // --- 直接复制 ---
    pdlp_use_cupdlpx(false),
    // ---------------
    ```
    *   在 `HighsOptions::initRecords()` 函数中（大约 1290 行）添加：
    ```cpp
    // --- 直接复制 ---
    record_bool = new OptionRecordBool(
        "pdlp_use_cupdlpx", 
        "Use cuPDLPx (enhanced GPU solver) instead of cuPDLP-C: Default = false",
        advanced, 
        &pdlp_use_cupdlpx, 
        false);
    records.push_back(record_bool);
    // ---------------
    ```

2.  **修改 `app/HighsRuntimeOptions.h`**:
    *   在 `struct HighsCommandLineOptions` 成员变量最后添加：
    ```cpp
    // --- 直接复制 ---
    std::string cmd_pdlp_use_cupdlpx = "";
    // ---------------
    ```
    *   在 `setupCommandLineOptions` 函数中添加解析逻辑：
    ```cpp
    // --- 直接复制 ---
    app.add_option("--pdlp_use_cupdlpx", cmd_options.cmd_pdlp_use_cupdlpx,
                   "Use cuPDLPx (enhanced GPU solver) instead of cuPDLP-C:\n"
                   "\"true\"/\"on\"\n"
                   "\"false\"/\"off\" * default");
    // ---------------
    ```
    *   在 `loadOptions` 函数中添加同步逻辑：
    ```cpp
    // --- 直接复制 ---
    if (c.cmd_pdlp_use_cupdlpx != "") {
      if (setLocalOptionValue(report_log_options, "pdlp_use_cupdlpx",
                              options.log_options, options.records,
                              c.cmd_pdlp_use_cupdlpx) != OptionStatus::kOk)
        return false;
    }
    // ---------------
    ```

### 阶段二：实现 Wrapper

1.  **修改 `highs/pdlp/CupdlpWrapper.h`**:
    *   在文件末尾 `#endif // PDLP_CUPDLP_WRAPPER_H_` 之前插入：
    ```cpp
    // --- 直接复制 ---
    #ifdef HIGHS_HAS_CUPDLPX
    HighsStatus solveLpCupdlpx(HighsLpSolverObject& solver_object);
    HighsStatus solveLpCupdlpx(const HighsOptions& options, HighsTimer& timer,
                               const HighsLp& lp, HighsBasis& highs_basis,
                               HighsSolution& highs_solution,
                               HighsModelStatus& model_status, HighsInfo& highs_info,
                               HighsCallback& callback);
    #endif
    // ---------------
    ```

2.  **创建新文件 `highs/pdlp/CupdlpxWrapper.cpp`**:
    *   直接填入以下完整内容：
    ```cpp
    #include "pdlp/CupdlpWrapper.h"
    #ifdef HIGHS_HAS_CUPDLPX
    #include <cmath>
    #include <algorithm>
    #include "io/HighsIO.h"
    #include "cuPDLPx/include/cupdlpx.h"

    HighsStatus solveLpCupdlpx(HighsLpSolverObject& solver_object) {
      return solveLpCupdlpx(solver_object.options_, solver_object.timer_,
          solver_object.lp_, solver_object.basis_, solver_object.solution_, 
          solver_object.model_status_, solver_object.highs_info_, solver_object.callback_);
    }

    HighsStatus solveLpCupdlpx(const HighsOptions& options, HighsTimer& timer,
        const HighsLp& lp, HighsBasis& highs_basis, HighsSolution& highs_solution,
        HighsModelStatus& model_status, HighsInfo& highs_info, HighsCallback& callback) {
        
        const HighsSparseMatrix* a_matrix;
        HighsSparseMatrix local_matrix;
        if (lp.a_matrix_.isRowwise()) { a_matrix = &lp.a_matrix_; } 
        else { local_matrix = lp.a_matrix_; local_matrix.ensureRowwise(); a_matrix = &local_matrix; }

        matrix_desc_t A_desc;
        A_desc.fmt = matrix_csr;
        A_desc.m = lp.num_row_; A_desc.n = lp.num_col_;
        A_desc.data.csr.nnz = a_matrix->numNz();
        A_desc.data.csr.row_ptr = const_cast<int*>(a_matrix->start_.data());
        A_desc.data.csr.col_ind = const_cast<int*>(a_matrix->index_.data());
        A_desc.data.csr.vals = const_cast<double*>(a_matrix->value_.data());

        lp_problem_t* prob = create_lp_problem(lp.col_cost_.data(), &A_desc, 
            lp.row_lower_.data(), lp.row_upper_.data(), lp.col_lower_.data(), 
            lp.col_upper_.data(), &lp.offset_);
        
        pdhg_parameters_t params;
        set_default_parameters(&params);
        params.termination_criteria.iteration_limit = (int)options.pdlp_iteration_limit;
        params.termination_criteria.eps_optimal_relative = options.pdlp_optimality_tolerance;
        params.termination_criteria.eps_feasible_relative = options.primal_feasibility_tolerance;
        params.verbose = options.log_to_console;

        cupdlpx_result_t* result = solve_lp_problem(prob, &params);
        if (!result) { lp_problem_free(prob); return HighsStatus::kError; }

        highs_solution.col_value.resize(lp.num_col_);
        std::copy(result->primal_solution, result->primal_solution + lp.num_col_, highs_solution.col_value.begin());
        highs_solution.row_dual.resize(lp.num_row_);
        std::copy(result->dual_solution, result->dual_solution + lp.num_row_, highs_solution.row_dual.begin());
        highs_solution.value_valid = true;
        highs_solution.dual_valid = true;
        
        highs_info.pdlp_iteration_count = result->total_count;
        highs_info.objective_function_value = result->primal_objective_value;
        
        switch (result->termination_reason) {
            case TERMINATION_REASON_OPTIMAL: model_status = HighsModelStatus::kOptimal; break;
            case TERMINATION_REASON_PRIMAL_INFEASIBLE: model_status = HighsModelStatus::kInfeasible; break;
            case TERMINATION_REASON_DUAL_INFEASIBLE: model_status = HighsModelStatus::kUnbounded; break;
            case TERMINATION_REASON_TIME_LIMIT: model_status = HighsModelStatus::kTimeLimit; break;
            default: model_status = HighsModelStatus::kUnknown;
        }

        cupdlpx_result_free(result);
        lp_problem_free(prob);
        highs_basis.valid = false;
        return HighsStatus::kOk;
    }
    #endif
    ```

### 阶段三：接入求解流程

1.  **修改 `highs/lp_data/HighsSolve.cpp`**:
    *   在文件开头添加：
    ```cpp
    // --- 直接复制 ---
    #ifdef HIGHS_HAS_CUPDLPX
    #include "pdlp/CupdlpWrapper.h"
    #endif
    // ---------------
    ```
    *   在 `solveLp` 函数中，找到 `Use cuPDLP-C to solve the LP` 附近插入：
    ```cpp
    // --- 直接复制（替换原有的 solveLpCupdlp 调用块） ---
    #ifdef HIGHS_HAS_CUPDLPX
    if (options.pdlp_use_cupdlpx) {
        sub_solver_call_time.num_call[kSubSolverPdlp]++;
        sub_solver_call_time.run_time[kSubSolverPdlp] = -solver_object.timer_.read();
        try { call_status = solveLpCupdlpx(solver_object); } 
        catch (...) { call_status = HighsStatus::kError; }
        sub_solver_call_time.run_time[kSubSolverPdlp] += solver_object.timer_.read();
        return_status = interpretCallStatus(options.log_options, call_status, return_status, "solveLpCupdlpx");
    } else
    #endif
    {
        sub_solver_call_time.num_call[kSubSolverPdlp]++;
        sub_solver_call_time.run_time[kSubSolverPdlp] = -solver_object.timer_.read();
        try { call_status = solveLpCupdlp(solver_object); } 
        catch (...) { call_status = HighsStatus::kError; }
        sub_solver_call_time.run_time[kSubSolverPdlp] += solver_object.timer_.read();
        return_status = interpretCallStatus(options.log_options, call_status, return_status, "solveLpCupdlp");
    }
    // ---------------
    ```

2.  **修改 `highs/mip/HighsLpRelaxation.cpp`**:
    *   在 `run()` 方法中添加对 `pdlp` 的处理：
    ```cpp
    // --- 直接复制 ---
    else if (mip_lp_solver == kPdlpString) {
      if (!this->solved_first_lp) {
        use_solver = kPdlpString;
        lpsolver.setOptionValue("pdlp_use_cupdlpx",
                                mipsolver.options_mip_->pdlp_use_cupdlpx);
      } else {
        use_solver = kSimplexString;
      }
    }
    // ---------------
    ```

### 阶段四：配置构建系统

1.  **修改 `highs/CMakeLists.txt`**:
    *   在 `FAST_BUILD` 分支内插入：
    ```cmake
    # --- 直接复制 ---
    option(HIGHS_WITH_CUPDLPX "Enable cuPDLPx GPU solver" OFF)
    if(HIGHS_WITH_CUPDLPX)
        find_package(CUDAToolkit REQUIRED)
        add_subdirectory(pdlp/cuPDLPx)
        target_sources(highs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/pdlp/CupdlpxWrapper.cpp")
        target_link_libraries(highs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/pdlp/cuPDLPx/build/libcupdlpx.so")
        target_include_directories(highs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/pdlp/cuPDLPx/src")
        target_compile_definitions(highs PRIVATE HIGHS_HAS_CUPDLPX)
    endif()
    # ---------------
    ```

---

**最后更新**: 2026-02-04  
**维护者**: 开发团队
