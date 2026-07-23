#include "pdlp/CupdlpWrapper.h"

#ifdef HIGHS_HAS_CUPDLPX

#include <algorithm>
#include <vector>

#include "io/HighsIO.h"
#include "lp_data/HighsLp.h"
#include "util/HighsSparseMatrix.h"
#include "cupdlpx.h"

HighsStatus solveLpCupdlpx(HighsLpSolverObject& solver_object) {
  return solveLpCupdlpx(solver_object.options_, solver_object.timer_,
                        solver_object.lp_, solver_object.basis_,
                        solver_object.solution_, solver_object.model_status_,
                        solver_object.highs_info_, solver_object.callback_);
}

HighsStatus solveLpCupdlpx(const HighsOptions& options, HighsTimer& timer,
                           const HighsLp& lp, HighsBasis& highs_basis,
                           HighsSolution& highs_solution,
                           HighsModelStatus& model_status,
                           HighsInfo& highs_info, HighsCallback& callback) {
  (void)timer;
  (void)callback;

  const HighsSparseMatrix* a_matrix;
  HighsSparseMatrix local_matrix;
  if (lp.a_matrix_.isRowwise()) {
    a_matrix = &lp.a_matrix_;
  } else {
    local_matrix = lp.a_matrix_;
    local_matrix.ensureRowwise();
    a_matrix = &local_matrix;
  }

  std::vector<int> row_ptr(a_matrix->start_.size());
  std::vector<int> col_ind(a_matrix->index_.size());
  std::transform(a_matrix->start_.begin(), a_matrix->start_.end(),
                 row_ptr.begin(),
                 [](HighsInt value) { return static_cast<int>(value); });
  std::transform(a_matrix->index_.begin(), a_matrix->index_.end(),
                 col_ind.begin(),
                 [](HighsInt value) { return static_cast<int>(value); });

  matrix_desc_t A_desc;
  A_desc.fmt = matrix_csr;
  A_desc.m = static_cast<int>(lp.num_row_);
  A_desc.n = static_cast<int>(lp.num_col_);
  A_desc.data.csr.nnz = static_cast<int>(a_matrix->numNz());
  A_desc.data.csr.row_ptr = row_ptr.data();
  A_desc.data.csr.col_ind = col_ind.data();
  A_desc.data.csr.vals = const_cast<double*>(a_matrix->value_.data());

  const objective_sense_t objective_sense =
      lp.sense_ == ObjSense::kMinimize ? OBJECTIVE_SENSE_MINIMIZE
                                       : OBJECTIVE_SENSE_MAXIMIZE;
  lp_problem_t* prob =
      create_lp_problem(lp.col_cost_.data(), &A_desc, lp.row_lower_.data(),
                        lp.row_upper_.data(), lp.col_lower_.data(),
                        lp.col_upper_.data(), &lp.offset_, &objective_sense);

  pdhg_parameters_t params;
  set_default_parameters(&params);
  params.termination_criteria.iteration_limit =
      static_cast<int>(options.pdlp_iteration_limit);
  params.termination_criteria.eps_optimal_relative =
      options.pdlp_optimality_tolerance;
  params.termination_criteria.eps_feasible_relative =
      options.primal_feasibility_tolerance;
  params.termination_criteria.time_sec_limit = options.time_limit;
  params.verbose = options.log_to_console;

  highsLogUser(options.log_options, HighsLogType::kInfo,
               "Entering solveLpCupdlpx\n");
  cupdlpx_result_t* result = solve_lp_problem(prob, &params);
  highsLogUser(options.log_options, HighsLogType::kInfo,
               "Leaving solveLpCupdlpx\n");
  if (!result) {
    lp_problem_free(prob);
    return HighsStatus::kError;
  }

  highs_solution.col_value.resize(lp.num_col_);
  std::copy(result->primal_solution, result->primal_solution + lp.num_col_,
            highs_solution.col_value.begin());
  highs_solution.row_value.resize(lp.num_row_);
  a_matrix->product(highs_solution.row_value, highs_solution.col_value);
  highs_solution.col_dual.resize(lp.num_col_);
  std::copy(result->reduced_cost, result->reduced_cost + lp.num_col_,
            highs_solution.col_dual.begin());
  highs_solution.row_dual.resize(lp.num_row_);
  std::copy(result->dual_solution, result->dual_solution + lp.num_row_,
            highs_solution.row_dual.begin());
  highs_solution.value_valid = true;
  highs_solution.dual_valid = true;

  highs_info.pdlp_iteration_count = result->total_count;
  highs_info.objective_function_value = result->primal_objective_value;

  switch (result->termination_reason) {
    case TERMINATION_REASON_OPTIMAL:
      model_status = HighsModelStatus::kOptimal;
      break;
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
      model_status = HighsModelStatus::kInfeasible;
      break;
    case TERMINATION_REASON_DUAL_INFEASIBLE:
      model_status = HighsModelStatus::kUnbounded;
      break;
    case TERMINATION_REASON_TIME_LIMIT:
      model_status = HighsModelStatus::kTimeLimit;
      break;
    case TERMINATION_REASON_ITERATION_LIMIT:
      model_status = HighsModelStatus::kIterationLimit;
      break;
    case TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED:
      model_status = HighsModelStatus::kUnboundedOrInfeasible;
      break;
    default:
      model_status = HighsModelStatus::kUnknown;
  }

  cupdlpx_result_free(result);
  lp_problem_free(prob);
  highs_basis.valid = false;
  return HighsStatus::kOk;
}

#endif
