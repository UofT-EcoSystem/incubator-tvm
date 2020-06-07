/*!
 * Copyright (c) 2020 by Contributors
 * \file ansor/search_task.h
 * \brief Distributed measurement infrastructure to measure the runtime costs of tensor programs
 */

#ifndef TVM_ANSOR_MEASURE_H_
#define TVM_ANSOR_MEASURE_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include "search_task.h"
#include "loop_state.h"

namespace tvm {
namespace ansor {

class SearchPolicy;
class MeasureInput; class BuildResult; class MeasureResult;
class Builder; class Runner; class MeasureCallback; class ProgramMeasurer;

/* \brief The error code of one measurement */
enum MeasureErrorNO {
  kNoError = 0,              // No error
  kInstantiationError = 1,   // Errors happen when apply transform steps from init state
  kCompileHostError = 2,     // Errors happen when compiling code on host (when build module)
  kCompileDeviceError = 3,   // Errors happen when compiling code on device (when load module)
  kRuntimeDeviceError = 4,   // Errors happen when run program on device
  kWrongAnswerError = 5,     // Answer is wrong when compared to a reference output
  kBuildTimeoutError = 6,    // Timeout during compilation
  kRunTimeoutError = 7,      // Timeout during run
  kUnknonwError = 8,         // Unknown error
};
extern const char *ErrorNoToStr[];

// Inputs and results of one measurement

/* \brief Store the input of a measurement */
class MeasureInputNode: public Object {
 public:
  SearchTask task;   // The search task
  State state;       // The program state to be measured

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("task", &task);
    v->Visit("state", &state);
  }

  static MeasureInput make(SearchTask task, State state);
  MeasureInput copy() const;  // Do deep copy

  static constexpr const char* _type_key = "ansor.MeasureInput";
  TVM_DECLARE_FINAL_OBJECT_INFO(MeasureInputNode, Object);
};
TVM_DEFINE_OBJECT_REF(MeasureInput, MeasureInputNode);

/* \brief Store the input of a build */
class BuildResultNode: public Object {
 public:
  std::string filename;    // The filename of built binary file
  Array<te::Tensor> args;  // The arguments
  int error_no;            // The error code (see MeasureErrorNO). 0 means no error.
  std::string error_msg;   // The error message if there is any error
  double time_cost;        // The time cost of build

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("filename", &filename);
    v->Visit("args", &args);
    v->Visit("error_no", &error_no);
    v->Visit("error_msg", &error_msg);
    v->Visit("time_cost", &time_cost);
  }

  static BuildResult make(std::string filename, Array<te::Tensor> args,
                          int error_no, std::string error_msg, double time_cost);

  static constexpr const char* _type_key = "ansor.BuildResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuildResultNode, Object);
};
TVM_DEFINE_OBJECT_REF(BuildResult, BuildResultNode);

/* \brief Store the results of a measurement */
class MeasureResultNode: public Object {
 public:
  Array<PrimExpr> costs;   // The time costs of execution
  int error_no;            // The error code (see MeasureErrorNO). 0 means no error.
  std::string error_msg;   // The error message if there is any error
  double all_cost;         // The time cost of build and run
  double timestamp;        // The time stamps of this measurement

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("costs", &costs);
    v->Visit("error_no", &error_no);
    v->Visit("error_msg", &error_msg);
    v->Visit("all_cost", &all_cost);
    v->Visit("timestamp", &timestamp);
  }

  MeasureResult copy() const;  // Do deep copy

  static MeasureResult make(Array<PrimExpr> costs, int error_no, std::string error_msg,
                            double all_cost, double timestamp);

  static constexpr const char* _type_key = "ansor.MeasureResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(MeasureResultNode, Object);
};
TVM_DEFINE_OBJECT_REF(MeasureResult, MeasureResultNode);


/* \brief Bass class of measurement callbacks */
class MeasureCallbackNode: public Object {
 public:
  /*! \biref Callback function that will be called on measurement input/result pairs
   * after measurement */
  virtual void callback(const SearchPolicy& policy,
                        const Array<MeasureInput>& inputs,
                        const Array<MeasureResult>& results) = 0;
  static constexpr const char *_type_key = "ansor.MeasureCallback";
  TVM_DECLARE_BASE_OBJECT_INFO(MeasureCallbackNode, Object);
};
TVM_DEFINE_MUTABLE_OBJECT_REF(MeasureCallback, MeasureCallbackNode);


// Base class for builder and runner

/* \brief Builder that builds the programs */
class BuilderNode: public Object {
 public:
  int n_parallel;  // The number of tasks to run in parallel
  int timeout;     // Timeout of a build

  /*! \biref Build programs and return results */
  virtual Array<BuildResult> Build(const Array<MeasureInput>& inputs, int verbose) = 0;

  static constexpr const char* _type_key = "ansor.Builder";
  TVM_DECLARE_BASE_OBJECT_INFO(BuilderNode, Object);
};
TVM_DEFINE_MUTABLE_OBJECT_REF(Builder, BuilderNode);

/* \brief Runner that runs the built programs and measure the time cost */
class RunnerNode: public Object {
 public:
  int timeout;   // Timeout of a run

  /*! \biref Run measurement and return results */
  virtual Array<MeasureResult> Run(const Array<MeasureInput>& inputs,
                                   const Array<BuildResult>& build_results,
                                   int verbose) = 0;

  static constexpr const char* _type_key = "ansor.Runner";
  TVM_DECLARE_BASE_OBJECT_INFO(RunnerNode, Object);
};
TVM_DEFINE_MUTABLE_OBJECT_REF(Runner, RunnerNode);


// Implementation of various builders and runners
/* \brief LocalBuilder use local CPU cores to build programs in parallel */
class LocalBuilderNode: public BuilderNode {
 public:
  std::string build_func;  // Build function

  static Builder make(int timeout, int n_parallel, const std::string& build_func);

  Array<BuildResult> Build(const Array<MeasureInput>& inputs, int verbose) final;

  static constexpr const char* _type_key = "ansor.LocalBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(LocalBuilderNode, BuilderNode);
};

/* \brief RPCRunner that uses RPC call to measures the time cost of programs on remote devices */
class RPCRunnerNode : public RunnerNode {
 public:
  std::string key;
  std::string host;
  int port;
  int priority;
  int n_parallel;
  int number;
  int repeat;
  int min_repeat_ms;
  double cooldown_interval;

  static Runner make(const std::string& key, const std::string& host, int port,
                     int priority, int timeout, int n_parallel, int number,
                     int repeat, int min_repeat_ms, double cooldown_interval);

  /*! \biref Run measurement and return results */
  Array<MeasureResult> Run(const Array<MeasureInput>& inputs,
                           const Array<BuildResult>& build_results,
                           int verbose) final;

  static constexpr const char* _type_key = "ansor.RPCRunner";
  TVM_DECLARE_FINAL_OBJECT_INFO(RPCRunnerNode, RunnerNode);
};

/* \brief LocalRunner that uses local CPU/GPU to measures the time cost of programs */
class LocalRunnerNode: public RunnerNode {
 public:
  int number;
  int repeat;
  int min_repeat_ms;
  double cooldown_interval;

  static Runner make(int timeout, int number, int repeat,
                     int min_repeat_ms, double cooldown_interval);

  /*! \biref Run measurement and return results */
  Array<MeasureResult> Run(const Array<MeasureInput>& inputs,
                           const Array<BuildResult>& build_results,
                           int verbose) final;

  static constexpr const char* _type_key = "ansor.LocalRunner";
  TVM_DECLARE_FINAL_OBJECT_INFO(LocalRunnerNode, RunnerNode);
};


/*!
 * \brief Measurer that measures the time costs of tvm programs
 * This class combines Builder and Runner, and provides a simpler API */
class ProgramMeasurerNode: public Object {
 public:
  static const int DEFAULT_MAX_CONTINOUS_ERROR = 150;

  int ct;
  int error_ct;   // continuous error counter
  std::unordered_map<std::string, double> best_flops;
  std::unordered_map<std::string, State> best_state;
  std::unordered_map<std::string, int> best_ct;

  Builder builder;
  Runner runner;
  Array<MeasureCallback> callbacks;
  int verbose;
  int max_continous_error;

  static ProgramMeasurer make(Builder builder, Runner runner,
                              Array<MeasureCallback> callbacks,
                              int verbose,
                              int max_continous_error = -1);

  /*! \brief Reset book keeping variables */
  void Reset();

  /*! \biref Do measurement */
  void Measure(const SearchTask& task,
               const SearchPolicy& policy,
               const std::vector<MeasureInput>& inputs,
               std::vector<MeasureResult>* results,
               int batch_size = -1);

  /*! \biref Do measurement silently */
  void SilentMeasure(const SearchTask& task,
                     const std::vector<MeasureInput>& inputs,
                     std::vector<MeasureResult>* results);

  static constexpr const char* _type_key = "ansor.ProgramMeasurer";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProgramMeasurerNode, Object);
};
TVM_DEFINE_MUTABLE_OBJECT_REF(ProgramMeasurer, ProgramMeasurerNode);


}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_MEASURE_H_
