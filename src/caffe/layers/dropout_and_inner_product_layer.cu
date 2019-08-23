#include <algorithm>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dropout_and_inner_product_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndInnerProductLayer<Dtype, MItype, MOtype>::GenerateProgram() {
  this->device_program_ = this->device_->CreateProgram();
  stringstream ss;

  ss << this->device_program_->setup();
  ss << this->device_program_->template define_type<Dtype>("Dtype");
  ss << this->device_program_->template define_type<MItype>("MItype");
  ss << this->device_program_->template define_type<MOtype>("MOtype");

  KernelArgs fw_args;
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "in", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint8_t>(
                    "mask", KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint8_t>(
                    "threshold", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "scale", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("PreDropoutForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  string indexToMask;
  switch (this->type_){
    case DROPOUT_K: indexToMask = "index % " + this->K_; break;
    case DROPOUT_MK: indexToMask = "index"; break;
    default: NOT_IMPLEMENTED;
  }
  ss << "mask[" << indexToMask << "] = (uint8_t)(mask[" << indexToMask <<  "] >= threshold);"  << std::endl;
  ss << "out[index] = in[index] * scale * (Dtype)(mask[" << indexToMask << "]);" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "in_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint8_t>(
                    "mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "scale", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "out_diff", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("PreDropoutBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "out_diff[index] = in_diff[index] * scale * (Dtype)(mask[index";
  switch (this->type_){
    case DROPOUT_K: ss << " % " << this->K_; break;
    case DROPOUT_MK: break;
    default: NOT_IMPLEMENTED;
  }
  ss << "]);" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndInnerProductLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                           const vector<Blob<MItype>*>& bottom,
                                           const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();

  if (this->phase_ == TRAIN) {
    const int_tp count = bottom[0]->count();
    vptr<uint8_t> mask = rand_vec_.mutable_gpu_data();
    this->device_->rng_uniform(rand_vec_.count(), mask); 
    
    shared_ptr<DeviceKernel> kernel = this->device_program_->GetKernel("PreDropoutForward");
    kernel->add_arg(&count);
    kernel->add_arg(&bottom_data);
    kernel->add_arg(&mask);
    kernel->add_arg(&uint_thres_);
    kernel->add_arg(&scale_);
    kernel->add_arg(&bottom_data);
    vector<size_t> work_size(1, count);
    vector<size_t> group;
    vector<size_t> local;
    this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    kernel->Execute(group, local);

    if (M_ == 1) {// No value to optimize
      this->device_->template gemv<Dtype>(CblasNoTrans, N_, K_, Dtype(1),
                               weight, bottom_data, Dtype(0), top_data,
                               nullptr,
                               &(this->blobs_quants_[0]->out_quantizer_values()),
                               &(this->bottom_quants_[0]->out_quantizer_values()),
                               nullptr,
                               &(this->top_quants_[0]->in_quantizer_values()));
      if (bias_term_)
        this->device_->template axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                               this->blobs_[1]->gpu_data(), top_data,
                               &bias_multiplier_qv_,
                               &(this->blobs_quants_[1]->out_quantizer_values()),
                               &(this->top_quants_[0]->in_quantizer_values()));
    } else if (type_ == DROPOUT_K){
      this->device_->template gemm_dropout<Dtype>(CblasNoTrans,
                            transpose_ ? CblasNoTrans : CblasTrans,
                            M_, N_, K_, Dtype(1),
                            bottom_data, weight, Dtype(0), top_data,
                            mask, 1.0, DROPOUT_K,
                            nullptr,
                            &(this->bottom_quants_[0]->out_quantizer_values()),
                            &(this->blobs_quants_[0]->out_quantizer_values()),
                            nullptr,
                            &(this->top_quants_[0]->in_quantizer_values()));
      if (bias_term_)
        this->device_->template gemm_dropout<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                            Dtype(1), bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), Dtype(1), top_data,
                            mask, 1.0, DROPOUT_K,
                            nullptr, &bias_multiplier_qv_,
                            &(this->blobs_quants_[1]->out_quantizer_values()),
                            nullptr,
                            &(this->top_quants_[0]->in_quantizer_values()));
    } else if (type_ == DROPOUT_MK) {// No optimization is available
        this->device_->template gemm<Dtype>(CblasNoTrans,
                            transpose_ ? CblasNoTrans : CblasTrans,
                            M_, N_, K_, Dtype(1),
                            bottom_data, weight, Dtype(0), top_data,
                            nullptr,
                            &(this->bottom_quants_[0]->out_quantizer_values()),
                            &(this->blobs_quants_[0]->out_quantizer_values()),
                            nullptr,
                            &(this->top_quants_[0]->in_quantizer_values()));
        if (bias_term_)
          this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                            Dtype(1), bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), Dtype(1), top_data,
                            nullptr, &bias_multiplier_qv_,
                            &(this->blobs_quants_[1]->out_quantizer_values()),
                            nullptr,
                            &(this->top_quants_[0]->in_quantizer_values()));
    } else {
      NOT_IMPLEMENTED;
    }
  } else {//otherwise, in TEST phase, it is just inner_product
    if (M_ == 1) {
      this->device_->template gemv<Dtype>(CblasNoTrans, N_, K_, Dtype(1),
                             weight, bottom_data, Dtype(0), top_data,
                             nullptr,
                             &(this->blobs_quants_[0]->out_quantizer_values()),
                             &(this->bottom_quants_[0]->out_quantizer_values()),
                             nullptr,
                             &(this->top_quants_[0]->in_quantizer_values()));
      if (bias_term_)
        this->device_->template axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                             this->blobs_[1]->gpu_data(), top_data,
                             &bias_multiplier_qv_,
                             &(this->blobs_quants_[1]->out_quantizer_values()),
                             &(this->top_quants_[0]->in_quantizer_values()));
    } else {
      this->device_->template gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, Dtype(1),
                          bottom_data, weight, Dtype(0), top_data,
                          nullptr,
                          &(this->bottom_quants_[0]->out_quantizer_values()),
                          &(this->blobs_quants_[0]->out_quantizer_values()),
                          nullptr,
                          &(this->top_quants_[0]->in_quantizer_values()));
      if (bias_term_)
        this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                          Dtype(1), bias_multiplier_.gpu_data(),
                          this->blobs_[1]->gpu_data(), Dtype(1), top_data,
                          nullptr, &bias_multiplier_qv_,
                          &(this->blobs_quants_[1]->out_quantizer_values()),
                          nullptr,
                          &(this->top_quants_[0]->in_quantizer_values()));
    }
  }
}


template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndInnerProductLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {

  if (this->param_propagate_down_[0]) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    // Bottom neurons should have been dropped in forward pass
    vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      this->device_->template gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                                          (Dtype) 1.,
                                          bottom_data, top_diff, (Dtype) 1.,
                                          this->blobs_[0]->mutable_gpu_diff());
    } else {
      this->device_->template gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_,
                                          (Dtype) 1.,
                                          top_diff, bottom_data, (Dtype) 1.,
                                          this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    this->device_->template gemv<Dtype>(CblasTrans, M_, N_, (Dtype) 1.,
                              top_diff, bias_multiplier_.gpu_data(), (Dtype) 1.,
                              this->blobs_[1]->mutable_gpu_diff());
  }

  if (propagate_down[0]) {
    vptr<const Dtype> top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (this->phase_ == TEST){
      if (transpose_) {
        this->device_->template gemm<Dtype>(CblasNoTrans, CblasTrans,
                              M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom[0]->mutable_gpu_diff());
      } else {
        this->device_->template gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                              M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom[0]->mutable_gpu_diff());
      }
    } else {
      vptr<const uint8_t> mask = rand_vec_.gpu_data();
      vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
      if (this->type_ == DROPOUT_K){
        if (transpose_) {
          this->device_->template gemm_dropout<Dtype>(CblasNoTrans, CblasTrans,
                              M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom_diff,
                              mask, scale_, DROPOUT_N);
        } else {
          this->device_->template gemm_dropout<Dtype>(CblasNoTrans, CblasNoTrans,
                              M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom_diff,
                              mask, scale_, DROPOUT_N);
        }
      } else if (this->type_ == DROPOUT_MK){
        if (transpose_) {
          this->device_->template gemm_dropout<Dtype>(CblasNoTrans, CblasTrans,
                              M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom_diff,
                              mask, 1.0, DROPOUT_MN);
        } else {
          this->device_->template gemm_dropout<Dtype>(CblasNoTrans, CblasNoTrans,
                              M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom_diff,
                              mask, 1.0, DROPOUT_MN);
        }
      } else {
        NOT_IMPLEMENTED;
      }

      // this is needed, otherwise the accuracy will be lower
      const int_tp count = bottom[0]->count();
      shared_ptr<DeviceKernel> kernel = this->device_program_->GetKernel("PreDropoutBackward");
      kernel->add_arg(&count);
      kernel->add_arg(&bottom_diff);
      kernel->add_arg(&mask);
      kernel->add_arg(&scale_);
      kernel->add_arg(&bottom_diff);
      vector<size_t> work_size(1, count);
      vector<size_t> group;
      vector<size_t> local;
      this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
      kernel->Execute(group, local);
    }
  }
}

INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, GenerateProgram,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, GenerateProgram,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, GenerateProgram,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, GenerateProgram,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, GenerateProgram,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, GenerateProgram,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, GenerateProgram,
                                  (uint64_t), (uint64_t), (uint64_t));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Forward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Forward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Forward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Forward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Forward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Forward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Forward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));

INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Backward_gpu,
                                  (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Backward_gpu,
                                  (float), (float), (float));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Backward_gpu,
                                  (double), (double), (double));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Backward_gpu,
                                  (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Backward_gpu,
                                  (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Backward_gpu,
                                  (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASST_FUNC_3T_GUARDED(DropoutAndInnerProductLayer, Backward_gpu,
                                  (uint64_t), (uint64_t), (uint64_t));

}  // namespace caffe
