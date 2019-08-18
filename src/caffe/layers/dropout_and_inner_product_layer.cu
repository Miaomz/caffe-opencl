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
void DropoutAndInnerProductLayer<Dtype, MItype, MOtype>::Forward_gpu(
                                           const vector<Blob<MItype>*>& bottom,
                                           const vector<Blob<MOtype>*>& top) {
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> top_data = top[0]->mutable_gpu_data();
  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();

  if (this->phase_ == TRAIN) {
    const int_tp count = bottom[0]->count();//count is the number of elements in bottom[0]
    vptr<uint8_t> mask = rand_vec_.mutable_gpu_data();
    this->device_->rng_bernoulli(count, 1 - threshold_, mask);
    //TODO mask the input in Kernel method

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
                            mask, scale_, DROPOUT_K,
                            nullptr,
                            &(this->bottom_quants_[0]->out_quantizer_values()),
                            &(this->blobs_quants_[0]->out_quantizer_values()),
                            nullptr,
                            &(this->top_quants_[0]->in_quantizer_values()));
      if (bias_term_)
        this->device_->template gemm_dropout<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                            Dtype(1), bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), Dtype(1), top_data,
                            mask, scale_, DROPOUT_K,
                            nullptr, &bias_multiplier_qv_,
                            &(this->blobs_quants_[1]->out_quantizer_values()),
                            nullptr,
                            &(this->top_quants_[0]->in_quantizer_values()));
    } else {// No optimization is available
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
    } else if (this->type_ == DROPOUT_K){
      vptr<const uint8_t> mask = rand_vec_.gpu_data();
      if (transpose_) {
        this->device_->template gemm_dropout<Dtype>(CblasNoTrans, CblasTrans,
                              M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom[0]->mutable_gpu_diff(),
                              mask, scale_, DROPOUT_N);
      } else {
        this->device_->template gemm_dropout<Dtype>(CblasNoTrans, CblasNoTrans,
                              M_, K_, N_,
                              (Dtype) 1., top_diff, this->blobs_[0]->gpu_data(),
                              (Dtype) 0., bottom[0]->mutable_gpu_diff(),
                              mask, scale_, DROPOUT_N);
      }
    } else if (this->type_ == DROPOUT_MK){
      vptr<const uint8_t> mask = rand_vec_.gpu_data();
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
      //TODO mask the bottom diff in kernel method
    }
  }
}

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
