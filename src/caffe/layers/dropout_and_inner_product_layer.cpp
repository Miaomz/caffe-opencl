#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/dropout_and_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndInnerProductLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const int_tp num_output =
      this->layer_param_.dropout_and_inner_product_param().num_output();
  bias_term_ = this->layer_param_.dropout_and_inner_product_param().bias_term();
  transpose_ = this->layer_param_.dropout_and_inner_product_param().transpose();
  N_ = num_output;
  const int_tp axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.dropout_and_inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, n inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO)<< "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int_tp> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape, this->device_));
    // fill the weights (for float types only)
    if (is_float_type<Dtype>()) {
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.dropout_and_inner_product_param().weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
    }
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int_tp> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape, this->device_));
      if (is_float_type<Dtype>()) {
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                this->layer_param_.dropout_and_inner_product_param().bias_filler()));
        bias_filler->Fill(this->blobs_[1].get());
      }
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  //Dropout part is as follows
  threshold_ = this->layer_param_.dropout_and_inner_product_param().dropout_ratio();
  type_ = (DropoutType)this->layer_param_.dropout_and_inner_product_param().dropout_type();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);

  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndInnerProductLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  // Figure out the dimensions
  const int_tp axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.dropout_and_inner_product_param().axis());
  const int_tp new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int_tp> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int_tp> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    bias_multiplier_qv_.scale = 1.0;
    bias_multiplier_qv_.zero = 0.0;
    bias_multiplier_qv_.one = 1.0;
    bias_multiplier_qv_.max = 1.0;
    bias_multiplier_qv_.min = 0.0;
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

  // Dropout part below. Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  vector<int_tp> rand_shape;
  switch(this->type_) {
    case DROPOUT_K:
      rand_shape.push_back(K_);
      rand_vec_.Reshape(rand_shape);
      break;
    case DROPOUT_MK:
      rand_vec_.Reshape(bottom[0]->shape());
      break;
    default: LOG(INFO)<< "Unknown or Illegal value of DropoutType!";
  }

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}


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
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "scale", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("PreDropoutForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "out[index] = in[index] * (Dtype)(mask[index";
  switch (this->type_){
    case DROPOUT_K: ss << " % " << this->K_ ; break;
    case DROPOUT_MK: break;
    default: NOT_IMPLEMENTED;
  }
  ss << "]) * scale;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "in_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
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
void DropoutAndInnerProductLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  // It seems that we do need to MODIFY the bottom data
  int_tp count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    uint8_t* mask = rand_vec_.mutable_cpu_data();
    caffe_rng_bernoulli(rand_vec_.count(), 1. - threshold_, mask); 
    if (this->type_ == DROPOUT_K){
      for (int_tp i = 0; i < count; ++i)
        bottom_data[i] = bottom_data[i] * mask[i % K_] * scale_;
    } else if (this->type_ == DROPOUT_MK) {
      for (int_tp i = 0; i < count; ++i)
        bottom_data[i] = bottom_data[i] * mask[i] * scale_;
    } else {
      LOG(INFO) << "Unknown or illegal Dropout type";
    }
  }

  caffe_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, Dtype(1), bottom_data, weight, Dtype(0), top_data,
      nullptr, &(this->bottom_quants_[0]->out_quantizer_values()),
      &(this->blobs_quants_[0]->out_quantizer_values()),
      nullptr, &(this->top_quants_[0]->in_quantizer_values()));

  if (bias_term_)
    caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, Dtype(1),
                      bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(),
                      Dtype(1), top_data,
                      nullptr, &bias_multiplier_qv_,
                      &(this->blobs_quants_[1]->out_quantizer_values()),
                      nullptr, &(this->top_quants_[0]->in_quantizer_values()));
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndInnerProductLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_gemv<Dtype>(CblasTrans, M_, N_, (Dtype) 1., top_diff,
                          bias_multiplier_.cpu_data(), (Dtype) 1.,
                          this->blobs_[1]->mutable_cpu_diff());
  }

  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom_diff);
    } else {
      caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom_diff);
    }

    if (this->phase_ == TRAIN) {
      const uint8_t* mask = rand_vec_.cpu_data();
      const int_tp count = bottom[0]->count();
      if (this->type_ == DROPOUT_K){
        for (int_tp i = 0; i < count; ++i)
          bottom_diff[i] = bottom_diff[i] * mask[i % K_] * scale_;
      } else if (this->type_ == DROPOUT_MK) {
        for (int_tp i = 0; i < count; ++i)
          bottom_diff[i] = bottom_diff[i] * mask[i] * scale_;
      } else {
        LOG(INFO) << "Unknown or illegal Dropout type";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DropoutAndInnerProductLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(DropoutAndInnerProductLayer,
                             (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndInnerProductLayer,
                             (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndInnerProductLayer,
                             (double), (double), (double));
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndInnerProductLayer,
                             (uint8_t), (uint8_t), (uint8_t));
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndInnerProductLayer,
                             (uint16_t), (uint16_t), (uint16_t));
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndInnerProductLayer,
                             (uint32_t), (uint32_t), (uint32_t));
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndInnerProductLayer,
                             (uint64_t), (uint64_t), (uint64_t));

REGISTER_LAYER_CLASS(DropoutAndInnerProduct);
REGISTER_LAYER_CLASS_INST(DropoutAndInnerProduct, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(DropoutAndInnerProduct, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(DropoutAndInnerProduct, (double), (double), (double));
REGISTER_LAYER_CLASS_INST(DropoutAndInnerProduct, (uint8_t), (uint8_t), (uint8_t));
REGISTER_LAYER_CLASS_INST(DropoutAndInnerProduct, (uint16_t), (uint16_t), (uint16_t));
REGISTER_LAYER_CLASS_INST(DropoutAndInnerProduct, (uint32_t), (uint32_t), (uint32_t));
REGISTER_LAYER_CLASS_INST(DropoutAndInnerProduct, (uint64_t), (uint64_t), (uint64_t));

}  // namespace caffe

