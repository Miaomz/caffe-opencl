#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/embed_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void EmbedLayer<Dtype, MItype, MOtype>::LayerSetUp(
                                      const vector<Blob<MItype>*>& bottom,
                                      const vector<Blob<MOtype>*>& top) {
  N_ = this->layer_param_.embed_param().num_output();
  CHECK_GT(N_, 0) << "EmbedLayer num_output must be positive.";
  K_ = this->layer_param_.embed_param().input_dim();
  CHECK_GT(K_, 0) << "EmbedLayer input_dim must be positive.";
  bias_term_ = this->layer_param_.embed_param().bias_term();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights --
    // transposed from InnerProductLayer for spatial locality.
    vector<int_tp> weight_shape(2);
    weight_shape[0] = K_;
    weight_shape[1] = N_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape, this->device_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.embed_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int_tp> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape, this->device_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.embed_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void EmbedLayer<Dtype, MItype, MOtype>::Reshape(
                                      const vector<Blob<MItype>*>& bottom,
                                      const vector<Blob<MOtype>*>& top) {
  // Figure out the dimensions
  M_ = bottom[0]->count();
  vector<int_tp> top_shape = bottom[0]->shape();
  top_shape.push_back(N_);
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int_tp> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void EmbedLayer<Dtype, MItype, MOtype>::Forward_cpu(
                                      const vector<Blob<MItype>*>& bottom,
                                      const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int_tp index;
  for (int_tp n = 0; n < M_; ++n) {
    index = static_cast<int_tp>(bottom_data[n]);
    DCHECK_GE(index, 0);
    DCHECK_LT(index, K_);
    DCHECK_EQ(static_cast<Dtype>(index), bottom_data[n]) << "non-integer input";
    caffe_copy(N_, weight + index * N_, top_data + n * N_);
  }
  if (bias_term_) {
    const Dtype* bias = this->blobs_[1]->cpu_data();
    caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, Dtype(1),
        bias_multiplier_.cpu_data(), bias, Dtype(1), top_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void EmbedLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  CHECK(!propagate_down[0]) << "Can't backpropagate to EmbedLayer input.";
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    int_tp index;
    for (int_tp n = 0; n < M_; ++n) {
      index = static_cast<int_tp>(bottom_data[n]);
      DCHECK_GE(index, 0);
      DCHECK_LT(index, K_);
      DCHECK_EQ(static_cast<Dtype>(index), bottom_data[n])
          << "non-integer input";
      caffe_axpy(N_, Dtype(1), top_diff + n * N_, weight_diff + index * N_);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_gemv<Dtype>(CblasTrans, M_, N_, Dtype(1), top_diff,
        bias_multiplier_.cpu_data(), Dtype(1), bias_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(EmbedLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(EmbedLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(EmbedLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(EmbedLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Embed);
REGISTER_LAYER_CLASS_INST(Embed, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Embed, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Embed, (double), (double), (double));

}  // namespace caffe
