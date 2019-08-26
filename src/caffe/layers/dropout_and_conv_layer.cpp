#include <algorithm>
#include <vector>

#ifdef USE_LIBDNN

#include "caffe/filler.hpp"
#include "caffe/layers/dropout_and_conv_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndConvolutionLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  this->use_colbuffer_ = false;
  // Configure the kernel size, padding, stride, and inputs.
  DropoutAndConvolutionParameter conv_param = this->layer_param_.dropout_and_convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int_tp first_spatial_axis = channel_axis_ + 1;
  const int_tp num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int_tp> spatial_dim_blob_shape(
      1, std::max(num_spatial_axes_, (int_tp) 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int_tp* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
      << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
      << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int_tp num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
    << "kernel_size must be specified once, or once per spatial dimension "
    << "(kernel_size specified " << num_kernel_dims << " times; "
    << num_spatial_axes_ << " spatial dims);";
    for (int_tp i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] =
      conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
    }
  }
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0)<< "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int_tp* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
      << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
      << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int_tp num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
        num_stride_dims == num_spatial_axes_)
    << "stride must be specified once, or once per spatial dimension "
    << "(stride specified " << num_stride_dims << " times; "
    << num_spatial_axes_ << " spatial dims);";
    const int_tp kDefaultStride = 1;
    for (int_tp i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
      conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int_tp* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
      << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
      << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int_tp num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
        num_pad_dims == num_spatial_axes_)
    << "pad must be specified once, or once per spatial dimension "
    << "(pad specified " << num_pad_dims << " times; "
    << num_spatial_axes_ << " spatial dims);";
    const int_tp kDefaultPad = 0;
    for (int_tp i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
      conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }

  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int_tp* dilation_data = dilation_.mutable_cpu_data();
  const int_tp num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }

  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &= kernel_shape_data[i] == 1 && stride_data[i] == 1
        && pad_data[i] == 0;
    if (!is_1x1_) {
      break;
    }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = conv_param.num_output();
  CHECK_GT(num_output_, 0);
  group_ = conv_param.group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
    << "Number of output should be multiples of group.";
  if (this->deconvolution_) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int_tp> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = conv_param.bias_term();
  vector<int_tp> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape, this->device_);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
      << weight_shaped_blob.shape_string() << "; instead, shape was "
      << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape, this->device_);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
      << bias_shaped_blob.shape_string() << "; instead, shape was "
      << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels X input channels per-group X kernel height X kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape, this->device_));
    if (is_float_type<Dtype>()) {
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
              conv_param.weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());
    }
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape, this->device_));
      if (is_float_type<Dtype>()) {
        shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                conv_param.bias_filler()));
        bias_filler->Fill(this->blobs_[1].get());
      }
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // Prepare the attributes for dropout
  threshold_ = conv_param.dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<uint_tp>(static_cast<long double>(std::numeric_limits<uint_tp>::max())
                                     * static_cast<long double>(threshold_));

  this->InitializeQuantizers(bottom, top);
  Reshape(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndConvolutionLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  this->use_colbuffer_ = false;

  const int_tp first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
    << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
    << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int_tp bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "shape mismatch - bottom[0]: " << bottom[0]->shape_string()
        << " vs. bottom[" << bottom_id << "]: "
        << bottom[bottom_id]->shape_string();
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  this->compute_output_shape();
  vector<int_tp> top_shape(bottom[0]->shape().begin(),
                        bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int_tp top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (this->deconvolution_) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int_tp> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int_tp* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int_tp i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (this->deconvolution_) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.

  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int_tp i = 0; i < num_spatial_axes_; ++i) {
    if (this->deconvolution_) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }

  col_buffer_.Reshape(col_buffer_shape_);

  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = this->deconvolution_ ? top_dim_ : bottom_dim_;

  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int_tp> bias_multiplier_shape(1, out_spatial_dim_);
    bool reshaped = bias_multiplier_.Reshape(bias_multiplier_shape);
    // This will trigger a memory copy if in GPU mode,
    // which may not be necessary.
    // Thus omit to set the values if not necessary.
    if (reshaped) {
      caffe_set(bias_multiplier_.count(), Dtype(1),
                bias_multiplier_.mutable_cpu_data());
    }
    bias_multiplier_qv_.scale = 1.0;
    bias_multiplier_qv_.zero = 0.0;
    bias_multiplier_qv_.one = 1.0;
    bias_multiplier_qv_.max = 1.0;
    bias_multiplier_qv_.min = 0.0;
  }

  bool shapes_changed = false;
  if (libdnn_.get() != nullptr) {
    vector<int_tp> libdnn_in_sh = libdnn_.get()->get_config().in_shape;
    vector<int_tp> libdnn_out_sh = libdnn_.get()->get_config().out_shape;
    const vector<int_tp>& new_in_sh = bottom[0]->shape();
    const vector<int_tp>& new_out_sh = top[0]->shape();
    bool in_eq = libdnn_in_sh.size() == new_in_sh.size()
                 && libdnn_in_sh[0] >= new_in_sh[0] 
                 && std::equal(libdnn_in_sh.begin() + 1,
                               libdnn_in_sh.end(), new_in_sh.begin() + 1);
    bool out_eq = libdnn_out_sh.size() == new_out_sh.size()
                 && libdnn_out_sh[0] >= new_out_sh[0] 
                 && std::equal(libdnn_out_sh.begin() + 1,
                               libdnn_out_sh.end(),new_out_sh.begin() + 1);
    shapes_changed = !in_eq || !out_eq;
  }

  if (libdnn_.get() == nullptr || shapes_changed) {
    int_tp* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
    int_tp* pad_data = this->pad_.mutable_cpu_data();
    int_tp* stride_data = this->stride_.mutable_cpu_data();
    int_tp* dilation_data = this->dilation_.mutable_cpu_data();

    vector<int_tp> kernel_vec;
    vector<int_tp> pad_vec;
    vector<int_tp> stride_vec;
    vector<int_tp> dilation_vec;

    for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
        kernel_vec.push_back(kernel_shape_data[i]);
        pad_vec.push_back(pad_data[i]);
        stride_vec.push_back(stride_data[i]);
        dilation_vec.push_back(dilation_data[i]);
    }

    LibDNNConvConfig config;
    config.dev_ptr = this->device_;
    config.in_shape = bottom[0]->shape();
    config.out_shape = top[0]->shape();
    config.kernel = kernel_vec;
    config.pad = pad_vec;
    config.stride = stride_vec;
    config.dilation = dilation_vec;
    config.group = this->group_;
    config.bias_term = this->bias_term_;
    config.fast_unsafe_math = true;
    config.weights_backward = this->param_propagate_down_[0];
    config.bias_backward = this->param_propagate_down_[1];

    if ((std::is_same<Dtype, float>::value
        && (this->device_->CheckCapability(
              DEVICE_INT64_GLOBAL_ATOMICS_SUPPORT) ||
            this->device_->CheckCapability(
             DEVICE_INT64_GLOBAL_EXTENDED_ATOMICS_SUPPORT))) ||
        (std::is_same<Dtype, double>::value
        && (this->device_->CheckCapability(
            DEVICE_INT64_GLOBAL_ATOMICS_SUPPORT) ||
            this->device_->CheckCapability(
                DEVICE_INT64_GLOBAL_EXTENDED_ATOMICS_SUPPORT)))) {
      config.wgalgo = LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;
      config.bwalgo = LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC;
    } else {
      config.wgalgo = LIBDNN_CONVOLUTION_WG_ALGO_DIRECT;
      config.bwalgo = LIBDNN_CONVOLUTION_BW_ALGO_IM2COL;
    }

    LibDNNConv<MItype, MOtype>* libdnn =
        new LibDNNConv<MItype, MOtype>(config);

    libdnn_.reset(libdnn);
  }

  // Reshape the rand_vec_ and generates the program
  rand_vec_.Reshape(bottom[0]->shape());
  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
DropoutAndConvolutionLayer<Dtype, MItype, MOtype>::~DropoutAndConvolutionLayer() {
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndConvolutionLayer<Dtype, MItype, MOtype>::Forward_gpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {

  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();
  vptr<const Dtype> bias;
  Dtype bias_mult;
  if (this->bias_term_) {
     bias_mult = this->bias_multiplier_.cpu_data()[0];
     bias = this->blobs_[1]->gpu_data();
  }

  for (int_tp i = 0; i < bottom.size(); ++i) {
    const QuantizerValues* const bottom_quant =
        &(this->bottom_quants_[i]->out_quantizer_values());
    const QuantizerValues* const weight_quant =
        this->blobs_quants_.size() > 0 ?
            &(this->blobs_quants_[0]->out_quantizer_values()) : nullptr;
    const QuantizerValues* const bias_quant =
        this->blobs_quants_.size() > 1 ?
            &(this->blobs_quants_[1]->out_quantizer_values()) : nullptr;
    const QuantizerValues* const top_quant =
        &(this->top_quants_[i]->in_quantizer_values());

    vptr<MItype> bottom_data = bottom[i]->mutable_gpu_data();
    const int_tp count = bottom[0]->count();
    if (this->phase_ == TRAIN) {
      vptr<uint_tp> mask = rand_vec_.mutable_gpu_data();
      this->device_->rng_uniform(count, mask);
      shared_ptr<DeviceKernel> kernel = this->device_program_->GetKernel("DropoutForward");
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
    }

    vptr<MOtype> top_data = top[i]->mutable_gpu_data();
    libdnn_.get()->Forward(bottom_data, weight, bias_mult,
                           bias, top_data, bottom[i]->shape()[0],
                           bottom_quant, weight_quant,
                           &(this->bias_multiplier_qv_),
                           bias_quant, top_quant);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndConvolutionLayer<Dtype, MItype, MOtype>::Backward_gpu(
    const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {

  vptr<const Dtype> weight = this->blobs_[0]->gpu_data();
  vptr<const Dtype> bias;
  vptr<Dtype> weight_diff = this->blobs_[0]->mutable_gpu_diff();
  vptr<Dtype> bias_diff;
  Dtype bias_mult;
  if (this->bias_term_) {
     bias = this->blobs_[1]->gpu_data();
     bias_diff = this->blobs_[1]->mutable_gpu_diff();
     bias_mult = this->bias_multiplier_.cpu_data()[0];
  }

  vptr<const Dtype> top_data = top[0]->gpu_data();
  vptr<const Dtype> top_diff = top[0]->gpu_diff();
  vptr<const Dtype> bottom_data = bottom[0]->gpu_data();
  vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
  vptr<const uint_tp> mask = rand_vec_.gpu_data();
  if (this->phase_ == TRAIN) {
    libdnn_.get()->BackwardDropout(propagate_down[0],
                                   propagate_down[0]||this->param_propagate_down_[0]||this->param_propagate_down_[1],
                                   top_data, top_diff,
                                   weight, weight_diff,
                                   bias_mult, bias, bias_diff,
                                   bottom_data, bottom_diff,
                                   bottom[0]->shape()[0],
                                   mask, this->uint_thres_, this->scale_);
  } else {
    libdnn_.get()->Backward(propagate_down[0],
                            propagate_down[0] ||
                            (this->param_propagate_down_[0] ||
                             this->param_propagate_down_[1]),
                            top_data, top_diff,
                            weight, weight_diff,
                            bias_mult, bias, bias_diff,
                            bottom_data, bottom_diff,
                            bottom[0]->shape()[0]);
  }

  for (int_tp i = 1; i < top.size(); ++i) {
    vptr<const Dtype> top_data = top[i]->gpu_data();
    vptr<const Dtype> top_diff = top[i]->gpu_diff();
    vptr<const Dtype> bottom_data = bottom[i]->gpu_data();
    vptr<Dtype> bottom_diff = bottom[i]->mutable_gpu_diff();
    libdnn_.get()->Backward(propagate_down[i], propagate_down[i] ||
                            (this->param_propagate_down_[0] ||
                             this->param_propagate_down_[1]),
                            top_data, top_diff,
                            weight, weight_diff,
                            bias_mult, bias, bias_diff,
                            bottom_data, bottom_diff,
                            bottom[i]->shape()[0]);
  }

  if (propagate_down[0]) {
    vptr<Dtype> bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      vptr<const uint_tp> mask = rand_vec_.gpu_data();
      const int_tp count = bottom[0]->count();
      shared_ptr<DeviceKernel> kernel = this->device_program_->GetKernel("DropoutBackward");
      kernel->add_arg(&count);
      kernel->add_arg(&bottom_diff);
      kernel->add_arg(&mask);
      kernel->add_arg(&uint_thres_);
      kernel->add_arg(&scale_);
      kernel->add_arg(&bottom_diff);

      vector<size_t> work_size(1, count);
      vector<size_t> group;
      vector<size_t> local;
      this->device_->get_threads(&work_size, &group, &local, kernel.get(), true);
    } 
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndConvolutionLayer<Dtype, MItype, MOtype>::Tune(
          vptr<Dtype> top_data, vptr<Dtype> top_diff,
          vptr<Dtype> bottom_data, vptr<Dtype> bottom_diff,
          int_tp batch_size) {
  vptr<Dtype> weight_data = this->blobs_[0]->mutable_gpu_data();
  vptr<Dtype> weight_diff = this->blobs_[0]->mutable_gpu_diff();
  vptr<Dtype> bias_data;
  vptr<Dtype> bias_diff;
  Dtype bias_mult;
  if (this->bias_term_) {
     bias_data = this->blobs_[1]->mutable_gpu_data();
     bias_diff = this->blobs_[1]->mutable_gpu_diff();
     bias_mult = this->bias_multiplier_.cpu_data()[0];
  }

  libdnn_.get()->Tune(top_data, top_diff,
                      weight_data, weight_diff,
                      bias_mult, bias_data, bias_diff,
                      bottom_data, bottom_diff,
                      batch_size);
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndConvolutionLayer<Dtype, MItype, MOtype>::compute_output_shape() {
  const int_tp* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int_tp* stride_data = this->stride_.cpu_data();
  const int_tp* pad_data = this->pad_.cpu_data();
  const int_tp* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  if (this->deconvolution_) {
    for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
      // i + 1 to skip channel axis
      const int_tp input_dim = this->input_shape(i + 1);
      const int_tp kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1)
          + 1;
      const int_tp output_dim = stride_data[i] * (input_dim - 1)
          + kernel_extent - 2 * pad_data[i];
      this->output_shape_.push_back(output_dim);
    }
  } else {
    for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
      // i + 1 to skip channel axis
      const int_tp input_dim = this->input_shape(i + 1);
      const int_tp kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1)
          + 1;
      const int_tp output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
          / stride_data[i] + 1;
      this->output_shape_.push_back(output_dim);
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndConvolutionLayer<Dtype, MItype, MOtype>::GenerateProgram() {
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
  fw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "threshold", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "scale", KERNEL_ARG_CONST));
  fw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "out", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("DropoutForward", fw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "out[index] = in[index] * (Dtype)(mask[index] > threshold) * scale;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  KernelArgs bw_args;
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "n", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MOtype>(
                    "in_diff", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "mask", KERNEL_ARG_CONST | KERNEL_ARG_GLOBAL_MEM));
  bw_args.push_back(this->device_program_->template create_kernel_arg<uint_tp>(
                    "threshold", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<Dtype>(
                    "scale", KERNEL_ARG_CONST));
  bw_args.push_back(this->device_program_->template create_kernel_arg<MItype>(
                    "out_diff", KERNEL_ARG_GLOBAL_MEM));
  ss << this->device_program_->function("DropoutBackward", bw_args);
  ss << this->device_program_->kernel_loop("uint_tp", "index", "n");
  ss << "out_diff[index] = in_diff[index] * scale *"
     << " (Dtype)(mask[index] > threshold);" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  this->device_program_->set_source(ss.str());
  this->device_program_->Compile(true, true);
}


#ifndef CPU ONLY

template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Blob<Dtype> >
                     DropoutAndConvolutionLayer<Dtype, MItype, MOtype>::col_buffer() {
  if (col_buffer_lock_id_ == -1) {
    shared_col_buffer_ = this->device_->template Buffer<Dtype>(
                                       col_buffer_shape_, &col_buffer_lock_id_);
  }
  return shared_col_buffer_;
}

template<typename Dtype, typename MItype, typename MOtype>
void DropoutAndConvolutionLayer<Dtype, MItype, MOtype>::unlock_col_buffer() {
  if (col_buffer_lock_id_ != -1) {
    shared_col_buffer_ = nullptr;
    this->device_->unlock_buffer(&col_buffer_lock_id_);
  }
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS_3T_GUARDED(DropoutAndConvolutionLayer, (half_fp), (half_fp),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndConvolutionLayer, (float), (float),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndConvolutionLayer, (double), (double),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndConvolutionLayer, (uint8_t), (uint8_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndConvolutionLayer, (uint16_t), (uint16_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndConvolutionLayer, (uint32_t), (uint32_t),
                             PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(DropoutAndConvolutionLayer, (uint64_t), (uint64_t),
                             PROTO_TYPES);

REGISTER_LAYER_CLASS(DropoutAndConvolution);
REGISTER_LAYER_CLASS_INST(DropoutAndConvolution, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(DropoutAndConvolution, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(DropoutAndConvolution, (double), (double), (double));
REGISTER_LAYER_CLASS_INST(DropoutAndConvolution, (uint8_t), (uint8_t), (uint8_t));
REGISTER_LAYER_CLASS_INST(DropoutAndConvolution, (uint16_t), (uint16_t), (uint16_t));
REGISTER_LAYER_CLASS_INST(DropoutAndConvolution, (uint32_t), (uint32_t), (uint32_t));
REGISTER_LAYER_CLASS_INST(DropoutAndConvolution, (uint64_t), (uint64_t), (uint64_t));

}   // namespace caffe
#endif  // USE_LIBDNN

