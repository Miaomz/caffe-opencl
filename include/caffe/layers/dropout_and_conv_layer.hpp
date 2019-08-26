#ifndef DROPOUT_AND_CONVOLUTION_LAYER_HPP_
#define DROPOUT_AND_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/libdnn/libdnn_conv.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
class DropoutAndConvolutionLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit DropoutAndConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param), col_buffer_lock_id_(-1),
        deconvolution_(false) {
    this->deconvolution_ = false;
  }
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
                          const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
                       const vector<Blob<MOtype>*>& top);
  virtual ~DropoutAndConvolutionLayer();

  virtual inline int_tp MinBottomBlobs() const {
    return 1;
  }
  virtual inline int_tp MinTopBlobs() const {
    return 1;
  }
  virtual inline bool EqualNumBottomTopBlobs() const {
    return true;
  }

  virtual void Tune(vptr<Dtype> top_data, vptr<Dtype> top_diff,
                    vptr<Dtype> bottom_data, vptr<Dtype> bottom_diff,
                    int_tp batch_size);

 protected:
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
                           const vector<Blob<MOtype>*>& top) {
    NOT_IMPLEMENTED;
  }
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<MItype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down, 
      const vector<Blob<MItype>*>& bottom);

  virtual void GenerateProgram(); 
#ifndef CPU_ONLY
  shared_ptr<Blob<Dtype> > col_buffer();
  void unlock_col_buffer();
#endif

  /// @brief The spatial dimensions of the input.
  inline int_tp input_shape(int_tp i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }

  // Compute height_out_ and width_out_ from other parameters.
  void compute_output_shape();


  /// @brief The spatial dimensions of a filter kernel.
  Blob<int_tp> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int_tp> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int_tp> pad_;
  /// @brief The spatial dimensions of the dilation.
  Blob<int_tp> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int_tp> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int_tp> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int_tp> output_shape_;
  const vector<int_tp>* bottom_shape_;

  int_tp num_spatial_axes_;
  int_tp bottom_dim_;
  int_tp top_dim_;

  int_tp channel_axis_;
  int_tp num_;
  int_tp channels_;
  int_tp group_;
  int_tp out_spatial_dim_;
  int_tp weight_offset_;
  int_tp num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;
  bool use_colbuffer_;
  bool deconvolution_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
  QuantizerValues bias_multiplier_qv_;
  shared_ptr<Blob<Dtype> > shared_col_buffer_;
  int_tp col_buffer_lock_id_;

  int_tp conv_out_channels_;
  int_tp conv_in_channels_;
  int_tp conv_out_spatial_dim_;
  int_tp kernel_dim_;
  int_tp col_offset_;
  int_tp output_offset_;
  int_tp num_kernels_im2col_;
  int_tp num_kernels_col2im_;
  bool use_skernel_;

  shared_ptr<LibDNNConv<MItype, MOtype> > libdnn_;

  /// when divided by uint_MAX, the randomly generated values @f$u\sim U(0,1)@f$
  Blob<uint_tp> rand_vec_;
  /// threshold for the GPU mode
  uint_tp uint_thres_;
  /// the probability @f$ p @f$ of dropping any input
  float threshold_;
  /// the scale for undropped inputs at train time @f$ 1 / (1 - p) @f$
  float scale_;
};

}  // namespace caffe

#endif  //DROPOUT_AND_CONVOLUTION_LAYER_HPP_

