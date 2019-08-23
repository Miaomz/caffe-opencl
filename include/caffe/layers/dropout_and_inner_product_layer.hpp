#ifndef CAFFE_DROPOUT_AND_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_DROPOUT_AND_INNER_PRODUCT_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#ifdef USE_OPENCL
#include <boost/filesystem.hpp>
#endif

namespace caffe {

/**
 * @brief A combination of InnerProductLayer and DropoutLayer to remove redundant computation with Billy's algorithm.
 *    The inner product layer computes an inner product with a set of learned weights, and (optionally) adds biases.
 *
 */
template<typename Dtype, typename MItype, typename MOtype>
class DropoutAndInnerProductLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit DropoutAndInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param) {
  }
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "DropoutAndInnerProduct"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 1; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void GenerateProgram();
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
  ///GenerateProgram is a memeber function of Dropout Layer. It is not needed here, I think

  int_tp M_;
  int_tp K_;
  int_tp N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights
  QuantizerValues bias_multiplier_qv_;

  /// rand_vec_ should only be 0 or 1 in CPU mode and GPU's backward pass,
  /// and simulate real numbers in GPU mode's forward pass
  Blob<uint8_t> rand_vec_;
  /// threshold for GPU kernel
  uint8_t uint_thres_;
  /// the probability @f$ p @f$ of dropping any input
  float threshold_;
  /// the scale for undropped inputs at train time @f$ 1 / (1 - p) @f$
  float scale_;
  /// denotes the dimension where the mask is applied
  DropoutType type_;
};

}  // namespace caffe

#endif  //CAFFE_DROPOUT_AND_INNER_PRODUCT_LAYER_HPP_

