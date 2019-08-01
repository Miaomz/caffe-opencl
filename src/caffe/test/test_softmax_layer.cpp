#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmax_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_softmax_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SoftmaxLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SoftmaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxLayerTest, TestDtypesFloatAndDevices);

TYPED_TEST(SoftmaxLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxLayer<Dtype, Dtype, Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test sum
  for (int_tp i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int_tp k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int_tp l = 0; l < this->blob_bottom_->width(); ++l) {
        Dtype sum = 0;
        for (int_tp j = 0; j < this->blob_top_->channels(); ++j) {
          sum += this->blob_top_->data_at(i, j, k, l);
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);
        // Test exact values
        Dtype scale = 0;
        for (int_tp j = 0; j < this->blob_bottom_->channels(); ++j) {
          scale += std::exp(this->blob_bottom_->data_at(i, j, k, l));
        }

        const Dtype delta = std::is_same<Dtype, half_fp>::value ?
                      2e-2 : 1e-3;
        for (int_tp j = 0; j < this->blob_bottom_->channels(); ++j) {
          EXPECT_GE(Dtype(this->blob_top_->data_at(i, j, k, l) + delta),
              Dtype(std::exp(this->blob_bottom_->data_at(i, j, k, l)) / scale))
              << "debug: " << i << " " << j;
          EXPECT_LE(Dtype(this->blob_top_->data_at(i, j, k, l) - delta),
              Dtype(std::exp(this->blob_bottom_->data_at(i, j, k, l)) / scale))
              << "debug: " << i << " " << j;
        }
      }
    }
  }
}

TYPED_TEST(SoftmaxLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxLayer<Dtype, Dtype, Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  CuDNNSoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNSoftmaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNSoftmaxLayerTest, TestDtypesFloatNoHalf);

TYPED_TEST(CuDNNSoftmaxLayerTest, TestForwardCuDNN) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
    if(!std::is_same<TypeParam, half_fp>::value) {
      LayerParameter layer_param;
      CuDNNSoftmaxLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
      layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      // Test sum
      for (int_tp i = 0; i < this->blob_bottom_->num(); ++i) {
        for (int_tp k = 0; k < this->blob_bottom_->height(); ++k) {
          for (int_tp l = 0; l < this->blob_bottom_->width(); ++l) {
            TypeParam sum = 0;
            for (int_tp j = 0; j < this->blob_top_->channels(); ++j) {
              sum += this->blob_top_->data_at(i, j, k, l);
            }
            EXPECT_GE(sum, 0.999);
            EXPECT_LE(sum, 1.001);
            // Test exact values
            TypeParam scale = 0;
            for (int_tp j = 0; j < this->blob_bottom_->channels(); ++j) {
              scale += std::exp(this->blob_bottom_->data_at(i, j, k, l));
            }
            for (int_tp j = 0; j < this->blob_bottom_->channels(); ++j) {
              EXPECT_GE((TypeParam)(this->blob_top_->
                        data_at(i, j, k, l) + 1e-3),
                        (TypeParam)(std::exp(this->blob_bottom_->
                        data_at(i, j, k, l)) / scale))
                        << "debug: " << i << " " << j;
              EXPECT_LE((TypeParam)(this->blob_top_->
                        data_at(i, j, k, l) - 1e-3),
                        (TypeParam)(std::exp(this->blob_bottom_->
                        data_at(i, j, k, l)) / scale))
                        << "debug: " << i << " " << j;
            }
          }
        }
      }
    }
  }
}

TYPED_TEST(CuDNNSoftmaxLayerTest, TestGradientCuDNN) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
    if(!std::is_same<TypeParam, half_fp>::value) {
      LayerParameter layer_param;
      CuDNNSoftmaxLayer<TypeParam, TypeParam, TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-2, 1e-3);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
    }
  }
}

#endif

}  // namespace caffe