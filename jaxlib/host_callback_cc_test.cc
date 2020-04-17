/* Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <sstream>

#include "jaxlib/host_callback.h"
#include "testing/base/public/gunit.h"

namespace jax {
namespace {

typedef std::vector<const uint8_t *> Arrays;

template <class T>
const uint8_t *InitializeData(T *data, size_t size_of_data) {
  for (int i = 0; i < size_of_data / sizeof(data[0]); ++i) {
    data[i] = static_cast<T>(i);
  }
  return reinterpret_cast<const uint8_t *>(data);
}

TEST(HostCallbackTest, TestEmpty) {
  PrintMetadata meta{"start", " sep "};
  std::ostringstream oss;
  Arrays empty;
  EmitArrays(oss, meta, empty);
  ASSERT_EQ(oss.str(), "start\n");
}

TEST(HostCallbackTest, TestScalar) {
  PrintMetadata meta{"start", " sep "};
  std::ostringstream oss;
  meta.args_type_and_shape.push_back(
      TypeAndShape{ElementType::I32, 4, Shape{}});
  int data[1];
  Arrays arrays{InitializeData(data, sizeof(data))};
  EmitArrays(oss, meta, arrays);
  ASSERT_EQ(oss.str(), ("start\n"
                        "arg[0]  shape = ()\n"
                        "0"));
}

TEST(HostCallbackTest, TestUnidimensionalArray) {
  PrintMetadata meta{"start", " sep "};
  int constexpr kSize0 = 10;
  meta.args_type_and_shape.push_back(
      TypeAndShape{ElementType::I32, 4, Shape{kSize0}});
  int data[kSize0];
  Arrays arrays{InitializeData(data, sizeof(data))};
  std::ostringstream oss;
  EmitArrays(oss, meta, arrays);
  ASSERT_EQ(oss.str(), ("start\n"
                        "arg[0]  shape = (10, )\n"
                        "[0 1 2 3 4 5 6 7 8 9]\n"));
}

TEST(HostCallbackTest, TestBidimensionalArray) {
  PrintMetadata meta{"start", " sep "};
  int constexpr kSize0 = 2;
  int constexpr kSize1 = 3;
  meta.args_type_and_shape.push_back(
      TypeAndShape{ElementType::F32, 4, Shape{kSize0, kSize1}});
  float data[kSize0 * kSize1];
  Arrays arrays{InitializeData(data, sizeof(data))};
  std::ostringstream oss;
  EmitArrays(oss, meta, arrays);
  ASSERT_EQ(oss.str(), ("start\n"
                        "arg[0]  shape = (2, 3, )\n"
                        "[[0.00 1.00 2.00]\n"
                        " [3.00 4.00 5.00]]\n\n"));
}

TEST(HostCallbackTest, TestSummarize) {
  PrintMetadata meta{"start", " sep "};
  int constexpr kSize0 = 10;
  meta.args_type_and_shape.push_back(
      TypeAndShape{ElementType::I32, 4, Shape{kSize0, kSize0, kSize0}});
  int data[kSize0 * kSize0 * kSize0];
  Arrays arrays{InitializeData(data, sizeof(data))};
  std::ostringstream oss;
  EmitArrays(oss, meta, arrays);
  ASSERT_EQ(oss.str(), R"EOS(start
arg[0]  shape = (10, 10, 10, )
[[[0 1 2 ... 7 8 9]
  [10 11 12 ... 17 18 19]
  [20 21 22 ... 27 28 29]
  ...
  [70 71 72 ... 77 78 79]
  [80 81 82 ... 87 88 89]
  [90 91 92 ... 97 98 99]]

 [[100 101 102 ... 107 108 109]
  [110 111 112 ... 117 118 119]
  [120 121 122 ... 127 128 129]
  ...
  [170 171 172 ... 177 178 179]
  [180 181 182 ... 187 188 189]
  [190 191 192 ... 197 198 199]]

 [[200 201 202 ... 207 208 209]
  [210 211 212 ... 217 218 219]
  [220 221 222 ... 227 228 229]
  ...
  [270 271 272 ... 277 278 279]
  [280 281 282 ... 287 288 289]
  [290 291 292 ... 297 298 299]]

 ...
 [[700 701 702 ... 707 708 709]
  [710 711 712 ... 717 718 719]
  [720 721 722 ... 727 728 729]
  ...
  [770 771 772 ... 777 778 779]
  [780 781 782 ... 787 788 789]
  [790 791 792 ... 797 798 799]]

 [[800 801 802 ... 807 808 809]
  [810 811 812 ... 817 818 819]
  [820 821 822 ... 827 828 829]
  ...
  [870 871 872 ... 877 878 879]
  [880 881 882 ... 887 888 889]
  [890 891 892 ... 897 898 899]]

 [[900 901 902 ... 907 908 909]
  [910 911 912 ... 917 918 919]
  [920 921 922 ... 927 928 929]
  ...
  [970 971 972 ... 977 978 979]
  [980 981 982 ... 987 988 989]
  [990 991 992 ... 997 998 999]]]

)EOS");
}

TEST(HostCallbackTest, TestTwoArrays) {
  PrintMetadata meta{"start", " sep "};
  std::ostringstream oss;
  meta.args_type_and_shape.push_back(
      TypeAndShape{ElementType::I32, 4, Shape{4}});
  meta.args_type_and_shape.push_back(
      TypeAndShape{ElementType::I32, 4, Shape{4}});
  int data[4];
  auto data_ptr = InitializeData(data, sizeof(data));
  Arrays arrays{data_ptr, data_ptr};
  EmitArrays(oss, meta, arrays);
  ASSERT_EQ(oss.str(), R"EOS(start
arg[0]  shape = (4, )
[0 1 2 3]
 sep arg[1]  shape = (4, )
[0 1 2 3]
)EOS");
}

}  // namespace
}  // namespace jax
