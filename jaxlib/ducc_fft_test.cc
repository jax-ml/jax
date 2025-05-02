#include <cstdlib>
#include <iostream>
#include <vector>

#include "ducc/google/threading.h"
#include "ducc/src/ducc0/fft/fft.h"
#include "ducc/src/ducc0/fft/fft1d_impl.h"  // IWYU pragma: keep, DUCC definitions.
#include "ducc/src/ducc0/fft/fftnd_impl.h"  // IWYU pragma: keep, DUCC definitions.
#include "ducc/src/ducc0/infra/mav.h"
#include "ducc/src/ducc0/infra/threading.h"
#include "unsupported/Eigen/CXX11/ThreadPool"

int main() {
  std::srand(42);
  const size_t batch_size = 3;
  const size_t in_size = 256;
  const size_t out_size = 383;
  const size_t out_last = out_size / 2 + 1;

  std::vector<float> in;
  in.reserve(batch_size * out_size * out_size);
  std::vector<float> in2;
  in2.reserve(batch_size * out_size * out_size);

  std::vector<std::complex<float>> out;
  out.reserve(batch_size * out_size * out_last);
  std::vector<std::complex<float>> out2;
  out2.reserve(batch_size * out_size * out_last);

  for (size_t idx = 0, b = 0; b < batch_size; ++b) {
    for (size_t i = 0; i < out_size; ++i) {
      for (size_t j = 0; j < in_size; ++j, ++idx) {
        float v = 1.0;
        in[idx] = v;
        in2[idx] = v;
      }
      for (size_t j = 0; j < out_size - in_size; ++j, ++idx) {
        in[idx] = 0.0;
        in2[idx] = 0.0;
      }
    }
  }
  ducc0::cfmav<float> m_in(in.data(), {batch_size, out_size, out_size},
                           {out_size * out_size, out_size, 1});
  ducc0::vfmav<std::complex<float>> m_out(out.data(),
                                          {batch_size, out_size, out_last},
                                          {out_size * out_last, out_last, 1});

  ducc0::cfmav<float> m_in2(in2.data(), {batch_size, out_size, out_size},
                            {out_size * out_size, out_size, 1});
  ducc0::vfmav<std::complex<float>> m_out2(out2.data(),
                                           {batch_size, out_size, out_last},
                                           {out_size * out_last, out_last, 1});

  {
    ducc0::google::NoThreadPool no_thread_pool;
    ducc0::detail_threading::ScopedUseThreadPool thread_pool_guard(
        no_thread_pool);
    ducc0::r2c(m_in, m_out, {1, 2}, true, static_cast<float>(1.0), 1);
  }

  {
    Eigen::ThreadPool* eigen_pool =
        new Eigen::ThreadPool(std::thread::hardware_concurrency());
    ducc0::google::EigenThreadPool eigen_thread_pool(*eigen_pool);
    ducc0::detail_threading::ScopedUseThreadPool thread_pool_guard(
        eigen_thread_pool);
    ducc0::r2c(m_in2, m_out2, {1, 2}, true, static_cast<float>(1.0),
               eigen_thread_pool.nthreads());
  }

  float max_diff = 0.0;
  for (size_t i = 0; i < 1 * out_size * out_last; ++i) {
    std::complex<float> v1 = out[i];
    std::complex<float> v2 = out2[i + (batch_size - 1) * out_size * out_last];
    if (i < 10) {
      std::cout << v1 << " " << v2 << "\n";
    }
    float diff = std::abs(v1.real() - v2.real());
    diff += std::abs(v1.imag() - v2.imag());
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  std::cout << max_diff << "\n";

  return 0;
}
