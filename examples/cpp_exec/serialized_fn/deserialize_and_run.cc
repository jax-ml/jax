// Test code to deserialize an serialized JAX executable and run it.
//
// Typical usage:
//
//$ bazel run -c opt examples/cpp_exec/serialized_fn:deserialize_and_run -- /tmp/serialized_fn

#include <fstream>
#include <ios>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api_cpu_internal.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

#define RETURN_IF_NOT_OK(status)                                       \
  {                                                                    \
    if (!status.ok()) {                                                \
      std::cerr << "Failed to " << #status << ": " << status.message() \
                << std::endl;                                          \
      return 1;                                                        \
    }                                                                  \
  }

std::string ReadFileToString(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  std::stringstream buffer;

  if (file.is_open()) {
    buffer << file.rdbuf();
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filename << std::endl;
  }

  return buffer.str();
}

int main(int argc, char** argv) {
  const std::string device_type = "cpu";
  auto status = ::pjrt::PjrtApi(device_type);
  if (!status.ok()) {
    RETURN_IF_NOT_OK(
        ::pjrt::SetPjrtApi(device_type, ::pjrt::cpu_plugin::GetCpuPjrtApi()));
  }

  auto c_api_client = xla::GetCApiClient(device_type).value();

  const std::string serialized = ReadFileToString(argv[1]);
  std::cout << "Successfully read " << serialized.size() << " bytes from file"
            << std::endl;
  auto maybe_executable =
      c_api_client->DeserializeExecutable(serialized, std::nullopt);
  RETURN_IF_NOT_OK(maybe_executable.status());
  auto executable = std::move(maybe_executable.value());

  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {1});

  std::vector<float> data_1(1, 3.4);
  auto buffer_1 = c_api_client->BufferFromHostBuffer(
          data_1.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          nullptr, c_api_client->addressable_devices()[0]);
  RETURN_IF_NOT_OK(buffer_1.status())

  std::vector<float> data_2(1, 8.4);

  auto buffer_2 = c_api_client->BufferFromHostBuffer(
          data_2.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          nullptr, c_api_client->addressable_devices()[0]);
  RETURN_IF_NOT_OK(buffer_2.status());

  auto result =
      executable->Execute({{buffer_1.value().get(), buffer_2.value().get()}},
                          xla::ExecuteOptions());
  RETURN_IF_NOT_OK(result.status());
  std::cout << "Successfully deserialized executable and executed it, "
            << "Result is: " << *(result.value()[0][0]->ToLiteralSync().value())
            << std::endl;
  return 0;
}
