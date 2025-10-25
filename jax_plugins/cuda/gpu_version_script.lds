VERS_1.0 {
  global:
    extern "C" {
      GetPjrtApi;
      MosaicGpuCompile;
      MosaicGpuUnload;
    };

  local:
    *;
};
