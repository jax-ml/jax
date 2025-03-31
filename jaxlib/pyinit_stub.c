// Stub that reexports Wrapped_PyInit_module as PyInit_module.

extern void* Wrapped_PyInit_@MODULE_NAME@();

#if defined(WIN32) || defined(_WIN32)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __attribute__ ((visibility("default")))
#endif

EXPORT_SYMBOL void* PyInit_@MODULE_NAME@() {
  return Wrapped_PyInit_@MODULE_NAME@();
}
